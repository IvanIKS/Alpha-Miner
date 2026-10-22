"""Checks for the LLM generator (llm_generator.py), the model adapters (llm_models.py) and
the shared verifier (lab.py). No network and no API key: the model is scripted, mocked, or
a local HTTP server that imitates the provider.

Run: pytest -q test_llm_generator.py
"""
import contextlib
import http.server
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import app.strategies.genetic.llm.lab as lab
import app.strategies.genetic.llm.llm_generator as lg
import app.strategies.genetic.llm.llm_models as lm
import app.strategies.genetic.strategy_gp as sg

DATA = lab.synthetic(seed=7, edge=lab.EDGE, n=6000)
N_SEARCH = 5000
RSI = sg.CLASSICS["rsi_reversion"]       # long_flat(osc_below(rsi(close, 14), 30), osc_above(rsi(close, 14), 70))


class ScriptedModel(lm.ChatModel):
    """Answers with prepared texts in order (then with an empty list) and records every call."""

    name = "scripted"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, system, messages, seed=None):
        self.calls.append((system, list(messages)))
        return lm.Completion(self.replies.pop(0) if self.replies else "[]", input_tokens=100, output_tokens=10)


def proposal(rule, layer="exit", level="bridge", **expected):
    return {"rule": rule, "layer": layer, "level": level, "mechanism": "test", "expected": expected or {"trades": "same"}}


# ------------------------------------------------------------ the reply --
def test_parse_reply_tolerates_fences_and_prose():
    items = [proposal(RSI, "both", "repair", trades="up", holding="sideways")]
    for text in (json.dumps(items), f"```json\n{json.dumps(items)}\n```", f"Proposals:\n{json.dumps(items)}\nDone."):
        (p,) = lg.parse_reply(text)
        assert (p.rule, p.layer, p.level, p.expected) == (RSI, "both", "repair", {"trades": "up"})
    assert lg.parse_reply("[]") == []
    (bad,) = lg.parse_reply('[{"layer": "exit"}]')
    assert bad.problem and not bad.rule
    with pytest.raises(ValueError):
        lg.parse_reply("RULE: long_flat(osc_below(ibs, 20), osc_above(ibs, 80))")


# ------------------------------------------------------------ semantics --
@pytest.mark.parametrize("name, family", [
    ("rsi_reversion", "reversion"), ("ma_cross", "trend"), ("donchian_breakout", "breakout"),
    ("bollinger_reversion", "reversion"), ("momentum", "trend"), ("rsi_with_trend_filter", "reversion"),
    ("volatility_breakout", "volatility"), ("volume_spike", "volume"), ("ibs_reversion", "reversion"),
    ("stochastic", "reversion"),
])
def test_family_of_classics(name, family):
    assert lg.family(sg.parse(sg.CLASSICS[name])) == family


def test_edit_levels_follow_the_diff():
    base = sg.parse(RSI)
    cases = {
        "long_flat(osc_below(rsi(close, 10), 25), osc_above(rsi(close, 10), 70))": ("repair", ("entry", "exit")),
        "long_flat(osc_below(rsi(close, 14), 30), osc_above(stoch(14), 70))": ("bridge", ("exit",)),
        "long_flat(and_(osc_below(rsi(close, 14), 30), gt(close, sma(close, 200))), osc_above(rsi(close, 14), 70))":
            ("bridge", ("entry",)),
        "long_flat(osc_below(stoch(20), 10), gt(close, sma(close, 20)))": ("redesign", ("entry", "exit")),
        "long_flat(cross_up(sma(close, 20), sma(close, 100)), osc_above(rsi(close, 14), 70))": ("migration", ("entry",)),
    }
    for text, (level, changed) in cases.items():
        edit = lg.analyse_edit(base, sg.parse(text))
        assert (edit.level, edit.layers) == (level, changed), text
        assert 0.0 <= edit.syntactic <= 1.0


def test_contract_names_what_is_wrong():
    base = sg.parse(RSI)
    redesign = sg.parse("long_flat(osc_below(stoch(20), 10), gt(close, sma(close, 20)))")
    problems = lg.contract_problems(lg.Proposal(sg.to_text(redesign), "exit", "bridge"),
                                    lg.analyse_edit(base, redesign), lg.Plan("bridge", ("exit",), ()), arm=1)
    assert any("redesign" in p and "plan allows bridge" in p for p in problems)
    assert any("declares exit" in p for p in problems)
    untied = sg.parse("long_flat(osc_below(rsi(close, W1), LVL1), osc_above(rsi(close, W2), LVL2))")
    template = sg.lift(base)[0]
    problems = lg.contract_problems(lg.Proposal(sg.to_text(untied), "both", "repair"),
                                    lg.analyse_edit(template, untied), lg.Plan("bridge", lg.LAYERS, ()), arm=3)
    assert problems == ["numbers are tuned by the optimizer; change the structure instead"]
    assert lg.contract_problems(lg.Proposal(RSI, "both", "repair"), lg.analyse_edit(base, base),
                                lg.Plan("repair", lg.LAYERS, ()), arm=1) == ["the rule is the current rule itself"]


# ------------------------------------------------------------ diagnosis --
def _stats(**changes):
    out = {"sharpe": 0.5, "gross": 0.6, "trades": 100, "exposure": 0.2, "holding": 20.0, "mdd": 0.1, "ret": 0.2}
    out.update(changes)
    return out


def test_diagnosis_flags():
    ev = {"train": _stats(), "val": _stats(), "giveback": 0.2, "against_trend": 0.3, "jitter_drop": 0.1}
    assert lg.diagnose(ev) == []
    assert lg.diagnose({**ev, "train": _stats(trades=5)}) == ["sparse_entry"]
    assert "overfit_validation" in lg.diagnose({**ev, "train": _stats(sharpe=2.5, gross=2.6), "val": _stats(sharpe=0.2)})
    assert "cost_drag" in lg.diagnose({**ev, "train": _stats(sharpe=-0.3, gross=0.8)})
    assert lg.diagnose({**ev, "giveback": 0.8}) == ["delayed_exit"]
    assert lg.diagnose({**ev, "against_trend": 0.9}) == ["low_trend_filter"]
    assert lg.diagnose({**ev, "jitter_drop": 2.0}) == ["parameter_sensitivity"]
    assert lg.diagnose({**ev, "train": _stats(mdd=0.4, ret=0.1)}) == ["excessive_drawdown"]
    many = {**ev, "train": _stats(trades=5, sharpe=-1.0, gross=1.0, mdd=0.5, ret=-0.2), "giveback": 0.9,
            "against_trend": 0.9, "jitter_drop": 3.0}
    assert len(lg.diagnose(many)) == 2


def test_trade_diagnostics_measure_give_back():
    pos = np.array([0, 1, 1, 1, 0] * 12, dtype=float)
    close = np.array([100, 100, 110, 105, 100] * 12, dtype=float)    # each trade gains 10 % and gives it all back
    giveback, against = lg.trade_diagnostics(pos, close, slice(0, len(pos)), trend_window=3)
    assert giveback == pytest.approx(1.0) and against == 0.0           # no losing trades


def test_actual_moves_compare_with_the_current_rule():
    old = {"stats": {"search": {"trades": 100, "holding": 10.0, "exposure": 0.2}, "val": {"sharpe": 0.5}}}
    new = {"stats": {"search": {"trades": 150, "holding": 10.5, "exposure": 0.1}, "val": {"sharpe": 0.55}}}
    assert lg.actual_moves(old, new) == {"trades": "up", "holding": "same", "exposure": "down", "val_sharpe": "same"}


# ------------------------------------------------------------- verifier --
def test_adoption_margin_grows_when_rules_behave_differently():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0002, 0.01, 6000)
    near = a + rng.normal(0, 0.001, 6000)
    other = rng.normal(0.0002, 0.01, 6000)
    assert lab.sharpe_diff_se(a, a) < 1e-6
    assert lab.sharpe_diff_se(a, near) < lab.sharpe_diff_se(a, other)


def test_ledger_windows_and_probe_budget():
    ledger = lab.Ledger(DATA, budget=40, n_search=N_SEARCH)
    row = ledger.backtest(sg.parse(RSI))
    s = row["stats"]
    assert s["train"]["trades"] + s["val"]["trades"] - s["search"]["trades"] in (0, 1)   # a trade open at the split counts twice
    lab.Verifier(ledger).verify(row)
    probes = [r for r in ledger.rows.values() if r["role"] == "probe"]
    assert ledger.used == 1 + len(probes) and all(r["charged"] for r in probes)
    used = ledger.used
    chosen = lab.select_final(ledger)
    assert ledger.used == used                                        # the final choice spends nothing
    assert chosen is None or chosen[0]["role"] != "probe"


# ------------------------------------------------------------ generator --
def test_round_repairs_parse_errors_contract_breaks_and_repeats():
    stoch_exit = "long_flat(osc_below(rsi(close, 14), 30), osc_above(stoch(14), 70))"
    first = json.dumps([
        proposal(stoch_exit, "exit", "bridge"),
        proposal("long_flat(osc_below(ema(close, 14), 30), osc_above(rsi(close, 14), 70))", "entry", "repair"),
        proposal("long_flat(and_(osc_below(rsi(close, 14), 30), gt(close, sma(close, 200))), "
                 "osc_above(rsi(close, 14), 70))", "exit", "bridge"),
        proposal(stoch_exit, "exit", "bridge"),
    ])
    second = json.dumps([proposal("long_flat(osc_below(rsi(close, 12), 30), osc_above(rsi(close, 12), 70))",
                                  "both", "repair")])
    model = ScriptedModel([first, second])
    g = lg.Generator(1, model, DATA, budget=100, n_search=N_SEARCH, lines=("rsi_reversion",),
                     first_level="bridge", max_calls=2).run()
    s = g.stats
    assert (s["proposals"], s["invalid"], s["contract"], s["repeats"], s["tested"]) == (5, 1, 1, 1, 2)
    system, messages = model.calls[1]
    assert sg.catalog() in system and "JSON list" in system
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    repair = messages[-1].content
    assert "unknown name 'ema'" in repair and "declares exit" in repair and "already proposed" in repair
    assert repair.count("\n- `") == 3                                 # only the fixable ones go back
    assert g.ledger.used <= 100 and g.stop_reason == "call limit"
    assert all(e["status"] != "cut by budget" for e in g.events)


def test_arm3_tunes_templates_and_leaves_numbers_to_the_optimizer():
    stoch_exit = "long_flat(osc_below(rsi(close, W1), LVL1), osc_above(stoch(W2), LVL2))"
    untied = "long_flat(osc_below(rsi(close, W1), LVL1), osc_above(rsi(close, W2), LVL2))"
    model = ScriptedModel([json.dumps([proposal(stoch_exit, "exit", "bridge"), proposal(untied, "both", "repair")])])
    g = lg.Generator(3, model, DATA, budget=120, k=4, n_search=N_SEARCH, lines=("rsi_reversion",), max_calls=2).run()
    assert (g.stats["tested"], g.stats["contract"]) == (1, 1)
    prompt = model.calls[0][1][0].content
    assert "Current template: long_flat(osc_below(rsi(close, W1), LVL1), osc_above(rsi(close, W1), LVL2))" in prompt
    assert "Tuned values: W1=" in prompt
    assert "placeholders" in model.calls[0][0]
    assert "optimizer" in model.calls[1][1][-1].content
    trials = [r for r in g.ledger.rows.values() if r["role"] == "trial"]
    assert 0 < len(trials) <= 2 * 4                                   # the classic and the proposal, k trials each


@pytest.mark.parametrize("arm", [1, 3, 4])
def test_mock_model_drives_every_arm_within_budget(arm):
    g = lg.Generator(arm, lg.MockModel(arm, seed=1), DATA, budget=80, k=3, n_search=N_SEARCH,
                     lines=("rsi_reversion", "stochastic"), max_calls=30).run()
    assert g.ledger.used <= 80
    assert g.stop_reason in ("budget", "call limit", "no new backtests")
    assert g.events and all("status" in e for e in g.events)
    assert "calls" in g.summary() and g.best() is not None
    if arm == 4:
        assert g.search_used <= 80 - round(0.2 * 80)


def test_recorded_model_replays_a_run_exactly(tmp_path):
    log = tmp_path / "calls.jsonl"

    def run(model, **options):
        g = lg.Generator(1, model, DATA, budget=60, n_search=N_SEARCH, lines=("rsi_reversion",), max_calls=6,
                         **options).run()
        return [(e["rule"], e["status"]) for e in g.events], sorted(g.ledger.rows)

    first = run(lm.RecordedModel(lg.MockModel(1, seed=3), log))
    assert first[0]
    assert run(lm.RecordedModel(None, log, mode="replay")) == first
    with pytest.raises(lm.ReplayMismatch):                            # another prompt: the log no longer fits
        run(lm.RecordedModel(None, log, mode="replay"), candidates=3)


# ------------------------------------------------------------- adapters --
class _Endpoint:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_claude_adapter_builds_the_request_and_reads_text_blocks():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="", signature="s"), SimpleNamespace(type="text", text="[]")],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7, cache_read_input_tokens=5, cache_creation_input_tokens=0),
        stop_reason="end_turn")
    endpoint = _Endpoint(response)
    model = lm.ClaudeModel("claude-sonnet-5-5", effort="low", client=SimpleNamespace(messages=endpoint))
    out = model.complete("SYSTEM", [lm.Message("user", "hello")], seed=3)
    assert (out.text, out.input_tokens, out.output_tokens, out.cache_read_tokens) == ("[]", 11, 7, 5)
    kw = endpoint.kwargs
    assert kw["model"] == "claude-sonnet-5-5" and kw["messages"] == [{"role": "user", "content": "hello"}]
    assert kw["system"] == [{"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}]
    assert kw["extra_body"] == {"output_config": {"effort": "low"}}
    assert not {"temperature", "top_p", "top_k", "seed"} & set(kw)


def test_openai_adapter_builds_the_request():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[]"), finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=9, completion_tokens=3))
    endpoint = _Endpoint(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    model = lm.OpenAICompatibleModel("local-model", temperature=0.5, max_tokens=100, client=client)
    out = model.complete("SYSTEM", [lm.Message("user", "hello")], seed=7)
    assert (out.text, out.input_tokens, out.output_tokens, out.stop_reason) == ("[]", 9, 3, "stop")
    kw = endpoint.kwargs
    assert kw["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "hello"}]
    assert (kw["temperature"], kw["max_tokens"], kw["seed"]) == (0.5, 100, 7) and "extra_body" not in kw


@contextlib.contextmanager
def _local_server(reply):
    """A one-route HTTP server that answers every POST with `reply` and records the requests."""
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            seen.append((self.path, json.loads(body)))
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()


def test_claude_adapter_through_the_sdk():
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")
    reply = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
             "content": [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "[]"}],
             "stop_reason": "end_turn", "stop_sequence": None,
             "usage": {"input_tokens": 12, "output_tokens": 3, "cache_creation_input_tokens": 0,
                       "cache_read_input_tokens": 8}}
    with _local_server(reply) as (url, seen):
        client = anthropic.Anthropic(api_key="test", base_url=url, max_retries=0,
                                     http_client=httpx.Client(trust_env=False))
        out = lm.ClaudeModel(client=client).complete("SYSTEM", [lm.Message("user", "hello")])
    path, body = seen[0]
    assert path.endswith("/v1/messages") and out.text == "[]" and out.cache_read_tokens == 8
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["output_config"] == {"effort": "low"} and "temperature" not in body


def test_openai_adapter_through_the_sdk():
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    reply = {"id": "c1", "object": "chat.completion", "created": 0, "model": "local",
             "choices": [{"index": 0, "message": {"role": "assistant", "content": "[]"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}}
    with _local_server(reply) as (url, seen):
        client = openai.OpenAI(api_key="test", base_url=url + "/v1", max_retries=0,
                               http_client=httpx.Client(trust_env=False))
        out = lm.OpenAICompatibleModel("local", client=client).complete("SYSTEM", [lm.Message("user", "hi")], seed=5)
    path, body = seen[0]
    assert path.endswith("/v1/chat/completions") and out.text == "[]" and out.input_tokens == 9
    assert body["messages"][0] == {"role": "system", "content": "SYSTEM"} and body["seed"] == 5
