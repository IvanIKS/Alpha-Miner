import app.strategies.genetic.llm.lab as lab
import app.strategies.genetic.llm.llm_generator as llm

def llm_optuna_optimisation_finalists_only(data, model, seed=0, budget=2400, *, candidates=4, n_search=lab.N_SEARCH, **options):
    generator = llm.Generator(4, model, data, budget, candidates=candidates, seed=seed,
                              n_search=n_search, **options)
    generator.run()
    return generator.ledger, generator