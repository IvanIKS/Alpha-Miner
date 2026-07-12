import { type FormEvent, useEffect, useMemo, useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { fetchOHLCVCoverage, triggerBackfill, type BackfillJob, type OHLCVCoverageEntry } from '../services/api'

const TF_ORDER = ['1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d','1w','1M']
const TF_COLORS: Record<string, string> = {
  '1m':'#f59e0b','3m':'#fb923c','5m':'#10b981','15m':'#34d399','30m':'#3b82f6',
  '1h':'#818cf8','2h':'#a78bfa','4h':'#c084fc','6h':'#f472b6','8h':'#fb7185',
  '12h':'#ef4444','1d':'#06b6d4','3d':'#0ea5e9','1w':'#84cc16','1M':'#22c55e',
}

function tfColor(tf: string) { return TF_COLORS[tf] ?? '#94a3b8' }

function sortTfs(tfs: string[]) {
  return [...tfs].sort((a, b) => {
    const ia = TF_ORDER.indexOf(a), ib = TF_ORDER.indexOf(b)
    return (ia < 0 ? 999 : ia) - (ib < 0 ? 999 : ib)
  })
}

const TODAY = new Date().toISOString().slice(0, 10)

function BackfillSection({
  allSymbols,
  allTfs,
  defaultSymbol,
}: {
  allSymbols: string[]
  allTfs: string[]
  defaultSymbol: string
}) {
  const { token } = useAuth()
  const [symbol, setSymbol] = useState(defaultSymbol)
  const [exchange, setExchange] = useState('binance')
  const [timeframe, setTimeframe] = useState(allTfs[0] ?? '1m')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState(TODAY)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [jobs, setJobs] = useState<BackfillJob[]>([])

  // Keep symbol in sync when allSymbols first load
  useEffect(() => {
    if (!symbol && defaultSymbol) setSymbol(defaultSymbol)
  }, [defaultSymbol])

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!token || !symbol || !dateFrom) return
    setSubmitting(true)
    setError(null)
    try {
      const job = await triggerBackfill(token, {
        symbol,
        exchange,
        timeframe,
        startTime: `${dateFrom}T00:00:00Z`,
        endTime: `${dateTo}T23:59:59Z`,
      })
      setJobs((prev) => [job, ...prev])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to queue backfill')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="card">
      <h2 className="section-title">Backfill Data</h2>
      <form onSubmit={handleSubmit} className="backfill-form">
        <datalist id="known-symbols">
          {allSymbols.map((s) => <option key={s} value={s} />)}
        </datalist>

        <label className="field">
          <span className="field-label">Symbol</span>
          <input
            list="known-symbols"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            placeholder="BTC/USDT"
            required
          />
        </label>

        <label className="field">
          <span className="field-label">Exchange</span>
          <input
            value={exchange}
            onChange={(e) => setExchange(e.target.value)}
            placeholder="binance"
            required
          />
        </label>

        <label className="field">
          <span className="field-label">Timeframe</span>
          <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
            {(allTfs.length > 0 ? allTfs : TF_ORDER).map((tf) => (
              <option key={tf} value={tf}>{tf}</option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="field-label">From</span>
          <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} required />
        </label>

        <label className="field">
          <span className="field-label">To</span>
          <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} required />
        </label>

        <div className="field backfill-submit-field">
          <button type="submit" className="primary-button" disabled={submitting || !symbol || !dateFrom}>
            {submitting ? 'Queuing…' : 'Queue Backfill'}
          </button>
        </div>
      </form>

      {error && (
        <p style={{ margin: '0.75rem 0 0', color: '#b91c1c', fontSize: '0.875rem' }}>{error}</p>
      )}

      {jobs.length > 0 && (
        <div style={{ marginTop: '1rem' }}>
          <p className="db-filter-title" style={{ marginBottom: '0.5rem' }}>Queued jobs</p>
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Timeframe</th>
                  <th>From</th>
                  <th>To</th>
                  <th>Job ID</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.jobId}>
                    <td>{job.exchange} · {job.symbol}</td>
                    <td><span className="db-tf-dot" style={{ background: tfColor(job.timeframe), display: 'inline-block', marginRight: 4, verticalAlign: 'middle' }} />{job.timeframe}</td>
                    <td style={{ color: '#6b7280' }}>{job.startTimestamp.slice(0, 10)}</td>
                    <td style={{ color: '#6b7280' }}>{job.endTimestamp.slice(0, 10)}</td>
                    <td style={{ fontFamily: 'monospace', fontSize: '0.78rem', color: '#6b7280' }}>{job.jobId.slice(0, 12)}…</td>
                    <td>
                      <span className="backfill-status-badge">{job.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

const LABEL_W = 148
const HDR_H = 22
const BAR_H = 32
const ROW_GAP = 3
const AXIS_H = 26
const PAD_R = 10
const SVG_W = 880

type Item =
  | { kind: 'hdr'; id: string; label: string }
  | { kind: 'bar'; id: string; tf: string; from: number; to: number }

type RenderedItem = Item & { y: number }

export function DatabaseMonitorScreen() {
  const { token } = useAuth()
  const [entries, setEntries] = useState<OHLCVCoverageEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selSymbols, setSelSymbols] = useState<Set<string>>(new Set())
  const [selTfs, setSelTfs] = useState<Set<string>>(new Set())

  async function load() {
    if (!token) return
    setLoading(true)
    setError(null)
    try {
      const data = await fetchOHLCVCoverage(token)
      setEntries(data)
      setSelSymbols(new Set(data.map((e) => e.symbol)))
      setSelTfs(new Set(data.map((e) => e.timeframe)))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [token])

  const allSymbols = useMemo(() => [...new Set(entries.map((e) => e.symbol))].sort(), [entries])
  const allTfs = useMemo(() => sortTfs([...new Set(entries.map((e) => e.timeframe))]), [entries])
  const multiExchange = useMemo(() => new Set(entries.map((e) => e.exchange)).size > 1, [entries])

  const items = useMemo<Item[]>(() => {
    const result: Item[] = []
    for (const sym of allSymbols) {
      if (!selSymbols.has(sym)) continue
      const visible = entries.filter((e) => e.symbol === sym && selTfs.has(e.timeframe))
      if (visible.length === 0) continue
      const exch = visible[0].exchange
      result.push({ kind: 'hdr', id: sym, label: multiExchange ? `${exch} · ${sym}` : sym })
      for (const tf of allTfs) {
        if (!selTfs.has(tf)) continue
        const e = entries.find((en) => en.symbol === sym && en.timeframe === tf)
        if (!e) continue
        result.push({
          kind: 'bar',
          id: `${sym}/${tf}`,
          tf,
          from: new Date(e.fromDate).getTime(),
          to: new Date(e.toDate).getTime(),
        })
      }
    }
    return result
  }, [entries, allSymbols, allTfs, selSymbols, selTfs, multiExchange])

  const renderedItems = useMemo<RenderedItem[]>(() => {
    let y = AXIS_H
    return items.map((item) => {
      const ri = { ...item, y }
      y += (item.kind === 'hdr' ? HDR_H : BAR_H) + ROW_GAP
      return ri
    })
  }, [items])

  const svgH = useMemo(() => {
    const last = renderedItems[renderedItems.length - 1]
    if (!last) return AXIS_H + 40
    return last.y + (last.kind === 'hdr' ? HDR_H : BAR_H) + 10
  }, [renderedItems])

  const globalMin = useMemo(
    () => renderedItems.reduce((m, it) => (it.kind === 'bar' ? Math.min(m, it.from) : m), Infinity),
    [renderedItems],
  )
  const globalMax = useMemo(
    () => renderedItems.reduce((m, it) => (it.kind === 'bar' ? Math.max(m, it.to) : m), -Infinity),
    [renderedItems],
  )
  const spanMs = isFinite(globalMin) && isFinite(globalMax) ? Math.max(globalMax - globalMin, 1) : 1
  const chartW = SVG_W - LABEL_W - PAD_R

  function xOf(t: number) { return LABEL_W + ((t - globalMin) / spanMs) * chartW }

  const ticks = useMemo(() => {
    if (!isFinite(globalMin) || !isFinite(globalMax)) return []
    const result: { label: string; frac: number }[] = []
    const rangeYears = spanMs / (365.25 * 24 * 3600e3)
    const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    const start = new Date(globalMin)

    if (rangeYears > 2) {
      for (let y = start.getFullYear(); ; y++) {
        const t = new Date(y, 0, 1).getTime()
        if (t > globalMax) break
        result.push({ label: String(y), frac: (t - globalMin) / spanMs })
      }
    } else if (rangeYears > 0.4) {
      let d = new Date(start.getFullYear(), Math.floor(start.getMonth() / 3) * 3, 1)
      while (d.getTime() <= globalMax) {
        if (d.getTime() >= globalMin)
          result.push({ label: `${MONTHS[d.getMonth()]} ${d.getFullYear()}`, frac: (d.getTime() - globalMin) / spanMs })
        d = new Date(d.getFullYear(), d.getMonth() + 3, 1)
      }
    } else {
      let d = new Date(start.getFullYear(), start.getMonth(), 1)
      while (d.getTime() <= globalMax) {
        if (d.getTime() >= globalMin)
          result.push({ label: MONTHS[d.getMonth()], frac: (d.getTime() - globalMin) / spanMs })
        d = new Date(d.getFullYear(), d.getMonth() + 1, 1)
      }
    }
    return result
  }, [globalMin, globalMax, spanMs])

  function toggleSet<T>(set: Set<T>, val: T): Set<T> {
    const next = new Set(set)
    next.has(val) ? next.delete(val) : next.add(val)
    return next
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Database Monitor</h1>
        <button type="button" className="secondary-button" onClick={load} disabled={loading}>
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div className="card" style={{ color: '#b91c1c', borderColor: 'rgba(239,68,68,0.4)' }}>
          {error}
        </div>
      )}

      <div className="card" style={{ padding: '1rem', overflow: 'hidden' }}>
        {loading ? (
          <p className="muted" style={{ margin: '0.5rem 0' }}>Loading coverage data…</p>
        ) : renderedItems.length === 0 ? (
          <p className="muted" style={{ margin: '0.5rem 0' }}>
            No data to display — select at least one asset and one timeframe below.
          </p>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <svg
              viewBox={`0 0 ${SVG_W} ${svgH}`}
              style={{ display: 'block', width: '100%', minWidth: 480 }}
              aria-label="OHLCV coverage timeline"
            >
              {/* X-axis grid lines + tick labels */}
              {ticks.map((tick) => {
                const cx = LABEL_W + tick.frac * chartW
                return (
                  <g key={`${tick.label}-${tick.frac}`}>
                    <line
                      x1={cx} y1={AXIS_H - 2} x2={cx} y2={svgH - 6}
                      stroke="rgba(148,163,184,0.3)" strokeWidth={1} strokeDasharray="3 3"
                    />
                    <text x={cx} y={13} textAnchor="middle" fontSize={10} fill="#9ca3af"
                      fontFamily="system-ui, sans-serif">
                      {tick.label}
                    </text>
                  </g>
                )
              })}

              {/* Items */}
              {renderedItems.map((item) => {
                if (item.kind === 'hdr') {
                  return (
                    <g key={item.id}>
                      <rect x={0} y={item.y} width={SVG_W} height={HDR_H}
                        fill="rgba(250,204,21,0.08)" />
                      <text
                        x={LABEL_W - 8} y={item.y + HDR_H - 6}
                        textAnchor="end" fontSize={11} fontWeight={700}
                        fill="#374151" fontFamily="system-ui, sans-serif"
                      >
                        {item.label}
                      </text>
                      <line
                        x1={LABEL_W} y1={item.y + HDR_H / 2}
                        x2={SVG_W - PAD_R} y2={item.y + HDR_H / 2}
                        stroke="rgba(148,163,184,0.2)" strokeWidth={1}
                      />
                    </g>
                  )
                }

                const color = tfColor(item.tf)
                const x1 = xOf(item.from)
                const x2 = xOf(item.to)
                const bw = Math.max(x2 - x1, 3)
                const mid = item.y + BAR_H / 2
                const fromStr = new Date(item.from).toISOString().slice(0, 10)
                const toStr = new Date(item.to).toISOString().slice(0, 10)

                return (
                  <g key={item.id}>
                    {/* track */}
                    <rect x={LABEL_W} y={item.y + 2} width={chartW} height={BAR_H - 4}
                      fill="rgba(249,250,251,0.7)" rx={3} />
                    {/* data bar */}
                    <rect x={x1} y={item.y + 5} width={bw} height={BAR_H - 10}
                      fill={color} opacity={0.85} rx={3}>
                      <title>{item.id} · {fromStr} → {toStr}</title>
                    </rect>
                    {/* tf label inside bar */}
                    {bw > 28 && (
                      <text
                        x={x1 + bw / 2} y={mid + 4}
                        textAnchor="middle" fontSize={9} fill="white" fontWeight={700}
                        fontFamily="system-ui, sans-serif" style={{ pointerEvents: 'none' }}
                      >
                        {item.tf}
                      </text>
                    )}
                    {/* dot + label on left */}
                    <circle cx={10} cy={mid} r={4} fill={color} />
                    <text x={18} y={mid + 4} fontSize={10} fill="#4b5563"
                      fontWeight={600} fontFamily="system-ui, sans-serif">
                      {item.tf}
                    </text>
                  </g>
                )
              })}
            </svg>
          </div>
        )}
      </div>

      {/* Backfill section */}
      {!loading && (
        <BackfillSection
          allSymbols={allSymbols}
          allTfs={allTfs}
          defaultSymbol={allSymbols[0] ?? ''}
        />
      )}

      {/* Filter panel */}
      {!loading && entries.length > 0 && (
        <div className="card db-filter-panel">
          <div className="db-filter-group">
            <span className="db-filter-title">Assets</span>
            <div className="db-filter-chips">
              {allSymbols.map((sym) => (
                <label key={sym} className="db-filter-chip">
                  <input
                    type="checkbox"
                    checked={selSymbols.has(sym)}
                    onChange={() => setSelSymbols((p) => toggleSet(p, sym))}
                  />
                  <span>{sym}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="db-filter-sep" />

          <div className="db-filter-group">
            <span className="db-filter-title">Timeframes</span>
            <div className="db-filter-chips">
              {allTfs.map((tf) => (
                <label key={tf} className="db-filter-chip">
                  <input
                    type="checkbox"
                    checked={selTfs.has(tf)}
                    onChange={() => setSelTfs((p) => toggleSet(p, tf))}
                  />
                  <span className="db-tf-dot" style={{ background: tfColor(tf) }} />
                  <span>{tf}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
