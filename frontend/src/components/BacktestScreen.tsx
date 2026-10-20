import { useEffect, useRef, useState, type SetStateAction} from 'react'
import {useAuth} from '../contexts/AuthContext'
import type {BacktestMetrics, BacktestRecord, StrategyParameterSchema, StrategySummary} from '../services/api'
import {
  fetchBacktestStatus,
  fetchBacktests,
  fetchEquityCurve,
  fetchStrategies,
  fetchStrategyParameters,
  runBacktest,
} from '../services/api'

function getParameterDefaultValue(param: StrategyParameterSchema): unknown {
  if (param.options && param.options.length > 0) {
    if (typeof param.default === 'string' && param.options.includes(param.default)) {
      return param.default
    }

    return param.options[0]
  }

  if (param.type === 'bool') {
    return typeof param.default === 'boolean' ? param.default : false
  }

  if (param.type === 'string') {
    return typeof param.default === 'string' ? param.default : ''
  }

  return typeof param.default === 'number' && Number.isFinite(param.default) ? param.default : ''
}

function buildParameterValues(schema: StrategyParameterSchema[]): Record<string, unknown> {
  return schema.reduce<Record<string, unknown>>((values, param) => {
    values[param.key] = getParameterDefaultValue(param)
    return values
  }, {})
}

function StrategyParamsForm({
                              schema,
                              values,
                              onChange,
                            }: {
  schema: StrategyParameterSchema[]
  values: Record<string, unknown>
  onChange: (key: string, value: unknown) => void
}) {
  return (
      <>
        {schema.map((param) => (
            <label key={param.key} className="field">
              <span className="field-label">{param.label}</span>
              {param.type === 'bool' ? (
                  <input
                      type="checkbox"
                      checked={Boolean(values[param.key] ?? getParameterDefaultValue(param))}
                      onChange={(event) => onChange(param.key, event.target.checked)}
                  />
              ) : param.options ? (
                  <select
                      value={String(values[param.key] ?? getParameterDefaultValue(param) ?? '')}
                      onChange={(event) => onChange(param.key, event.target.value)}
                  >
                    {param.options.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                    ))}
                  </select>
              ) : param.type === 'string' ? (
                  <input
                      type="text"
                      value={String(values[param.key] ?? getParameterDefaultValue(param) ?? '')}
                      onChange={(event) => onChange(param.key, event.target.value)}
                  />
              ) : (
                  <input
                      type="number"
                      min={param.min ?? undefined}
                      max={param.max ?? undefined}
                      step={param.type === 'float' ? '0.01' : 1}
                      value={String(values[param.key] ?? getParameterDefaultValue(param) ?? '')}
                      onChange={(event) => {
                        const rawValue = event.target.value
                        const nextValue = param.type === 'int' ? Number.parseInt(rawValue, 10) : Number(rawValue)
                        onChange(param.key, Number.isNaN(nextValue) ? 0 : nextValue)
                      }}
                  />
              )}
            </label>
        ))}
      </>
  )
}

function parseDatesFromDatasetName(name: string): { from: string; to: string } | null {
  const m = name.match(/(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})/)
  return m ? { from: m[1], to: m[2] } : null
}

function EquityChart({
  data,
  buyAndHoldData,
  dateFrom,
  dateTo,
}: {
  data: number[]
  buyAndHoldData?: number[]
  dateFrom?: string
  dateTo?: string
}) {
  if (data.length < 2) {
    return (
      <div className="chart-placeholder" aria-label="Backtest equity chart">
        <div className="chart-line" />
      </div>
    )
  }

  const W = 800
  const H = 230
  const PAD = { top: 16, right: 20, bottom: 40, left: 66 }
  const innerW = W - PAD.left - PAD.right
  const innerH = H - PAD.top - PAD.bottom

  const hasBuyAndHold = buyAndHoldData != null && buyAndHoldData.length >= 2
  const allValues = hasBuyAndHold ? [...data, ...buyAndHoldData!] : data
  const min = Math.min(...allValues)
  const max = Math.max(...allValues)
  const range = max - min || 1

  const toXAt = (i: number, len: number) => PAD.left + (i / (len - 1)) * innerW
  const toX = (i: number) => toXAt(i, data.length)
  const toY = (v: number) => PAD.top + (1 - (v - min) / range) * innerH

  function fmtVal(v: number): string {
    const abs = Math.abs(v)
    if (abs >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
    if (abs >= 1_000) return `${(v / 1_000).toFixed(1)}K`
    return v.toFixed(1)
  }

  const pathD = data
    .map((v, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`)
    .join(' ')

  const areaD =
    pathD +
    ` L${toX(data.length - 1).toFixed(1)},${(PAD.top + innerH).toFixed(1)}` +
    ` L${PAD.left.toFixed(1)},${(PAD.top + innerH).toFixed(1)} Z`

  const isProfit = data[data.length - 1] >= data[0]
  const color = isProfit ? '#22c55e' : '#ef4444'
  const fillColor = isProfit ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)'

  let buyAndHoldPathD: string | null = null
  if (hasBuyAndHold) {
    buyAndHoldPathD = buyAndHoldData!
      .map((v, i) => `${i === 0 ? 'M' : 'L'}${toXAt(i, buyAndHoldData!.length).toFixed(1)},${toY(v).toFixed(1)}`)
      .join(' ')
  }

  // Y-axis ticks
  const Y_TICKS = 5
  const yTicks = Array.from({ length: Y_TICKS }, (_, i) => {
    const v = min + (range * i) / (Y_TICKS - 1)
    return { y: toY(v), label: fmtVal(v) }
  })

  // X-axis ticks — 5 evenly spaced, labelled by date or percentage
  const startMs = dateFrom ? new Date(dateFrom).getTime() : null
  const endMs = dateTo ? new Date(dateTo).getTime() : null
  const X_TICKS = 5
  const xTicks = Array.from({ length: X_TICKS }, (_, i) => {
    const frac = i / (X_TICKS - 1)
    const idx = Math.round(frac * (data.length - 1))
    let label: string
    if (startMs !== null && endMs !== null) {
      label = new Date(startMs + frac * (endMs - startMs)).toISOString().slice(0, 10)
    } else {
      label = `${Math.round(frac * 100)}%`
    }
    return { x: toX(idx), label, i }
  })

  const baseline = PAD.top + innerH

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: '100%' }} aria-label="Equity curve chart">
      {/* Horizontal grid lines */}
      {yTicks.map((t) => (
        <line key={t.label} x1={PAD.left} y1={t.y} x2={PAD.left + innerW} y2={t.y}
          stroke="rgba(148,163,184,0.18)" strokeWidth={1} />
      ))}

      {/* Axis lines */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={baseline}
        stroke="rgba(148,163,184,0.5)" strokeWidth={1} />
      <line x1={PAD.left} y1={baseline} x2={PAD.left + innerW} y2={baseline}
        stroke="rgba(148,163,184,0.5)" strokeWidth={1} />

      {/* Buy-and-hold reference line (behind strategy curve) */}
      {buyAndHoldPathD && (
        <path
          d={buyAndHoldPathD}
          fill="none"
          stroke="rgba(148,163,184,0.55)"
          strokeWidth="1.5"
          strokeDasharray="5 3"
          strokeLinejoin="round"
        />
      )}

      {/* Area fill + strategy equity line */}
      <path d={areaD} fill={fillColor} />
      <path d={pathD} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />

      {/* Y-axis labels */}
      {yTicks.map((t) => (
        <text key={t.label} x={PAD.left - 6} y={t.y + 4} textAnchor="end"
          fontSize={11} fill="#6b7280" fontFamily="system-ui, sans-serif">
          {t.label}
        </text>
      ))}

      {/* X-axis tick marks + labels */}
      {xTicks.map(({ x, label, i }) => (
        <g key={label}>
          <line x1={x} y1={baseline} x2={x} y2={baseline + 4}
            stroke="rgba(148,163,184,0.5)" strokeWidth={1} />
          <text
            x={x} y={baseline + 16}
            textAnchor={i === 0 ? 'start' : i === X_TICKS - 1 ? 'end' : 'middle'}
            fontSize={10} fill="#9ca3af" fontFamily="system-ui, sans-serif"
          >
            {label}
          </text>
        </g>
      ))}

      {/* Legend */}
      {buyAndHoldPathD && (
        <g>
          <line x1={PAD.left + innerW - 140} y1={PAD.top + 10} x2={PAD.left + innerW - 122} y2={PAD.top + 10}
            stroke={color} strokeWidth="2" />
          <text x={PAD.left + innerW - 118} y={PAD.top + 14}
            fontSize={10} fill="#6b7280" fontFamily="system-ui, sans-serif">Strategy</text>
          <line x1={PAD.left + innerW - 60} y1={PAD.top + 10} x2={PAD.left + innerW - 42} y2={PAD.top + 10}
            stroke="rgba(148,163,184,0.7)" strokeWidth="2" strokeDasharray="4 2" />
          <text x={PAD.left + innerW - 38} y={PAD.top + 14}
            fontSize={10} fill="#6b7280" fontFamily="system-ui, sans-serif">B&amp;H</text>
        </g>
      )}
    </svg>
  )
}

const POLL_INTERVAL_MS = 2000

export function BacktestScreen() {
  const {token} = useAuth()
  const [strategies, setStrategies] = useState<StrategySummary[]>([])
  const [strategyId, setStrategyId] = useState('')
  const [parameterSchema, setParameterSchema] = useState<StrategyParameterSchema[]>([])
  const [parameterValues, setParameterValues] = useState<Record<string, unknown>>({})
  const [amount, setAmount] = useState('1000')
  const [symbol, setSymbol] = useState('')
  const [exchange, setExchange] = useState('')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [timeframe, setTimeframe] = useState('1m')
  const [stats, setStats] = useState<BacktestMetrics | null>(null)
  const [equityCurve, setEquityCurve] = useState<number[]>([])
  const [buyAndHoldCurve, setBuyAndHoldCurve] = useState<number[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [taskStatus, setTaskStatus] = useState<string>('')
  const [errorMessage, setErrorMessage] = useState('')
  const [parameterErrorMessage, setParameterErrorMessage] = useState('')
  const [isLoadingParameters, setIsLoadingParameters] = useState(false)
  const [backtests, setBacktests] = useState<BacktestRecord[]>([])
  const [isLoadingBacktests, setIsLoadingBacktests] = useState(false)
  const [selectedBacktestId, setSelectedBacktestId] = useState<number | null>(null)
  const [chartDateFrom, setChartDateFrom] = useState<string | undefined>()
  const [chartDateTo, setChartDateTo] = useState<string | undefined>()

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = () => {
    if (pollingRef.current !== null) {
      clearInterval(pollingRef.current)
      pollingRef.current = null
    }
  }

  const startPolling = (authToken: string, backtestId: number) => {
    stopPolling()
    pollingRef.current = setInterval(() => {
      void fetchBacktestStatus(authToken, backtestId)
          .then(async (response: { status: SetStateAction<string>; result: any }) => {
          setTaskStatus(response.status)
          if (response.status === 'done') {
            stopPolling()
            setStats(response.result)
            setIsRunning(false)
            setSelectedBacktestId(backtestId)
            loadBacktests(authToken)
            try {
              const curves = await fetchEquityCurve(authToken, backtestId)
              setEquityCurve(curves.equityCurve)
              setBuyAndHoldCurve(curves.buyAndHoldCurve)
            } catch {
              // equity curve is optional
            }
          } else if (response.status === 'failed') {
            stopPolling()
            setErrorMessage('Backtest failed on the worker. Check data availability.')
            setIsRunning(false)
          }
        })
        .catch(() => {
          // transient network error — keep polling
        })
    }, POLL_INTERVAL_MS)
  }

  useEffect(() => () => stopPolling(), [])

  useEffect(() => {
    if (!token) {
      return
    }

    let isMounted = true
    void fetchStrategies(token)
      .then((data) => {
        if (!isMounted) {
          return
        }
        setStrategies(data)
        setStrategyId((currentStrategyId) => {
          if (currentStrategyId && data.some((strategy) => String(strategy.id) === currentStrategyId)) {
            return currentStrategyId
          }

          return String(data[0]?.id ?? '')
        })
      })
      .catch((error) => {
        if (!isMounted) {
          return
        }
        const message = error instanceof Error ? error.message : 'Failed to load strategies'
        setErrorMessage(message)
      })

    return () => {
      isMounted = false
    }
  }, [token])

  async function handleSelectBacktest(bt: BacktestRecord) {
    stopPolling()
    setIsRunning(false)
    setTaskStatus('')
    setErrorMessage('')
    setSelectedBacktestId(bt.id)

    let metrics: BacktestMetrics | null = null
    if (bt.results) {
      try {
        const raw = JSON.parse(bt.results) as Record<string, number>
        metrics = {
          overallProfitPercent: raw.overall_profit_percent ?? 0,
          meanMonthlyProfitPercent: raw.mean_monthly_profit_percent ?? 0,
          maxDrawdownPercent: raw.max_drawdown_percent ?? 0,
          winRatePercent: raw.win_rate_percent ?? 0,
          sharpeRatio: raw.sharpe_ratio ?? 0,
        }
      } catch { /* ignore */ }
    }
    setStats(metrics)
    setEquityCurve([])
    setBuyAndHoldCurve([])

    const dates = parseDatesFromDatasetName(bt.datasetName)
    setChartDateFrom(dates?.from)
    setChartDateTo(dates?.to)

    if (!token) return
    try {
      const curves = await fetchEquityCurve(token, bt.id)
      setEquityCurve(curves.equityCurve)
      setBuyAndHoldCurve(curves.buyAndHoldCurve)
    } catch { /* equity curve is optional */ }
  }

  const loadBacktests = (authToken: string) => {
    setIsLoadingBacktests(true)
    void fetchBacktests(authToken)
      .then((data) => setBacktests(data))
      .catch(() => {})
      .finally(() => setIsLoadingBacktests(false))
  }

  useEffect(() => {
    if (!token) {
      return
    }
    loadBacktests(token)
  }, [token])

  useEffect(() => {
    if (!token || !strategyId) {
      setParameterSchema([])
      setParameterValues({})
      setParameterErrorMessage('')
      setIsLoadingParameters(false)
      return
    }

    let isMounted = true
    setIsLoadingParameters(true)
    setParameterErrorMessage('')

    void fetchStrategyParameters(token, Number(strategyId))
      .then((schema) => {
        if (!isMounted) {
          return
        }

        const filtered = schema.filter((p) => p.key !== 'symbol' && p.key !== 'exchange')
        setParameterSchema(filtered)
        setParameterValues(buildParameterValues(filtered))
      })
      .catch((error) => {
        if (!isMounted) {
          return
        }

        const message = error instanceof Error ? error.message : 'Failed to load strategy parameters'
        setParameterSchema([])
        setParameterValues({})
        setParameterErrorMessage(message)
      })
      .finally(() => {
        if (isMounted) {
          setIsLoadingParameters(false)
        }
      })

    return () => {
      isMounted = false
    }
  }, [token, strategyId])

  const statusLabel: Record<string, string> = {
    pending: 'Queued...',
    running: 'Running...',
    done: 'Done',
    failed: 'Failed',
  }

  return (
    <section aria-labelledby="backtest-heading" className="page">
      <h1 id="backtest-heading" className="page-title">
        Backtest
      </h1>

      <div className="backtest-layout">
        <aside className="card backtest-sidebar">
          <h2 className="section-title">Configuration</h2>

          <form
            className="form"
            onSubmit={(event) => {
              event.preventDefault()
              if (!token) {
                return
              }

              const selectedStrategy = strategies.find((current) => String(current.id) === strategyId)
              if (!selectedStrategy) {
                setErrorMessage('Choose a strategy first')
                return
              }
              if (!dateFrom || !dateTo) {
                setErrorMessage('Choose a date range first')
                return
              }

              const parsedAmount = Number(amount) || 0
              setIsRunning(true)
              setErrorMessage('')
              setStats(null)
              setEquityCurve([])
              setBuyAndHoldCurve([])
              setSelectedBacktestId(null)
              setChartDateFrom(dateFrom)
              setChartDateTo(dateTo)
              setTaskStatus('pending')
              void runBacktest(token, {
                strategyId: selectedStrategy.id,
                amount: parsedAmount,
                symbol,
                exchange,
                timeframe,
                dateFrom,
                dateTo,
                parameters: parameterValues,
              })
                .then((job) => {
                  startPolling(token, job.backtestId)
                })
                .catch((error) => {
                  const message = error instanceof Error ? error.message : 'Backtest failed'
                  setErrorMessage(message)
                  setIsRunning(false)
                  setTaskStatus('')
                })
            }}
          >
            <label className="field">
              <span className="field-label">Strategy</span>
              <select value={strategyId} onChange={(event) => setStrategyId(event.target.value)}>
                {strategies.map((currentStrategy) => (
                  <option key={currentStrategy.id} value={String(currentStrategy.id)}>
                    {currentStrategy.name}
                  </option>
                ))}
              </select>
            </label>

            <div className="field">
              <span className="field-label">Parameters</span>
              {isLoadingParameters ? (
                <p className="field-help">Loading strategy parameters...</p>
              ) : parameterSchema.length > 0 ? (
                <StrategyParamsForm
                  schema={parameterSchema}
                  values={parameterValues}
                  onChange={(key, value) => {
                    setParameterValues((currentValues) => ({
                      ...currentValues,
                      [key]: value,
                    }))
                  }}
                />
              ) : (
                <p className="field-help">
                  No configurable parameters are exposed for this strategy.
                </p>
              )}
              {parameterErrorMessage ? <p className="field-help field-help--error">{parameterErrorMessage}</p> : null}
            </div>

            <label className="field">
              <span className="field-label">Amount of money</span>
              <input
                type="number"
                min={0}
                step="0.0001"
                value={amount}
                onChange={(event) => setAmount(event.target.value)}
              />
            </label>

            <label className="field">
              <span className="field-label">Symbol</span>
              <input
                type="text"
                placeholder="e.g. BTCUSDT"
                value={symbol}
                onChange={(event) => setSymbol(event.target.value.toUpperCase())}
              />
            </label>

            <label className="field">
              <span className="field-label">Exchange</span>
              <input
                type="text"
                placeholder="e.g. Binance"
                value={exchange}
                onChange={(event) => setExchange(event.target.value)}
              />
            </label>

            <div className="field field--inline">
              <label className="field field--flex">
                <span className="field-label">Date from</span>
                <input
                  type="date"
                  value={dateFrom}
                  onChange={(event) => setDateFrom(event.target.value)}
                />
              </label>
              <label className="field field--flex">
                <span className="field-label">Date to</span>
                <input
                  type="date"
                  value={dateTo}
                  onChange={(event) => setDateTo(event.target.value)}
                />
              </label>
            </div>

            <label className="field">
              <span className="field-label">Timeframe</span>
              <select value={timeframe} onChange={(event) => setTimeframe(event.target.value)}>
                <option value="1m">1m</option>
                <option value="5m">5m</option>
                <option value="15m">15m</option>
                <option value="1h">1h</option>
                <option value="4h">4h</option>
                <option value="1d">1d</option>
              </select>
            </label>

            <button type="submit" className="primary-button" disabled={isRunning}>
              {isRunning ? (statusLabel[taskStatus] ?? 'Running...') : 'Run backtest'}
            </button>
            {errorMessage ? <p className="field-help field-help--error">{errorMessage}</p> : null}
          </form>
        </aside>

        <div className="backtest-main">
          <div className="card backtest-chart-card">
            <h2 className="section-title">Equity curve</h2>
            <EquityChart data={equityCurve} buyAndHoldData={buyAndHoldCurve} dateFrom={chartDateFrom} dateTo={chartDateTo} />
          </div>

          <div className="backtest-bottom">
            <div className="card backtest-summary-card">
              <h2 className="section-title">Summary</h2>
              <dl className="stats-grid">
                <div>
                  <dt>Overall profit</dt>
                  <dd>
                    {stats ? `${stats.overallProfitPercent.toFixed(1)}%` : '—'}
                  </dd>
                </div>
                <div>
                  <dt>Mean profit by month</dt>
                  <dd>
                    {stats ? `${stats.meanMonthlyProfitPercent.toFixed(2)}%` : '—'}
                  </dd>
                </div>
                <div>
                  <dt>Max drawdown</dt>
                  <dd>
                    {stats ? `${stats.maxDrawdownPercent.toFixed(1)}%` : '—'}
                  </dd>
                </div>
                <div>
                  <dt>Win rate</dt>
                  <dd>{stats ? `${stats.winRatePercent.toFixed(0)}%` : '—'}</dd>
                </div>
                <div>
                  <dt>Sharpe ratio</dt>
                  <dd>{stats?.sharpeRatio != null ? stats.sharpeRatio.toFixed(2) : '—'}</dd>
                </div>
              </dl>
            </div>

            <aside className="card backtest-notes-card">
              <h2 className="section-title">Additional info</h2>
              <p className="muted">
                Backtests run asynchronously via Celery. The equity curve and metrics are loaded once the worker completes.
              </p>
            </aside>
          </div>
        </div>
      </div>

      <div className="card table-card" style={{ marginTop: '1.5rem' }}>
        <h2 className="section-title">Previous backtests</h2>
        <div className="table-wrapper" role="region" aria-label="Backtest history">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Strategy</th>
                <th scope="col">Dataset</th>
                <th scope="col">Date</th>
                <th scope="col">Status</th>
                <th scope="col">Profit</th>
                <th scope="col">Win rate</th>
                <th scope="col">Sharpe</th>
                <th scope="col">Drawdown</th>
                <th scope="col"><span className="sr-only">Actions</span></th>
              </tr>
            </thead>
            <tbody>
              {isLoadingBacktests ? (
                <tr>
                  <td colSpan={9} className="table-empty">Loading...</td>
                </tr>
              ) : backtests.length === 0 ? (
                <tr>
                  <td colSpan={9} className="table-empty">No backtests yet.</td>
                </tr>
              ) : (
                backtests.map((bt) => {
                  let parsed: Record<string, number> | null = null
                  if (bt.results) {
                    try {
                      parsed = JSON.parse(bt.results) as Record<string, number>
                    } catch {
                      // ignore invalid JSON
                    }
                  }
                  const isSelected = selectedBacktestId === bt.id
                  return (
                    <tr
                      key={bt.id}
                      onClick={() => { void handleSelectBacktest(bt) }}
                      style={{ cursor: 'pointer', background: isSelected ? 'rgba(250,204,21,0.18)' : undefined }}
                    >
                      <td>{bt.id}</td>
                      <td>{bt.strategyName}</td>
                      <td>{bt.datasetName}</td>
                      <td>{new Date(bt.testTime).toLocaleString()}</td>
                      <td>{bt.status ?? 'done'}</td>
                      <td>{parsed ? `${parsed.overall_profit_percent.toFixed(1)}%` : '—'}</td>
                      <td>{parsed ? `${parsed.win_rate_percent.toFixed(0)}%` : '—'}</td>
                      <td>{parsed?.sharpe_ratio != null ? parsed.sharpe_ratio.toFixed(2) : '—'}</td>
                      <td>{parsed ? `${parsed.max_drawdown_percent.toFixed(1)}%` : '—'}</td>
                      <td>
                        <span style={{ fontSize: '0.78rem', color: isSelected ? '#92400e' : '#9ca3af' }}>
                          {isSelected ? '▶ viewing' : 'view'}
                        </span>
                      </td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}
