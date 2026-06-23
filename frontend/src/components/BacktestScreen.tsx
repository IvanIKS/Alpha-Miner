import { useEffect, useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import type { BacktestRecord, BacktestResult, StrategyParameterSchema, StrategySummary } from '../services/api'
import { fetchBacktests, fetchStrategies, fetchStrategyParameters, runBacktest } from '../services/api'

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

export function BacktestScreen() {
  const { token } = useAuth()
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
  const [stats, setStats] = useState<BacktestResult | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')
  const [parameterErrorMessage, setParameterErrorMessage] = useState('')
  const [isLoadingParameters, setIsLoadingParameters] = useState(false)
  const [backtests, setBacktests] = useState<BacktestRecord[]>([])
  const [isLoadingBacktests, setIsLoadingBacktests] = useState(false)

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
                .then((result) => {
                  setStats(result)
                  loadBacktests(token)
                })
                .catch((error) => {
                  const message = error instanceof Error ? error.message : 'Backtest failed'
                  setErrorMessage(message)
                })
                .finally(() => {
                  setIsRunning(false)
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
              {isRunning ? 'Running...' : 'Run backtest'}
            </button>
            {errorMessage ? <p className="field-help field-help--error">{errorMessage}</p> : null}
          </form>
        </aside>

        <div className="backtest-main">
          <div className="card backtest-chart-card">
            <h2 className="section-title">Equity curve</h2>
            <div className="chart-placeholder" aria-label="Backtest equity chart">
              <div className="chart-line" />
            </div>
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
              </dl>
            </div>

            <aside className="card backtest-notes-card">
              <h2 className="section-title">Additional info</h2>
              <p className="muted">
                This screen is now connected to the backend API. The submitted run is persisted as a
                backtest record and the summary is read from stored results.
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
                <th scope="col">Profit</th>
                <th scope="col">Win rate</th>
                <th scope="col">Sharpe</th>
                <th scope="col">Drawdown</th>
              </tr>
            </thead>
            <tbody>
              {isLoadingBacktests ? (
                <tr>
                  <td colSpan={8} className="table-empty">Loading...</td>
                </tr>
              ) : backtests.length === 0 ? (
                <tr>
                  <td colSpan={8} className="table-empty">No backtests yet.</td>
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
                  return (
                    <tr key={bt.id}>
                      <td>{bt.id}</td>
                      <td>{bt.strategyName}</td>
                      <td>{bt.datasetName}</td>
                      <td>{new Date(bt.testTime).toLocaleString()}</td>
                      <td>{parsed ? `${parsed.overall_profit_percent.toFixed(1)}%` : '—'}</td>
                      <td>{parsed ? `${parsed.win_rate_percent.toFixed(0)}%` : '—'}</td>
                      <td>{parsed ? parsed.sharpe_ratio.toFixed(2) : '—'}</td>
                      <td>{parsed ? `${parsed.max_drawdown_percent.toFixed(1)}%` : '—'}</td>
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

