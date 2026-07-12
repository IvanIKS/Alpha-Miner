import { useEffect, useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { fetchOHLCVCoverage, type OHLCVCoverageEntry } from '../services/api'

function formatDate(iso: string): string {
  const d = new Date(iso)
  return d.toISOString().slice(0, 16).replace('T', ' ')
}

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

export function DatabaseMonitorScreen() {
  const { token } = useAuth()
  const [entries, setEntries] = useState<OHLCVCoverageEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  async function load() {
    if (!token) return
    setLoading(true)
    setError(null)
    try {
      const data = await fetchOHLCVCoverage(token)
      setEntries(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load coverage data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [token])

  const exchanges = [...new Set(entries.map((e) => e.exchange))].sort()

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Database Monitor</h1>
        <div className="page-header-actions">
          <button type="button" className="secondary-button" onClick={load} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {error && (
        <div className="card" style={{ color: '#b91c1c', borderColor: 'rgba(239,68,68,0.4)' }}>
          {error}
        </div>
      )}

      {!loading && !error && entries.length === 0 && (
        <div className="card">
          <p className="muted">No OHLCV data found in the database.</p>
        </div>
      )}

      {exchanges.map((exchange) => {
        const group = entries.filter((e) => e.exchange === exchange)
        const symbols = [...new Set(group.map((e) => e.symbol))].sort()

        return (
          <div key={exchange} className="card table-card">
            <div style={{ padding: '0.75rem 1rem', borderBottom: '1px solid rgba(209,213,219,0.9)' }}>
              <h2 className="section-title" style={{ margin: 0, textTransform: 'capitalize' }}>
                {exchange}
              </h2>
            </div>
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Timeframe</th>
                    <th>From</th>
                    <th>To</th>
                    <th style={{ textAlign: 'right' }}>Candles</th>
                  </tr>
                </thead>
                <tbody>
                  {symbols.map((symbol) => {
                    const rows = group.filter((e) => e.symbol === symbol).sort((a, b) =>
                      a.timeframe.localeCompare(b.timeframe),
                    )
                    return rows.map((row, idx) => (
                      <tr key={`${row.symbol}-${row.timeframe}`}>
                        {idx === 0 && (
                          <td rowSpan={rows.length} style={{ fontWeight: 600, verticalAlign: 'top', paddingTop: '0.85rem' }}>
                            {symbol}
                          </td>
                        )}
                        <td>
                          <span className="db-badge">{row.timeframe}</span>
                        </td>
                        <td style={{ fontVariantNumeric: 'tabular-nums', color: '#6b7280' }}>
                          {formatDate(row.fromDate)}
                        </td>
                        <td style={{ fontVariantNumeric: 'tabular-nums', color: '#6b7280' }}>
                          {formatDate(row.toDate)}
                        </td>
                        <td style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                          {formatCount(row.candleCount)}
                        </td>
                      </tr>
                    ))
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )
      })}

      {loading && (
        <div className="card">
          <p className="muted">Loading coverage data…</p>
        </div>
      )}
    </div>
  )
}
