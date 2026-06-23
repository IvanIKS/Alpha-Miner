export type StrategySummary = {
  id: number
  name: string
  description: string | null
  timeframe: string
}

export type StrategyParameterSchema = {
  key: string
  label: string
  type: 'int' | 'float' | 'string' | 'bool'
  default: number | string | boolean | null
  min?: number | null
  max?: number | null
  options?: string[]
}

export type TradingBot = {
  id: number
  name: string
  strategyId: number
  strategyName: string
  symbols: string[]
  startedAt: string
  moneyAmount: number | null
  moneyCurrency: string | null
}

export type CreateOrUpdateBotInput = {
  id?: number
  name: string
  strategyId: number
  symbols: string[]
  moneyAmount: number | null
  moneyCurrency: string | null
}

export type BacktestRequest = {
  strategyId: number
  amount?: number
  symbol?: string
  exchange?: string
  timeframe?: string
  dateFrom: string
  dateTo: string
  parameters?: Record<string, unknown>
}

export type BacktestResult = {
  overallProfitPercent: number
  meanMonthlyProfitPercent: number
  maxDrawdownPercent: number
  winRatePercent: number
}

export type BacktestRecord = {
  id: number
  strategyId: number
  strategyName: string
  testTime: string
  datasetName: string
  results: string
}

type LoginResponse = {
  access_token: string
  token_type: string
}

type ApiBot = {
  id: number
  name: string
  strategy_id: number
  strategy_name: string
  symbols: string[]
  started_at: string
  money_amount: number | null
  money_symbol: string | null
}

const AUTH_TOKEN_STORAGE_KEY = 'trading-app:token'
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export function getAuthToken(): string | null {
  return window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY)
}

export function setAuthToken(token: string): void {
  window.localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, token)
}

export function clearAuthToken(): void {
  window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY)
}

async function apiRequest<T>(
  path: string,
  options: RequestInit = {},
  token?: string,
): Promise<T> {
  const headers = new Headers(options.headers)
  const isFormData = typeof FormData !== 'undefined' && options.body instanceof FormData
  if (!headers.has('Content-Type') && options.body != null && !isFormData) {
    headers.set('Content-Type', 'application/json')
  }
  if (token) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  })

  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const payload = (await response.json()) as { detail?: string }
      if (typeof payload.detail === 'string' && payload.detail) {
        detail = payload.detail
      }
    } catch {
      // Ignore invalid JSON and keep generic error message.
    }
    throw new Error(detail)
  }

  if (response.status === 204) {
    return undefined as T
  }

  return (await response.json()) as T
}

export async function login(loginValue: string, password: string): Promise<string> {
  const payload = await apiRequest<LoginResponse>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ login: loginValue, password }),
  })
  return payload.access_token
}

export async function fetchStrategies(token: string): Promise<StrategySummary[]> {
  return apiRequest<StrategySummary[]>('/strategies', { method: 'GET' }, token)
}

export async function fetchStrategyParameters(
  token: string,
  strategyId: number,
): Promise<StrategyParameterSchema[]> {
  return apiRequest<StrategyParameterSchema[]>(`/strategies/${strategyId}/parameters`, { method: 'GET' }, token)
}

function mapApiBot(bot: ApiBot): TradingBot {
  return {
    id: bot.id,
    name: bot.name,
    strategyId: bot.strategy_id,
    strategyName: bot.strategy_name,
    symbols: bot.symbols,
    startedAt: bot.started_at,
    moneyAmount: bot.money_amount,
    moneyCurrency: bot.money_symbol,
  }
}

export async function fetchBots(token: string): Promise<TradingBot[]> {
  const payload = await apiRequest<ApiBot[]>('/bots', { method: 'GET' }, token)
  return payload.map(mapApiBot)
}

export async function createOrUpdateBot(token: string, input: CreateOrUpdateBotInput): Promise<TradingBot> {
  if (input.id != null) {
    const updated = await apiRequest<ApiBot>(
      `/bots/${input.id}`,
      {
        method: 'PUT',
        body: JSON.stringify({
          name: input.name,
          strategy_id: input.strategyId,
          symbols: input.symbols,
          money_amount: input.moneyAmount,
          money_symbol: input.moneyCurrency,
        }),
      },
      token,
    )
    return mapApiBot(updated)
  }

  const created = await apiRequest<ApiBot>(
    '/bots',
    {
      method: 'POST',
      body: JSON.stringify({
        name: input.name,
        strategy_id: input.strategyId,
        symbols: input.symbols,
        money_amount: input.moneyAmount,
        money_symbol: input.moneyCurrency,
      }),
    },
    token,
  )
  return mapApiBot(created)
}

export async function deleteBot(token: string, id: number): Promise<void> {
  await apiRequest<void>(`/bots/${id}`, { method: 'DELETE' }, token)
}

export async function fetchBacktests(token: string): Promise<BacktestRecord[]> {
  type ApiRecord = {
    id: number
    strategy_id: number
    strategy_name: string
    test_time: string
    dataset_name: string
    results: string
  }
  const payload = await apiRequest<ApiRecord[]>('/backtests', { method: 'GET' }, token)
  return payload.map((r) => ({
    id: r.id,
    strategyId: r.strategy_id,
    strategyName: r.strategy_name,
    testTime: r.test_time,
    datasetName: r.dataset_name,
    results: r.results,
  }))
}

export async function runBacktest(token: string, request: BacktestRequest): Promise<BacktestResult> {
  type RunBacktestApiResponse = {
    id: number
    strategy_id: number
    strategy_name: string
    test_time: string
    dataset_name: string
    overall_profit_percent: number
    mean_monthly_profit_percent: number
    max_drawdown_percent: number
    win_rate_percent: number
  }

  const formData = new FormData()
  formData.append('strategy_id', String(request.strategyId))
  formData.append('money_amount', String(request.amount ?? 0))
  formData.append('symbol', request.symbol ?? '')
  formData.append('exchange', request.exchange ?? 'binance')
  formData.append('timeframe', request.timeframe ?? '1m')
  formData.append('date_from', request.dateFrom)
  formData.append('date_to', request.dateTo)
  if (request.parameters && Object.keys(request.parameters).length > 0) {
    formData.append('parameters', JSON.stringify(request.parameters))
  }

  const payload = await apiRequest<RunBacktestApiResponse>(
    '/backtest',
    {
      method: 'POST',
      body: formData,
    },
    token,
  )

  return {
    overallProfitPercent: Number(payload.overall_profit_percent),
    meanMonthlyProfitPercent: Number(payload.mean_monthly_profit_percent),
    maxDrawdownPercent: Number(payload.max_drawdown_percent),
    winRatePercent: Number(payload.win_rate_percent),
  }
}
