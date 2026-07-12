import { Navigate, Outlet, Route, Routes, useLocation } from 'react-router-dom'
import { Header } from './components/Header'
import { LoginScreen } from './components/LoginScreen'
import { BotsScreen } from './components/BotsScreen'
import { BacktestScreen } from './components/BacktestScreen'
import { DatabaseMonitorScreen } from './components/DatabaseMonitorScreen'
import { useAuth } from './contexts/AuthContext'

function RequireAuth() {
  const { isAuthenticated } = useAuth()
  const location = useLocation()

  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location }} />
  }

  return <Outlet />
}

function AppShell() {
  const { logout } = useAuth()

  return (
    <div className="app-root">
      <Header
        onLogout={() => {
          logout()
        }}
      />
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  )
}

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginScreen />} />
      <Route element={<RequireAuth />}>
        <Route element={<AppShell />}>
          <Route index element={<Navigate to="/bots" replace />} />
          <Route path="/bots" element={<BotsScreen />} />
          <Route path="/backtest" element={<BacktestScreen />} />
          <Route
            path="/agentic-lab"
            element={
              <div className="placeholder-card">
                <h2>Agentic Lab</h2>
                <p>Coming soon. This area will host experimental trading agents.</p>
              </div>
            }
          />
          <Route path="/db-monitor" element={<DatabaseMonitorScreen />} />
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/login" replace />} />
    </Routes>
  )
}

