import type { FormEvent } from 'react'
import { useState } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

export function LoginScreen() {
  const navigate = useNavigate()
  const location = useLocation()
  const { isAuthenticated, login: performLogin } = useAuth()

  if (isAuthenticated) {
    const state = location.state as { from?: { pathname?: string } } | null
    return <Navigate to={state?.from?.pathname ?? '/bots'} replace />
  }
  const [loginValue, setLoginValue] = useState('')
  const [password, setPassword] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    if (loginValue.trim() && password.trim()) {
      setErrorMessage('')
      setIsSubmitting(true)
      try {
        await performLogin(loginValue.trim(), password)
        const state = location.state as { from?: { pathname?: string } } | null
        const redirectTo = state?.from?.pathname ?? '/bots'
        navigate(redirectTo, { replace: true })
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Login failed'
        setErrorMessage(message)
      } finally {
        setIsSubmitting(false)
      }
    }
  }

  return (
    <div className="app-root app-root--centered">
      <div className="card login-card">
        <div className="login-logo">
          <div className="logo-mark logo-mark--large">S</div>
          <div className="logo-text">
            <span className="logo-primary">Synenko</span>
            <span className="logo-secondary">&amp; partners</span>
          </div>
        </div>

        <form className="form" onSubmit={handleSubmit}>
          <label className="field">
            <span className="field-label">Login</span>
            <input
              type="text"
              autoComplete="username"
              value={loginValue}
              onChange={(event) => setLoginValue(event.target.value)}
            />
          </label>

          <label className="field">
            <span className="field-label">Password</span>
            <input
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>

          <button type="submit" className="primary-button" disabled={isSubmitting}>
            {isSubmitting ? 'Logging in...' : 'Login'}
          </button>

          {errorMessage ? <p className="field-help field-help--error">{errorMessage}</p> : null}

          <div className="login-divider" aria-hidden="true">
            <span>or</span>
          </div>

          <div className="login-socials">
            <button type="button" className="secondary-button">
              Continue with Google
            </button>
            <button type="button" className="secondary-button">
              Continue with GitHub
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

