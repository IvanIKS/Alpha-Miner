import { createContext, useContext, useMemo, useState } from 'react'
import { clearAuthToken, getAuthToken, login as loginRequest, setAuthToken } from '../services/api'

type AuthContextValue = {
  isAuthenticated: boolean
  token: string | null
  login: (loginValue: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(() => getAuthToken())

  const value = useMemo<AuthContextValue>(
    () => ({
      isAuthenticated: token != null,
      token,
      login: async (loginValue: string, password: string) => {
        const nextToken = await loginRequest(loginValue, password)
        setAuthToken(nextToken)
        setToken(nextToken)
      },
      logout: () => {
        clearAuthToken()
        setToken(null)
      },
    }),
    [token],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

