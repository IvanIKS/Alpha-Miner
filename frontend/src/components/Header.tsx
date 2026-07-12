import { NavLink } from 'react-router-dom'

type HeaderProps = {
  onLogout: () => void
}

export function Header({ onLogout }: HeaderProps) {
  return (
    <header className="header">
      <div className="header-left">
        <div className="logo-mark">S</div>
        <div className="logo-text">
          <span className="logo-primary">Synenko</span>
          <span className="logo-secondary">&amp; partners</span>
        </div>
      </div>

      <nav aria-label="Main">
        <ul className="nav-list">
          <li>
            <NavLink
              to="/bots"
              className={({ isActive }) => (isActive ? 'nav-link nav-link--active' : 'nav-link')}
              end
            >
              Bots
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/backtest"
              className={({ isActive }) => (isActive ? 'nav-link nav-link--active' : 'nav-link')}
            >
              Backtest
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/agentic-lab"
              className={({ isActive }) => (isActive ? 'nav-link nav-link--active' : 'nav-link')}
            >
              Agentic Lab
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/db-monitor"
              className={({ isActive }) => (isActive ? 'nav-link nav-link--active' : 'nav-link')}
            >
              DB Monitor
            </NavLink>
          </li>
        </ul>
      </nav>

      <div className="header-right">
        <button type="button" className="pill-button" aria-label="Switch language">
          EN
        </button>
        <button type="button" className="ghost-button" onClick={onLogout}>
          Logout
        </button>
      </div>
    </header>
  )
}

