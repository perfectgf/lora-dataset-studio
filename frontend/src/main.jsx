import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './fonts.css'
import './index.css'
import { registerBundledPlugins } from './plugins/bundled'
import { loadPlugins } from './plugins/loadPlugins'
import { configureHostRuntime } from './plugins/runtimeHost.jsx'

// Plugins first, then React: a plugin's nav entry, route or panel has to be
// in the registry BEFORE the first render, exactly like a bundled one —
// loadPlugins never rejects, so a plugin that fails to load costs a line on
// the Plugins page, never the app. Legacy entries join the same loader.
configureHostRuntime()
registerBundledPlugins()

function mount() {
  ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  )
}

loadPlugins().then(mount, mount)
