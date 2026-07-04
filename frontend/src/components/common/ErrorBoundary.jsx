import { Component } from 'react';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    // Log en PROD aussi : rester silencieux hors DEV donnait zéro diagnostic quand
    // un vrai testeur tombait sur le crash plein écran. On émet message + stack +
    // component-stack dans la console du navigateur pour pouvoir remonter la cause
    // racine exacte (ex. « tel composant lève sur tel champ undefined »).
    try {
      console.error(
        '[ErrorBoundary]',
        error?.message || error,
        '\nstack:', error?.stack,
        '\ncomponentStack:', info?.componentStack,
      );
    } catch { /* never let logging itself break the fallback UI */ }
  }

  render() {
    if (this.state.hasError) {
      // Root-level boundary: full-screen recovery UI with a reload action so a
      // render crash never leaves the user staring at a blank white page.
      if (this.props.showReload) {
        return (
          <div
            className="min-h-screen flex flex-col items-center justify-center gap-4 p-6 text-center bg-app text-content"
            role="alert"
          >
            <p className="text-lg font-semibold">
              {this.props.fallbackMessage || 'An unexpected error occurred.'}
            </p>
            <button
              onClick={() => window.location.reload()}
              className="py-2.5 px-5 rounded-lg bg-gradient-primary text-white text-sm font-semibold cursor-pointer hover:-translate-y-px transition-transform"
            >
              Reload the page
            </button>
          </div>
        );
      }
      // Inline boundary (e.g. a single widget): compact fallback.
      return (
        <div className="text-center py-8 text-content-muted" role="alert">
          <p>{this.props.fallbackMessage || 'Something went wrong. Please refresh the page.'}</p>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
