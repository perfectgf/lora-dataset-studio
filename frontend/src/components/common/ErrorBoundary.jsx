import { Component } from 'react';
import { I18nContext } from '../../i18n/I18nContext';

class ErrorBoundary extends Component {
  static contextType = I18nContext;

  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      componentStack: '',
      copied: false,
    };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
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
    this.setState({ componentStack: info?.componentStack || '' });
  }

  retry = () => {
    this.setState({
      hasError: false,
      error: null,
      componentStack: '',
      copied: false,
    });
  };

  goHome = () => {
    window.location.hash = '#/datasets';
    this.retry();
  };

  diagnosticText() {
    const { error, componentStack } = this.state;
    const route = `${window.location.pathname}${window.location.hash}`;
    return [
      `Route: ${route}`,
      `Message: ${error?.message || String(error || 'Unknown render error')}`,
      error?.stack ? `Stack:\n${error.stack}` : '',
      componentStack ? `Component stack:${componentStack}` : '',
    ].filter(Boolean).join('\n\n');
  }

  copyDiagnostic = async () => {
    try {
      await navigator.clipboard.writeText(this.diagnosticText());
      this.setState({ copied: true });
    } catch {
      this.setState({ copied: false });
    }
  }

  render() {
    const t = this.context?.t || ((key) => key);
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
              {this.props.fallbackMessage || t('errorBoundary.unexpected')}
            </p>
            {this.state.error?.message && (
              <p className="max-w-2xl break-words text-sm text-content-muted">
                {this.state.error.message}
              </p>
            )}
            <div className="flex flex-wrap items-center justify-center gap-2">
              <button
                type="button"
                onClick={this.retry}
                className="py-2.5 px-5 rounded-lg bg-gradient-primary text-white text-sm font-semibold cursor-pointer hover:-translate-y-px transition-transform"
              >
                {t('errorBoundary.tryAgain')}
              </button>
              <button
                type="button"
                onClick={this.goHome}
                className="py-2.5 px-5 rounded-lg border border-border-strong bg-surface-raised text-content text-sm font-semibold cursor-pointer hover:bg-surface-overlay"
              >
                {t('errorBoundary.backToDatasets')}
              </button>
              <button
                type="button"
                onClick={() => window.location.reload()}
                className="py-2.5 px-5 rounded-lg border border-border-strong text-content-muted text-sm font-semibold cursor-pointer hover:text-content"
              >
                {t('errorBoundary.reload')}
              </button>
            </div>
            <details className="w-full max-w-2xl rounded-lg border border-border bg-surface p-3 text-left">
              <summary className="cursor-pointer text-sm font-medium text-content-muted">
                {t('errorBoundary.technicalDetails')}
              </summary>
              <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap break-words text-xs text-content-subtle">
                {this.diagnosticText()}
              </pre>
              <button
                type="button"
                onClick={this.copyDiagnostic}
                className="mt-3 rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium text-content-muted hover:text-content"
              >
                {this.state.copied ? t('common.copied') : t('errorBoundary.copyDiagnostic')}
              </button>
            </details>
          </div>
        );
      }
      // Inline boundary (e.g. a single widget): compact fallback.
      return (
        <div className="text-center py-8 text-content-muted" role="alert">
          <p>{this.props.fallbackMessage || t('errorBoundary.inline')}</p>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
