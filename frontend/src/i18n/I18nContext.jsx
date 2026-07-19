import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { translate } from './core';
import { buildLocaleRegistry, detectLocale } from './localeRegistry';

const STORAGE_KEY = 'lds.locale';
const localeModules = import.meta.glob('./locales/*.json', {
  eager: true,
  import: 'default',
});
const { catalogs: CATALOGS, locales: AVAILABLE_LOCALES } =
  buildLocaleRegistry(localeModules);
const DEFAULT_LOCALE = CATALOGS.en ? 'en' : AVAILABLE_LOCALES[0]?.code;

export const I18nContext = createContext(null);

function initialLocale() {
  let saved = null;
  try {
    saved = localStorage.getItem(STORAGE_KEY);
  } catch { /* private browsing or blocked storage */ }
  return detectLocale({
    saved,
    browserLanguages: navigator.languages?.length
      ? navigator.languages
      : [navigator.language],
    locales: AVAILABLE_LOCALES,
    defaultLocale: DEFAULT_LOCALE,
  });
}

export function I18nProvider({ children }) {
  const [locale, setLocaleState] = useState(initialLocale);

  const setLocale = useCallback((next) => {
    if (!CATALOGS[next]) return;
    setLocaleState(next);
    try { localStorage.setItem(STORAGE_KEY, next); } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    const active = AVAILABLE_LOCALES.find((item) => item.code === locale);
    document.documentElement.lang = locale;
    document.documentElement.dir = active?.direction || 'ltr';
    document.title = active?.documentTitle || 'LoRA Dataset Studio';
  }, [locale]);

  const t = useCallback((key, values) => (
    translate(CATALOGS[locale], CATALOGS[DEFAULT_LOCALE], key, values)
  ), [locale]);

  const value = useMemo(() => ({
    locale,
    locales: AVAILABLE_LOCALES,
    setLocale,
    t,
  }), [locale, setLocale, t]);
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (!context) throw new Error('useI18n must be used inside I18nProvider');
  return context;
}
