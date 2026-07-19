export function getMessage(catalog, key) {
  return String(key || '')
    .split('.')
    .reduce((value, part) => (value && typeof value === 'object' ? value[part] : undefined), catalog);
}

export function formatMessage(message, values = {}) {
  return String(message ?? '').replace(/\{(\w+)\}/g, (match, name) => (
    Object.prototype.hasOwnProperty.call(values, name) ? String(values[name]) : match
  ));
}

export function translate(catalog, fallbackCatalog, key, values = {}) {
  const message = getMessage(catalog, key) ?? getMessage(fallbackCatalog, key) ?? key;
  return formatMessage(message, values);
}
