// Formate une strength : 2 décimales, garde « 1.0 » lisible.
export const fmt = (s) => Number(s).toFixed(2).replace(/0$/, '').replace(/\.$/, '.0');
