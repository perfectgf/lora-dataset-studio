const PEXELS_HOSTS = new Set(['pexels.com', 'www.pexels.com']);
const MAX_URL_CHARS = 2048;
const MAX_PHOTOGRAPHER_CHARS = 160;

function safePexelsUrl(value) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (!trimmed || trimmed.length > MAX_URL_CHARS || /[\u0000-\u001f]/.test(trimmed)) return null;
  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol !== 'https:' || !PEXELS_HOSTS.has(parsed.hostname.toLowerCase())
        || parsed.username || parsed.password || (parsed.port && parsed.port !== '443')) return null;
  } catch {
    return null;
  }
  return trimmed;
}

/** Fail-closed attribution for both fresh scan items and persisted metadata. */
export function pexelsAttribution(metadata) {
  if (!metadata || typeof metadata !== 'object' || metadata.platform !== 'pexels') return null;
  if (typeof metadata.photographer !== 'string') return null;
  const rawPhotographer = metadata.photographer.trim();
  if (!rawPhotographer || rawPhotographer.length > MAX_PHOTOGRAPHER_CHARS) return null;
  const photographer = rawPhotographer.replace(/\s+/g, ' ');
  const sourceUrl = safePexelsUrl(metadata.source_url);
  const photographerUrl = safePexelsUrl(metadata.photographer_url);
  if (!sourceUrl || !photographerUrl) return null;
  return { photographer, sourceUrl, photographerUrl };
}
