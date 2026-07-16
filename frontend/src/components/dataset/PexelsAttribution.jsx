import { pexelsAttribution } from '../../utils/pexelsAttribution';

export default function PexelsAttribution({ metadata, className = '' }) {
  const attribution = pexelsAttribution(metadata);
  if (!attribution) return null;
  return (
    <span className={className}
      title={`Photo by ${attribution.photographer} · Pexels`}>
      Photo by{' '}
      <a href={attribution.photographerUrl} target="_blank" rel="noopener noreferrer"
        title={`View ${attribution.photographer}'s Pexels profile`}
        className="underline decoration-white/20 underline-offset-2 hover:text-content">
        {attribution.photographer}
      </a>
      {' · '}
      <a href={attribution.sourceUrl} target="_blank" rel="noopener noreferrer"
        title="Open this photo on Pexels"
        className="underline decoration-white/20 underline-offset-2 hover:text-content">
        Pexels
      </a>
    </span>
  );
}
