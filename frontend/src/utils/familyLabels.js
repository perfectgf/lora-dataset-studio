/* How the app NAMES a training family, in one place.

   It used to live inside lineageChrome.jsx. The LoRA Canvas has to name families
   in a JSX-free helper — the refusal to mix Krea and Z-Image in one run has to
   SAY which two families were mixed, and `node --test` cannot parse JSX — so the
   map moved here and lineageChrome re-exports it. One table, no drift. */

export const FAMILY_LABEL = {
  zimage: 'Z-Image', krea: 'Krea 2', sdxl: 'SDXL',
  video: 'Video',
  flux: 'FLUX.1', flux2klein: 'FLUX.2 Klein', anima: 'Anima',
};

export const famLabel = (f) => FAMILY_LABEL[f] || f || 'LoRA';
