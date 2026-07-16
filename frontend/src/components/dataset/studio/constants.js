// Constantes partagées du Studio de test LoRA.
// 0 = base model (LoRA off) — a useful control column; low values sweep down to it.
// Base row (always visible): 0 → 2.0, fine under 1.0, coarser above it.
export const STRENGTH_CHOICES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0];
// Extended row, revealed behind « + » (progressive disclosure): above 2.0 you are
// looking for the LoRA's over-cook / breaking point, so coarser steps up to the
// server-accepted ceiling of 4.0. Kept out of the always-on row to avoid clutter.
export const STRENGTH_CHOICES_EXTENDED = [2.25, 2.5, 2.75, 3.0, 3.5, 4.0];
export const DEFAULT_STRENGTHS = [0.7, 0.85, 1.0];
// Handoff vers la page generate (lu par IndexPage au montage).
export const PENDING_ZIMAGE_APPLY_KEY = 'pendingZImageApply';
export const ZIMAGE_LORAS_LS_KEY = 'zimageLoras_v1';
// Handoff SDXL : bascule Generate en mode 'normal' (HQ) avec un checkpoint SDXL ;
// la pile LoRA SDXL est passée via sdxlLoras_v1 (relue au mount par ZImageLoraConfig).
export const PENDING_SDXL_APPLY_KEY = 'pendingSdxlApply';
export const SDXL_LORAS_LS_KEY = 'sdxlLoras_v1';
// Handoff Krea : bascule Generate en mode 'krea' (UNET fixe, pas de checkpoint base) ;
// la pile LoRA Krea est passée via kreaLoras_v1 (relue au mount par le picker Krea).
export const PENDING_KREA_APPLY_KEY = 'pendingKreaApply';
export const KREA_LORAS_LS_KEY = 'kreaLoras_v1';
// Libellés des familles d'entraînement (= pipelines), pour le sélecteur de famille.
export const FAMILY_LABELS = { zimage: 'Z-Image', sdxl: 'SDXL', krea: 'Krea 2' };
// generate n'a que 3 formats → mapping depuis les 5 ratios du studio.
export const ASPECT_TO_GENERATE = {
  '9:16': 'portrait', '3:4': 'portrait', '1:1': 'square', '4:3': 'landscape', '16:9': 'landscape',
};
