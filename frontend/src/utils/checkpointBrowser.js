export function defaultCheckpointBase(bases) {
  const choices = Array.isArray(bases) ? bases : [];
  const official = choices.find((item) => item?.value === '');
  return official ? '' : (choices[0]?.value || '');
}

export function checkpointSelectionMatchesTraining(checkpointType, checkpointBase, trainType, trainBase) {
  return checkpointType === trainType && checkpointBase === trainBase;
}

export function trainFamilyLabel(type) {
  if (type === 'sdxl') return 'SDXL';
  if (type === 'krea') return 'Krea 2';
  if (type === 'flux') return 'FLUX.1';
  if (type === 'flux2klein') return 'FLUX.2 Klein';
  return 'Z-Image';
}

export function loraFolderLabel(type) {
  if (type === 'sdxl') return 'loras/sdxl';
  if (type === 'krea') return 'loras/krea';
  if (type === 'flux') return 'loras/flux';
  if (type === 'flux2klein') return 'loras/flux2klein';
  return 'loras/z image';
}
