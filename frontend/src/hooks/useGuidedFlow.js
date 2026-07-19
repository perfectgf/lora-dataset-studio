import { useMemo } from 'react';
import { useI18n } from '../i18n/I18nContext';

/* Pure derivation of the guided path from the dataset payload + capabilities.
   No fetching here — lives at the workspace's existing poll rhythm. */
export function deriveSteps(d, caps, checkpointCount = 0) {
  const images = (d && d.images) || [];
  const live = images.filter((i) => i.status !== 'failed');
  const kept = live.filter((i) => i.status === 'keep');
  const triage = live.filter((i) => i.status === 'pending' && i.filename);   // generated, awaiting keep/reject
  const generating = live.filter((i) => i.status === 'pending' && !i.filename);
  const captioned = kept.filter((i) => (i.caption || '').trim());
  const scored = kept.filter((i) => i.face_state);
  const trainMode = !!(caps && caps.training_visible);

  const steps = [
    { id: 'reference', label: 'Reference', targetId: 'gf-reference',
      done: !!(d && d.ref_filename), subtitle: d && d.ref_filename ? 'set' : 'one clear photo' },
    { id: 'generate', label: 'Generate', targetId: 'gf-generate',
      done: live.length > 0, subtitle: `${live.length}/25`, busy: generating.length > 0 },
    { id: 'curate', label: 'Curate', targetId: 'gf-images',
      done: live.length > 0 && triage.length === 0 && kept.length > 0,
      subtitle: triage.length ? `${triage.length} to triage` : `${kept.length} kept` },
    { id: 'caption', label: 'Caption', targetId: 'gf-captions',
      done: kept.length > 0 && captioned.length === kept.length,
      subtitle: `${captioned.length}/${kept.length} captioned` },
  ];
  if (caps && caps.face_scoring) {
    steps.push({ id: 'score', label: 'Score', targetId: 'gf-curation', optional: true,
      done: kept.length > 0 && scored.length === kept.length, subtitle: 'optional' });
  }
  steps.push(trainMode
    ? { id: 'finish', label: 'Train', targetId: 'gf-training',
        done: checkpointCount > 0, subtitle: checkpointCount ? `${checkpointCount} checkpoint(s)` : '' }
    : { id: 'finish', label: 'Export', targetId: 'gf-export', done: false, subtitle: 'ZIP',
        unavailable: false });
  if (trainMode) steps.push({ id: 'checkpoints', label: 'Checkpoints & LoRAs', targetId: 'gf-checkpoints',
    done: checkpointCount > 0,
    subtitle: checkpointCount ? `${checkpointCount} available` : 'after training' });
  steps.push({ id: 'studio', label: 'Studio', targetId: 'gf-studio',
    done: !!(d && d.best_settings),
    unavailable: !(caps && caps.studio_visible),
    hint: caps && caps.studio_visible ? '' : 'Configure ComfyUI in Settings' });

  const nextStep = steps.find((s) => !s.done && !s.optional && !s.unavailable) || null;
  return { steps, nextStep };
}

export default function useGuidedFlow(d, caps, checkpointCount = 0) {
  const { t } = useI18n();
  return useMemo(() => {
    const result = deriveSteps(d, caps, checkpointCount);
    const images = (d && d.images) || [];
    const live = images.filter((image) => image.status !== 'failed');
    const kept = live.filter((image) => image.status === 'keep');
    const triage = live.filter((image) => image.status === 'pending' && image.filename);
    const captioned = kept.filter((image) => (image.caption || '').trim());
    const steps = result.steps.map((step) => {
      let subtitle = step.subtitle;
      if (step.id === 'reference') {
        subtitle = step.done
          ? t('workspace.guided.subtitle.set')
          : t('workspace.guided.subtitle.onePhoto');
      } else if (step.id === 'curate') {
        subtitle = triage.length
          ? t('workspace.guided.subtitle.toTriage', { count: triage.length })
          : t('workspace.guided.subtitle.kept', { count: kept.length });
      } else if (step.id === 'caption') {
        subtitle = t('workspace.guided.subtitle.captioned', {
          done: captioned.length,
          total: kept.length,
        });
      } else if (step.id === 'score') {
        subtitle = t('workspace.guided.optional');
      } else if (step.id === 'finish' && checkpointCount > 0) {
        subtitle = t('workspace.guided.subtitle.checkpoints', { count: checkpointCount });
      } else if (step.id === 'checkpoints') {
        subtitle = checkpointCount > 0
          ? t('workspace.guided.subtitle.available', { count: checkpointCount })
          : t('workspace.guided.subtitle.afterTraining');
      }
      return {
        ...step,
        label: t(`workspace.guided.labels.${step.id}`),
        subtitle,
        hint: step.id === 'studio' && step.unavailable
          ? t('workspace.guided.configureComfyUI')
          : step.hint,
      };
    });
    const nextStep = steps.find((step) => !step.done && !step.optional && !step.unavailable) || null;
    return { steps, nextStep };
  }, [d, caps, checkpointCount, t]);
}
