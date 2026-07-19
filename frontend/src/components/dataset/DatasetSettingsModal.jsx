/**
 * DatasetSettingsModal — edit a dataset's identity after creation.
 *
 * Name and trigger word for character/concept datasets, plus the concept
 * description for concept datasets. Style is explicitly always-on and never
 * exposes its internal run identifier as an activation trigger.
 * Changing the concept description is what drives the caption avoid-list, so editing
 * it resets that list; the parent's toast nudges a re-caption for existing captions.
 *
 * The KIND itself (character/concept/style), fixed at creation, is editable here too.
 * It is the disruptive change — it flips the caption strategy and which panels show —
 * so a changed pill reveals an honest confirmation block that spells out what changes
 * and what is preserved (nothing is deleted; see datasetKindSwitch.js). Existing
 * captions keep the OLD strategy until re-captioned; the modal never re-captions for
 * you. The switch is refused (409) by the server while work is in progress.
 *
 * Prompt suffixes (collapsible, community feature request): a free creative
 * direction — one global text + one per framing (face/bust/body/back) — appended
 * to every GENERATED variation at generation time. Never stored into the
 * per-image prompt (a regenerate would double-apply it), never ahead of the
 * identity lock. The whole map is replaced on save; empty fields clear.
 */
import { useState } from 'react';
import { HelpBadge } from '../../help/HelpMode';
import { useI18n } from '../../i18n/I18nContext';
import { kindSwitchSummary, normalizeKindLabel } from './datasetKindSwitch';

const FIELD =
  'px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm ' +
  'placeholder:text-content-subtle focus:border-indigo-500 outline-none';

const SUFFIX_FRAMINGS = ['face', 'bust', 'body', 'back'];
const KIND_OPTIONS = ['character', 'concept', 'style'];

export default function DatasetSettingsModal({ d, busy, onSave, onClose }) {
  const { t } = useI18n();
  const initialKind = normalizeKindLabel(d.kind);
  const [kind, setKind] = useState(initialKind);
  const concept = kind === 'concept';
  const style = kind === 'style';
  const [name, setName] = useState(d.name || '');
  const [trigger, setTrigger] = useState(d.trigger_word || '');
  const [desc, setDesc] = useState(d.concept_desc || '');
  const stored = d.prompt_suffixes || {};
  const [gSuffix, setGSuffix] = useState(d.prompt_suffix || '');
  const [fSuffix, setFSuffix] = useState({
    face: stored.face || '', bust: stored.bust || '',
    body: stored.body || '', back: stored.back || '',
  });
  // Discreet by default: open only when a suffix is already set.
  const [suffixOpen, setSuffixOpen] = useState(Boolean(
    (d.prompt_suffix || '').trim()
    || Object.values(stored).some((v) => (v || '').trim())));

  const kindChanged = kind !== initialKind;
  // Existing captions were written under the OLD strategy — the confirmation nudges
  // a re-caption (never automatic). "Kept + captioned" is what re-caption would touch.
  const hasCaptions = (d.images || []).some((i) => i.status === 'keep' && i.caption);
  const switchSummary = kindChanged
    ? kindSwitchSummary(initialKind, kind, { hasCaptions }) : null;

  const canSave = name.trim() && (style || trigger.trim()) && (!concept || desc.trim());
  const save = async () => {
    if (!canSave || busy) return;
    const res = await onSave({
      name: name.trim(),
      // The kind is sent every save; the server only acts on a real change.
      kind,
      trigger_word: style ? (d.trigger_word || '') : trigger.trim(),
      concept_desc: concept ? desc.trim() : undefined,
      // Always sent: '' / {} clear on the server (the map is replaced whole).
      prompt_suffix: gSuffix.trim(),
      prompt_suffixes: Object.fromEntries(
        Object.entries(fSuffix)
          .map(([k, v]) => [k, (v || '').trim()])
          .filter(([, v]) => v)),
    });
    if (res?.ok) onClose();
  };

  return (
    <div role="dialog" aria-modal="true" aria-label={t('workspace.datasetSettings.title')}
      className="fixed inset-0 z-[9990] bg-black/80 flex items-center justify-center p-3"
      onClick={onClose}>
      <div className="w-full max-w-md rounded-xl border border-border bg-surface-overlay p-4 flex flex-col gap-3 max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}>
        <h2 className="text-content font-semibold flex items-center gap-1.5">
          ⚙️ {t('workspace.datasetSettings.title')}
        </h2>

        <label className="flex flex-col gap-1">
          <span className="text-content-muted text-xs">{t('workspace.datasetSettings.name')}</span>
          <input value={name} onChange={(e) => setName(e.target.value)} className={FIELD} />
        </label>

        {/* Kind selector — same three-way pill as the New-dataset form. Changing it
            reveals the confirmation block below (what changes / what is kept). */}
        <div className="flex flex-col gap-1.5">
          <span className="text-content-muted text-xs flex items-center gap-1">
            {t('workspace.datasetSettings.kind')}
            <HelpBadge topic="dataset-kind-switch" />
          </span>
          <div className="flex gap-1.5">
            {KIND_OPTIONS.map((val) => (
              <button key={val} type="button" onClick={() => setKind(val)}
                title={t(`workspace.datasetSettings.kinds.${val}.hint`)}
                aria-pressed={kind === val}
                className={`flex-1 px-3 py-1.5 rounded-lg border text-xs font-semibold transition-colors ${
                  kind === val
                    ? 'border-primary/60 bg-primary/15 text-content'
                    : 'border-border bg-app/40 text-content-muted hover:bg-surface-raised'}`}>
                {t(`workspace.datasetSettings.kinds.${val}.label`)}
              </button>
            ))}
          </div>
        </div>

        {style ? (
          <div className="rounded-lg border border-cyan-400/30 bg-cyan-500/10 px-3 py-2 text-[0.75rem] text-cyan-100">
            <b>{t('workspace.datasetSettings.styleNoticeTitle')}</b>{' '}
            {t('workspace.datasetSettings.styleNotice')}
          </div>
        ) : (
          <label className="flex flex-col gap-1">
            <span className="text-content-muted text-xs">{t('workspace.datasetSettings.trigger')}</span>
            <input value={trigger} onChange={(e) => setTrigger(e.target.value)}
              placeholder={t('workspace.datasetSettings.triggerPlaceholder')} className={`${FIELD} font-mono`} />
            <span className="text-content-subtle text-[0.6875rem]">
              {t('workspace.datasetSettings.triggerHelp')}
            </span>
          </label>
        )}

        {concept && (
          <label className="flex flex-col gap-1">
            <span className="text-content-muted text-xs">{t('workspace.datasetSettings.conceptDescription')}</span>
            <textarea value={desc} onChange={(e) => setDesc(e.target.value)} rows={2}
              placeholder={t('workspace.datasetSettings.conceptPlaceholder')}
              className={`${FIELD} resize-y`} />
            <span className="text-content-subtle text-[0.6875rem]">
              {t('workspace.datasetSettings.conceptHelp')}
            </span>
          </label>
        )}

        <div className="flex flex-col gap-1">
          <button type="button" onClick={() => setSuffixOpen(!suffixOpen)}
            aria-expanded={suffixOpen}
            className="flex items-center gap-1.5 text-left text-content-muted hover:text-content text-xs font-medium">
            <span className={`transition-transform ${suffixOpen ? 'rotate-90' : ''}`}>▸</span>
            ✨ {t('workspace.datasetSettings.suffixes.title')}
            <span className="text-content-subtle font-normal">
              — {t('workspace.datasetSettings.suffixes.optional')}
            </span>
          </button>
          {suffixOpen && (
            <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface px-3 py-2.5">
              <label className="flex flex-col gap-1">
                <span className="text-content-muted text-xs">{t('workspace.datasetSettings.suffixes.allShots')}</span>
                <input value={gSuffix} onChange={(e) => setGSuffix(e.target.value)}
                  maxLength={300} placeholder={t('workspace.datasetSettings.suffixes.placeholder')}
                  className={FIELD} />
              </label>
              <div className="grid grid-cols-2 gap-2">
                {SUFFIX_FRAMINGS.map((key) => (
                  <label key={key} className="flex flex-col gap-1">
                    <span className="text-content-muted text-xs">
                      {t(`workspace.datasetSettings.suffixes.framings.${key}`)}
                    </span>
                    <input value={fSuffix[key]} maxLength={300}
                      onChange={(e) => setFSuffix({ ...fSuffix, [key]: e.target.value })}
                      className={FIELD} />
                  </label>
                ))}
              </div>
              <span className="text-content-subtle text-[0.6875rem]">
                {t('workspace.datasetSettings.suffixes.help')}
              </span>
            </div>
          )}
        </div>

        {/* Honest confirmation: what a kind switch changes and what it keeps. Only
            shown once the pill actually differs from the stored kind. */}
        {switchSummary && (
          <div className="rounded-lg border border-amber-400/40 bg-amber-500/5 px-3 py-2.5 flex flex-col gap-2 text-[0.75rem]">
            <div className="text-amber-200 font-semibold flex items-center gap-1.5">
              ⚠️ {t('workspace.datasetSettings.switch.title', {
                from: t(`workspace.datasetSettings.kinds.${switchSummary.from}.label`),
                to: t(`workspace.datasetSettings.kinds.${switchSummary.to}.label`),
              })}
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-content-muted font-medium">{t('workspace.datasetSettings.switch.changesTitle')}</span>
              <ul className="list-disc pl-4 flex flex-col gap-0.5 text-content-muted">
                {switchSummary.changeKeys.map((key) => (
                  <li key={key}>{t(`workspace.datasetSettings.switch.changes.${key}`)}</li>
                ))}
              </ul>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-content-muted font-medium">{t('workspace.datasetSettings.switch.keptTitle')}</span>
              <ul className="list-disc pl-4 flex flex-col gap-0.5 text-content-subtle">
                {switchSummary.preservedKeys.map((key) => (
                  <li key={key}>{t(`workspace.datasetSettings.switch.kept.${key}`)}</li>
                ))}
              </ul>
            </div>
            {switchSummary.recaption && (
              <div className="rounded border border-amber-400/30 bg-amber-500/10 px-2 py-1.5 text-amber-100">
                {t('workspace.datasetSettings.switch.recaption', {
                  kind: t(`workspace.datasetSettings.kinds.${switchSummary.from}.label`),
                })}
              </div>
            )}
          </div>
        )}

        <div className="flex justify-end gap-2 pt-1">
          <button type="button" onClick={onClose}
            className="px-3 py-1.5 rounded-lg border border-border bg-surface text-content-muted hover:text-content text-sm">
            {t('common.cancel')}
          </button>
          <button type="button" onClick={save} disabled={!canSave || busy}
            className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            {kindChanged
              ? t('workspace.datasetSettings.changeAndSave')
              : t('common.save')}
          </button>
        </div>
      </div>
    </div>
  );
}
