// Sélecteur multi-toggle des strengths à balayer.
// Extrait behavior-preserving de LoraTestStudio.jsx (bloc « Strengths »), enrichi
// d'une divulgation progressive symétrique : les valeurs > 2.0 (jusqu'à 4.0) sont
// cachées derrière un bouton « + » discret, les valeurs NÉGATIVES (jusqu'à −2.0,
// le pôle inverse — indispensable pour tester un slider LoRA) derrière un bouton
// « − » miroir. Si une valeur étendue/négative est sélectionnée, sa rangée reste
// ouverte (jamais de sélection invisible).
import { useState } from 'react';
import { STRENGTH_CHOICES_EXTENDED, STRENGTH_CHOICES_NEGATIVE } from './constants';
import { hasExtendedSelection, hasNegativeSelection } from './strengthDisclosure';

export default function StrengthPicker({ choices, selected, onToggle, fmt,
                                         extendedChoices = STRENGTH_CHOICES_EXTENDED,
                                         negativeChoices = STRENGTH_CHOICES_NEGATIVE }) {
  // Ouverture manuelle par « + »/« − » ; l'ouverture EFFECTIVE force aussi si une
  // valeur de la rangée est sélectionnée (rechargement d'un prompt récent,
  // persistance…).
  const [expanded, setExpanded] = useState(false);
  const [negExpanded, setNegExpanded] = useState(false);
  const forced = hasExtendedSelection(selected);
  const negForced = hasNegativeSelection(selected);
  const open = expanded || forced;
  const negOpen = negExpanded || negForced;
  const hasExtended = Array.isArray(extendedChoices) && extendedChoices.length > 0;
  const hasNegative = Array.isArray(negativeChoices) && negativeChoices.length > 0;

  const chip = (s) => (
    <button key={s} type="button" onClick={() => onToggle(s)}
      aria-pressed={selected.includes(s)}
      className={`px-2.5 py-1 rounded-lg border text-[0.75rem] tabular-nums transition-colors ${
        selected.includes(s)
          ? 'border-purple-400/60 bg-purple-500/20 text-purple-200 font-semibold'
          : 'border-border bg-surface text-content-muted'}`}>
      {fmt(s)}
    </button>
  );

  return (
    <div className="flex flex-col gap-1">
      <span className="text-content-muted text-[0.625rem] uppercase">Strengths</span>
      <div className="flex gap-2 flex-wrap items-center">
        {hasNegative && (
          <button type="button" onClick={() => setNegExpanded((v) => !v)}
            disabled={negForced}
            aria-expanded={negOpen}
            aria-controls="strength-negative"
            aria-label={negOpen ? 'Hide negative strengths' : 'Show negative strengths (down to -2.0)'}
            title={negForced
              ? 'A negative strength is selected — deselect it to collapse'
              : negOpen ? 'Hide negative strengths'
                : 'Show negative strengths (down to -2.0) — pulls the LoRA the other way (slider LoRAs)'}
            className={`px-2.5 py-1 rounded-lg border text-[0.75rem] leading-none tabular-nums transition-colors disabled:opacity-60 ${
              negOpen
                ? 'border-purple-400/40 bg-purple-500/10 text-purple-200'
                : 'border-border bg-surface text-content-muted'}`}>
            −
          </button>
        )}
        {choices.map(chip)}
        {hasExtended && (
          <button type="button" onClick={() => setExpanded((v) => !v)}
            disabled={forced}
            aria-expanded={open}
            aria-controls="strength-extended"
            aria-label={open ? 'Hide strengths above 2.0' : 'Show strengths above 2.0 (up to 4.0)'}
            title={forced
              ? 'A strength above 2.0 is selected — deselect it to collapse'
              : open ? 'Hide strengths above 2.0' : 'Show strengths above 2.0 (up to 4.0)'}
            className={`px-2.5 py-1 rounded-lg border text-[0.75rem] leading-none tabular-nums transition-colors disabled:opacity-60 ${
              open
                ? 'border-purple-400/40 bg-purple-500/10 text-purple-200'
                : 'border-border bg-surface text-content-muted'}`}>
            +
          </button>
        )}
      </div>
      {hasNegative && negOpen && (
        <div id="strength-negative" className="flex gap-2 flex-wrap items-center">
          {negativeChoices.map(chip)}
          <span className="text-content-muted text-[0.625rem]">
            negative = LoRA pulled the other way (slider LoRAs)
          </span>
        </div>
      )}
      {hasExtended && open && (
        <div id="strength-extended" className="flex gap-2 flex-wrap items-center">
          {extendedChoices.map(chip)}
        </div>
      )}
    </div>
  );
}
