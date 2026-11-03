import { useState } from 'react';
import { Search } from 'lucide-react';
import RecentPrompts from './RecentPrompts';
import DescribeImageModal from './DescribeImageModal';
import DatasetCaptionControl from './DatasetCaptionControl';
import EnhancePromptButton from './EnhancePromptButton';
import CivitaiBrowserButton from './CivitaiBrowserButton';
import { MemberPreviewButton } from '../../patreon/MemberPreviews.jsx';

// Test prompt field: textarea, reset-to-default, Enhance, Describe and recent prompts, extracted
// unchanged from LoraTestStudio.jsx. value is effectivePrompt, placeholder is d.prompt, and
// isCustom means edited from default. Render RecentPrompts only for a non-empty list.
// batchPrompts/onToggleBatchPrompt/onClearBatchPrompts pass through the replay batch; state
// belongs to RunSetupPanel, which builds launch requests. injectTrigger/onInjectTrigger likewise
// pass through the Trigger word checkbox. civitaiPicks/onToggleCivitaiPick/onClearCivitaiPicks add
// checked Civitai prompts directly to the same batch, outside history. Without handlers, the
// browser retains only Use prompt.
export default function PromptField({ value, placeholder, onChange, onReset, isCustom, recentPrompts, datasetId, onDeletePrompt,
  batchPrompts = null, onToggleBatchPrompt = null, onClearBatchPrompts = null,
  civitaiPicks = null, onToggleCivitaiPick = null, onClearCivitaiPicks = null,
  injectTrigger = true, onInjectTrigger = null }) {
  const [describeOpen, setDescribeOpen] = useState(false);
  // A described prompt replaces the field; if the user already typed one, confirm
  // before clobbering it (never silently discard their text).
  const applyDescription = (text) => {
    if (value && value.trim()
      && !window.confirm('Replace the current test prompt with the described one?')) return;
    onChange(text);
  };
  const applyCaption = (text) => {
    if (value && value.trim()
      && !window.confirm('Replace the current test prompt with a random caption drawn from your locked source?')) return;
    onChange(text);
  };
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-content-muted text-[0.625rem] uppercase">Test prompt</span>
          {onInjectTrigger && (
            <label className="flex items-center gap-1 text-content-subtle text-[0.625rem] cursor-pointer"
              title="Prefix the dataset's trigger word to this prompt when generating. Uncheck to send the prompt exactly as written — useful when a render keeps typing the trigger back (speech bubbles, signs) or for pure style/scene tests.">
              <input type="checkbox" checked={injectTrigger}
                onChange={(e) => onInjectTrigger(e.target.checked)} />
              Trigger word
            </label>
          )}
        </div>
        <div className="flex max-w-full flex-wrap items-center justify-end gap-1">
          <DatasetCaptionControl onCaption={applyCaption} />
          <EnhancePromptButton prompt={value} onResult={onChange} />
          <button type="button" onClick={() => setDescribeOpen(true)}
            title="Describe an image into a test prompt (vision model)"
            className="px-2 py-0.5 rounded border border-border bg-surface text-content-subtle text-[0.625rem] hover:text-content">
            <Search aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Describe
          </button>
          <CivitaiBrowserButton prompt={value} onPrompt={onChange}
            picks={civitaiPicks} onTogglePick={onToggleCivitaiPick} />
          <MemberPreviewButton placement="studio.prompt" />
        </div>
      </div>
      {/*
       * Show Civitai batch selections HERE below the field: history, with its own count, may be
       * empty on a new dataset. An invisible batch would launch blindly.
       */}
      {Array.isArray(civitaiPicks) && civitaiPicks.length > 0 && (
        <p className="m-0 flex flex-wrap items-center gap-1.5 text-[0.625rem] text-content-subtle"
          data-testid="civitai-batch-count">
          <span className="rounded bg-purple-500/20 px-1.5 py-0.5 font-semibold text-purple-200 tabular-nums">
            🌐 {civitaiPicks.length} Civitai prompt{civitaiPicks.length === 1 ? '' : 's'} in the batch
          </span>
          <span>— one pass each on the next run</span>
          {onClearCivitaiPicks && (
            <button type="button" onClick={onClearCivitaiPicks}
              className="inline-flex min-h-10 items-center px-1 underline decoration-dotted hover:text-content lg:min-h-0 lg:px-0">
              Clear
            </button>
          )}
        </p>
      )}
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={5}
        placeholder={placeholder}
        aria-label="LoRA test prompt"
        className="w-full rounded-lg border border-border bg-surface px-2 py-1.5 text-[0.75rem] text-content resize-y focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400"
      />
      {isCustom && (
        <button type="button" onClick={onReset}
          className="self-start px-2 py-0.5 rounded bg-surface text-content-subtle text-[0.625rem] hover:text-content"
          title="Revert to the default identity prompt">
          ↺ default
        </button>
      )}
      {Array.isArray(recentPrompts) && recentPrompts.length > 0 && (
        <RecentPrompts items={recentPrompts} datasetId={datasetId} selectedPrompt={value}
          onPick={onChange} onDelete={onDeletePrompt}
          batch={batchPrompts} onToggleBatch={onToggleBatchPrompt}
          onClearBatch={onClearBatchPrompts} />
      )}
      <DescribeImageModal open={describeOpen} onClose={() => setDescribeOpen(false)}
        onResult={applyDescription} />
    </div>
  );
}
