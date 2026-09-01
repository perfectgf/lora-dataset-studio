/* 🎬 The section rail of the VIDEO dataset workspace — the mirror of
 * components/dataset/workspaceSections.js, and deliberately the same SHAPE:
 * stable ids (deep-linked through ?section=), an icon and a title for the rail,
 * a mono eyebrow tag and a short description for the section header, and a
 * `panels` list of jump destinations inside the section.
 *
 * WHY THIS FILE EXISTS AT ALL. Until it did, a video dataset was a CARD at the
 * bottom of the library: an accordion of clips, one textarea each, and the
 * training block. Everything a video set could be worked on with lived on the
 * BANK — the grid, the lightbox, the search — which triages SHOTS, before any
 * encode. So the surface that holds what you are actually going to train on had
 * no way to look at it. The image lane has never worked that way (a bank feeds a
 * dataset, and the dataset gets a workspace), and CLAUDE.md's standing rule is
 * that a difference between the two surfaces needs the maintainer's sign-off
 * rather than an accident.
 *
 * Two sections of the image rail are deliberately ABSENT here, and their absence
 * is a scope decision rather than an oversight: Curation (the quality passes run
 * on the bank's shots, and re-running them on encoded clips is backend work) and
 * Import & export (a video dataset IS its output folder — a flat directory of
 * .mp4 + homonym .txt — so "export" has no destination yet that the folder does
 * not already provide).
 */
import { Clapperboard, GraduationCap, Paperclip, PenLine } from 'lucide-react';

export const VIDEO_DATASET_SECTIONS = [
  { id: 'clips', title: 'Clips', icon: Clapperboard, eyebrow: 'overview',
    helpTopic: 'video-dataset-clips',
    description: 'Every clip in the set — play one, read what it will train as, and drop the ones that should never have been cut.',
    panels: [
      { id: 'review', title: 'Review clips', targetId: 'vds-clips-review', when: 'always' },
      { id: 'bulk', title: 'Bulk actions', targetId: 'vds-clips-bulk', when: 'hasSelection' },
    ] },
  { id: 'captions', title: 'Captions', icon: PenLine, eyebrow: 'text',
    helpTopic: 'video-dataset-captions',
    // Said here because it is the one fact that makes this section different
    // from the image one: the trainer reads the FILE, never our database.
    description: 'Captions are what training reads each clip by. Every save here rewrites the .txt sitting next to the .mp4 — that file is what the trainer opens.',
    panels: [
      { id: 'list', title: 'Edit captions', targetId: 'vds-captions-list', when: 'always' },
      // On CLIPS, not on captions. The prefix operation is written for the
      // silent ones, so gating this on "something already has a caption"
      // hid it from the one set it was designed for: a freshly promoted,
      // entirely uncaptioned one.
      { id: 'tools', title: 'Caption tools', targetId: 'vds-captions-tools', when: 'hasClips' },
    ] },
  // Only for a target that TRAINS on control images (MiniMax H3 ref2va). For
  // every other profile the section would be a permanently empty rail entry, so
  // it is not in the rail at all — see visibleVideoDatasetSections.
  { id: 'references', title: 'References', icon: Paperclip, eyebrow: 'identity',
    helpTopic: 'video-dataset-references', when: 'requiresReferences',
    description: 'The identity images this target trains against. Without them the trainer runs unconditioned in silence, so the server refuses the launch instead.',
    panels: [
      { id: 'attach', title: 'Attach references', targetId: 'vds-references-attach', when: 'always' },
    ] },
  { id: 'training', title: 'Training', icon: GraduationCap, eyebrow: 'train',
    helpTopic: 'video-dataset-training',
    description: 'Turn the clips into a video LoRA — on this machine or on a rented pod. One set of dials, two destinations.',
    panels: [
      { id: 'launch', title: 'Launch & progress', targetId: 'vds-training-launch', when: 'always' },
      // A jump destination inside Training rather than a section of its own: the
      // checkpoint list is rendered by VideoTrainingBlock from state that only
      // it holds (the cloud poll), and splitting it would mean two components
      // polling the same endpoint. `hasCheckpoints` is answered by that same
      // component through onCheckpointGroups — a rail entry that scrolls to
      // nothing is worse than no entry.
      { id: 'checkpoints', title: 'Checkpoints', targetId: 'vds-training-checkpoints', when: 'hasCheckpoints' },
    ] },
];

export function isVideoDatasetSection(id) {
  return VIDEO_DATASET_SECTIONS.some((s) => s.id === id);
}
