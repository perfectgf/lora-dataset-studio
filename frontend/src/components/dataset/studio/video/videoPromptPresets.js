/**
 * ⚡ Quick prompts for the Video Test Studio — the MiniMax H3 preset set.
 *
 * WHERE THIS COMES FROM
 * These are the presets of the maintainer's other app's MiniMax H3 studio,
 * carried over verbatim so the two tools answer the same way: same labels,
 * same wording, same order. They follow MiniMax's official ComfyUI template
 * and the structure the community settled on:
 *
 *   [Visual style / genre] → [Subject + scene] → [Shot / action]
 *   → [Camera move] → (Timing) → Audio: [ambient / SFX / music / voice]
 *
 * Two idioms come straight from the official examples and are their own tabs:
 * multi-shot ("SHOT 1: … SHOT 2: Cut to …") and the timeline
 * ("Timeline: [0s-1s] … [1s-2.5s] …"). H3's differentiator is NATIVE STEREO
 * AUDIO — voice, SFX and music are generated in the same pass and therefore
 * DESCRIBED IN THE PROMPT, which is why 🔊 Audio and 🎙️ Voice exist at all.
 *
 * They STACK: a chip appends on its own line, so a shot is built by picking a
 * Scenario or a Style and then layering Camera + Audio + Voice on top. That is
 * the same gesture as ✨ Enrich leaving your text in place rather than
 * replacing it.
 *
 * Every category is safe for work.
 *
 * WHAT IS NOT CARRIED OVER: the negative presets. This lane has no negative
 * field — adding seven chips that write into nothing would be a menu that
 * lies. They come the day the field does.
 */

export const VIDEO_QUICK_PROMPT_CATEGORIES = [
  {
    id: 'scenarios',
    emoji: '🎬',
    label: 'Scenarios',
    prompts: [
      {
        label: 'Café Window',
        prompt: 'Cinematic film look, warm natural light, shallow depth of field. The subject from <Picture 1> sits by a sunlit café window and slowly lifts a warm cup toward their lips as steam curls upward. The camera holds a soft medium shot, then eases into a slow push-in. Audio: low café murmur, a gentle clink of ceramic, distant espresso machine hiss.',
      },
      {
        label: 'Neon Street',
        prompt: 'Moody neon-noir look, anamorphic flare, wet-pavement reflections. The subject from <Picture 1> strolls slowly down a rain-slicked street at night and glances toward the lens with a faint smile. The camera tracks alongside in a smooth lateral move. Audio: distant city hum, soft footsteps on wet ground, a low synth pad.',
      },
      {
        label: 'Golden Rooftop',
        prompt: 'Warm golden-hour cinematography, soft rim light, gentle film grain. The subject from <Picture 1> stands at a rooftop railing as the city glows below and turns slowly to face the camera, hair drifting in the wind. The camera performs a slow crane up and back to reveal the skyline. Audio: distant traffic, soft wind, a warm ambient drone.',
      },
      {
        label: 'Product Reveal',
        prompt: 'Editorial product film, pitch-black studio void, dramatic duotone rim lighting, deep shadow falloff. The product from <Picture 1> rests on a dark reflective surface as the lights slowly pulse brighter. SHOT 1: slow, deliberate push-in on the form. SHOT 2: cut to an extreme macro glide across the surface detail. Audio: deep ambient pad, a subtle rising whoosh, a soft mechanical click.',
      },
      {
        label: 'Rainy Window',
        prompt: 'Soft grey daylight, intimate cinematic framing, shallow focus. The subject from <Picture 1> curls beside a rain-streaked window and slowly turns the page of a book. The camera holds a quiet medium shot with a faint drift. Audio: steady rain patter on glass, a quiet page rustle, distant muffled thunder.',
      },
      {
        label: 'Kitchen Morning',
        prompt: 'Bright naturalistic morning light, handheld documentary feel. The subject from <Picture 1> pours coffee and steam rises, then they glance up and smile. The camera follows the pour with a loose handheld frame. Audio: liquid pouring, a ceramic set-down, birdsong through an open window.',
      },
      {
        label: 'Concert Stage',
        prompt: 'High-energy concert look, strobing colored spotlights, haze-filled air. The subject from <Picture 1> steps into the light and raises a hand as the crowd roars. The camera pushes in fast then whips to a wide stage reveal. Audio: a driving bassline, a cheering crowd, sharp cymbal hits.',
      },
    ],
  },
  {
    id: 'multishot',
    emoji: '🎞️',
    label: 'Multi-Shot',
    prompts: [
      {
        label: 'Push-in → Macro',
        prompt: 'SHOT 1: the scene opens exactly on image 1 as the camera executes a slow, deliberate push-in. SHOT 2: cut to an extreme macro profile gliding slowly across the surface detail. The environment stays constant throughout.',
      },
      {
        label: 'Wide → Close',
        prompt: 'SHOT 1: a wide establishing frame holds on the subject in the space. SHOT 2: hard cut to a tight close-up on the face as the eyes lift to camera. Lighting and setting remain consistent.',
      },
      {
        label: 'Reveal Turn',
        prompt: 'SHOT 1: the subject stands with their back to camera, framed medium. SHOT 2: cut as they turn to face the lens, the camera settling into a slow push-in. Continuous location and wardrobe.',
      },
      {
        label: 'Two-Angle Orbit',
        prompt: 'SHOT 1: a locked-off front view of the subject holding still. SHOT 2: cut to a slow three-quarter orbit revealing the profile, same lighting and background.',
      },
      {
        label: 'Action Beat',
        prompt: 'SHOT 1: the subject begins the motion in a medium frame. SHOT 2: cut to a low-angle close-up at the peak of the action, camera tracking with the movement. Keep the scene continuous.',
      },
    ],
  },
  {
    id: 'timeline',
    emoji: '⏱️',
    label: 'Timeline',
    prompts: [
      {
        label: '5s Beats',
        prompt: 'Timeline:\n[0s-1.5s] the subject holds still as the camera eases into a slow push-in.\n[1.5s-3.5s] they begin the main action, the camera tracking gently with them.\n[3.5s-5s] the motion settles and the camera holds a steady final frame.',
      },
      {
        label: 'Title Sequence',
        prompt: 'Timeline:\n[0s-1s] the frame opens on image 1, a title fades in with a subtle glow.\n[1s-3s] the camera drifts slowly as the subject moves.\n[3s-5s] the title resolves and holds, motion easing to stillness.\nHard cuts only, no dissolves.',
      },
      {
        label: 'Transformation',
        prompt: 'Timeline:\n[0s-2s] the subject holds a stable portrait framing.\n[2s-4s] a continuous transformation flows across them without a cut.\n[4s-5s] the new look settles as the camera holds steady.',
      },
    ],
  },
  {
    id: 'camera',
    emoji: '🎥',
    label: 'Camera',
    prompts: [
      { label: 'Slow Push-In', prompt: 'The camera executes a slow, deliberate push-in, tightening from medium to close-up with shallow depth of field.' },
      { label: 'Pull-Out Reveal', prompt: 'The camera pulls back smoothly from a tight frame into a wide reveal of the surrounding space, parallax exposing depth.' },
      { label: 'Lateral Tracking', prompt: 'A slow horizontal tracking shot drifts laterally past the subject, background parallax revealing depth.' },
      { label: 'Orbit Around', prompt: 'The camera circles the subject in a slow, smooth orbit, revealing successive angles while they hold still.' },
      { label: 'Crane Up', prompt: 'The camera rises smoothly on a crane from eye level to a high angle, the frame widening as it lifts.' },
      { label: 'Extreme Macro', prompt: 'An extreme macro shot glides slowly across fine surface detail, textures rendered in razor-sharp focus.' },
      { label: 'Locked-Off', prompt: 'The camera stays completely locked-off and static as the subject moves within a stable frame.' },
      { label: 'Handheld Follow', prompt: 'A loose handheld camera follows the subject with subtle, organic movement and natural sway.' },
    ],
  },
  {
    id: 'audio',
    emoji: '🔊',
    label: 'Audio',
    prompts: [
      { label: 'Room Tone', prompt: 'Audio: quiet ambient room tone, a soft breath, faint distant hum.' },
      { label: 'City Night', prompt: 'Audio: distant city traffic, soft footsteps, a low synth pad under the scene.' },
      { label: 'Rain & Thunder', prompt: 'Audio: steady rain patter, distant rolling thunder, a gentle wind.' },
      { label: 'Café Ambience', prompt: 'Audio: low café murmur, clinking ceramic, a distant espresso machine.' },
      { label: 'Nature', prompt: 'Audio: birdsong, rustling leaves, a soft breeze through trees.' },
      { label: 'Cinematic Score', prompt: 'Audio: a swelling cinematic orchestral score, warm strings building gently.' },
      { label: 'Lo-fi Beat', prompt: 'Audio: a mellow lo-fi hip-hop beat, soft vinyl crackle, a warm bassline.' },
      { label: 'Whoosh & Impact', prompt: 'Audio: a rising whoosh into a deep impact hit, then a low resonant tail.' },
      { label: 'Crowd Cheer', prompt: 'Audio: a roaring crowd, sharp claps, an energetic driving bassline.' },
    ],
  },
  {
    id: 'voice',
    emoji: '🎙️',
    label: 'Voice',
    prompts: [
      { label: 'Warm Whisper', prompt: 'The subject speaks softly to camera in a warm, intimate whisper, saying: "Hey… I\'ve missed you."' },
      { label: 'Confident VO', prompt: 'A confident narrator voice speaks clearly over the scene in a calm, cinematic tone.' },
      { label: 'Cheerful Greeting', prompt: 'The subject smiles and says cheerfully in a bright, friendly voice: "Welcome — so glad you\'re here!"' },
      { label: 'Singing Softly', prompt: 'The subject sings softly with a gentle melodic voice, a light and airy tone.' },
      { label: 'French Accent', prompt: 'The subject speaks in English with a soft French accent, warm and unhurried.' },
      { label: 'Excited Shout', prompt: 'The subject shouts excitedly toward the camera with high energy: "Let\'s go!"' },
    ],
  },
  {
    id: 'style',
    emoji: '🎨',
    label: 'Visual Style',
    prompts: [
      { label: 'Cinematic Film', prompt: 'Cinematic film look, shallow depth of field, warm key light, subtle film grain.' },
      { label: 'Editorial Product', prompt: 'Editorial product film, pitch-black studio void, dramatic duotone rim lighting, glossy reflections.' },
      { label: 'Vaporwave VHS', prompt: 'Vaporwave look: pink and blue gradient palette, VHS tracking artifacts, RGB chromatic aberration, lo-fi nostalgic mood.' },
      { label: 'Film Noir', prompt: 'High-contrast film-noir black-and-white, hard directional light, deep shadows, venetian-blind slats.' },
      { label: 'Anime Cel', prompt: 'Hand-drawn anime cel-shaded look, bold outlines, vibrant flat colors, expressive lighting.' },
      { label: 'Documentary', prompt: 'Naturalistic documentary look, available light, loose handheld framing, muted realistic color.' },
      { label: 'Dreamy Bloom', prompt: 'Soft dreamy look, heavy lens bloom, pastel palette, glowing highlights, gentle haze.' },
    ],
  },
];

/** Every preset, flat — what the tests enumerate so a new one cannot slip in
 *  unmeasured, and what the picker counts. */
export const allQuickPrompts = () =>
  VIDEO_QUICK_PROMPT_CATEGORIES.flatMap((c) => c.prompts.map((p) => ({ ...p, category: c.id })));

/** The preset as it should be written for THIS lane's mode.
 *
 *  The scenario and timeline presets point at the start frame the way H3's own
 *  template does — "the subject from <Picture 1>", "the frame opens on image 1".
 *  In text-to-video there IS no picture, and a prompt that references one asks
 *  the sampler to honour a frame that was never supplied. So the reference is
 *  dropped rather than the preset: same shot, same camera, same audio, minus a
 *  sentence about an image that does not exist. `videoQuickPrompts.test.js`
 *  enumerates all of them and fails if any t2v text still names a picture. */
export function promptForMode(prompt, mode) {
  if (mode !== 't2v') return prompt;
  return prompt
    .replace(/\s+from <Picture \d+>/gi, '')
    .replace(/\bopens (?:exactly )?on image \d+\b/gi, 'opens on the subject');
}

/** Appending, not replacing — the same promise ✨ Enrich makes. A chip lands on
 *  its own line under whatever is already written, and picking the same chip
 *  twice is not a way to say it twice: an exact duplicate line is refused. */
export function appendQuickPrompt(current, addition) {
  const add = (addition || '').trim();
  if (!add) return current || '';
  const base = (current || '').trimEnd();
  if (!base) return add;
  const lines = base.split('\n').map((l) => l.trim());
  if (lines.includes(add)) return base;
  return `${base}\n${add}`;
}
