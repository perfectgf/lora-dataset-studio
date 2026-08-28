import SystemStatsReadout from '../shared/SystemStatsReadout';
import { MACHINE_LOAD_PREF_KEY } from '../../utils/systemStats';

/* 📊 The board's mount of the machine-load readout (see SystemStatsReadout for
 * everything it is and refuses to be). It keeps its historical identity on
 * purpose: the localStorage key predates the header mount, so renaming it
 * would silently re-open the readout for everyone who had folded it away —
 * and the testids are what canvasResponsive.test.js and the probe hold on to.
 *
 * 📱 It used to be `hidden` below `sm`, and the reason given was the toolbar:
 * on a 400-px screen that row already wrapped twice. That reason has gone —
 * the readout is not in the toolbar below `2xl` any more, it is in the board's
 * ⋯ shelf, where it costs the board nothing until it is opened. And the phone
 * turns out to be the device that wants it MOST: it is the screen you check
 * the machine from when you are not sitting at it, which is the whole reason
 * the board is opened over Tailscale in the first place.
 */
export default function CanvasSystemStats() {
  return (
    <SystemStatsReadout prefKey={MACHINE_LOAD_PREF_KEY} defaultEnabled
      testId="canvas-system-stats" helpTopic="canvas-machine-load" />
  );
}
