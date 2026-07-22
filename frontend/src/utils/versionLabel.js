/**
 * What version am I actually running?
 *
 * APP_VERSION only moves when a release is cut, so on a git checkout following
 * main it names the LAST RELEASE, not the code in front of you: someone twenty
 * commits past a release is told they are on that release, and "up to date" then
 * reads as a contradiction. The update payload already carries the branch and the
 * short sha — this just stops throwing them away.
 *
 * Packaged installs (no .git) are unaffected: there the release version IS the
 * truth, and adding anything would be noise.
 */
export function versionLabel(status) {
  const version = (status?.current || '').trim();
  const base = version ? `v${version}` : '';
  if (!status?.is_git) return base;
  const sha = (status.current_sha || '').trim();
  if (!sha) return base;
  const branch = (status.branch || '').trim();
  const ref = branch ? `${branch} ${sha}` : sha;
  // The sha is what you are running; the version is the release it descends from.
  return base ? `${base} · ${ref}` : ref;
}
