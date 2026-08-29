# `backend/comfy_nodes/` — ComfyUI nodes this app ships itself

Every OTHER custom node the app depends on lives in somebody else's repository
and is fetched at the user's click (`setup_installer._NODE_PACKS`: git clone,
ZIP fallback). The folders in HERE are different: they are our source, versioned
with the app, and INSTALLED BY COPY into the user's
`<ComfyUI>/custom_nodes/<folder>`.

Why a folder rather than a dependency
-------------------------------------
ComfyUI only loads nodes from its own `custom_nodes/`, at startup. An app that
does not own that ComfyUI has exactly two ways to put code there: fetch it from
a remote, or copy it from what it already carries. For a node that is OURS, the
copy wins on every axis — it works offline, needs no git, cannot half-install,
and can never drift from the app version that expects it.

The contract for anything added here
------------------------------------
1. **Pure, dependency-free Python.** These files are imported by the USER's
   ComfyUI interpreter, which this app does not own and must never pip into.
   `torch` and `comfy.*` are fair game (ComfyUI is running, by definition);
   nothing else is.
2. **A `.lds-version` stamp decides re-copy.** The installer writes the app
   version into the deployed folder. Same stamp = leave it alone; different
   stamp = overwrite. This is the OPPOSITE of the third-party rule in
   `_node_pack_already_there`, which preserves an existing folder because the
   user may have pinned a version of someone else's pack. Nobody pins ours.
3. **Class names are namespaced.** A user's `custom_nodes/` is a flat, shared
   namespace with no ownership: two packs registering the same class silently
   shadow each other. Ours are prefixed so they cannot be the pack that breaks
   somebody's install.
4. **Absence must be survivable.** A graph may only name one of these classes
   when the feature that needs it is switched ON. Default-off features keep the
   class out of the default graph entirely, so an install that never copied the
   folder is not a broken install — it is an install that has not opted in.

Shipping is automatic: `packaging/build_release_zip.ps1` robocopies all of
`backend/` into the release, so a folder added here reaches every install with
no change to the build recipe.
