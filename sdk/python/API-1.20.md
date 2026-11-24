# API 1.20 — declared custom-node preparation

`ctx.register_node_pack(action)` registers a plugin's declared ComfyUI node
installation with the existing Setup workers. It does not install anything at
plugin import or LDS startup. The plugin's package, screen and preparation UI
remain together; the core owns the generic installation machinery.

## Manifest

Keep `pack` and the human-facing `url` used by existing `node_packs` entries.
Add the expected ComfyUI `classes` and an optional `installation` object:

```json
{
  "id": "sample.nodes",
  "name": "Example node integration",
  "version": "1.0.0",
  "api": 1,
  "schema_version": 2,
  "publisher": {"id": "sample", "name": "Sample"},
  "python_package": "lds_example_nodes",
  "compatibility": {
    "lds": ">=2026.1", "api": ">=1.20,<2", "python": ">=3.10",
    "os": ["windows"], "arch": ["x86_64"]
  },
  "permissions": ["comfyui", "filesystem"],
  "owns": {"install_actions": ["sample_nodes"]},
  "node_packs": [{
    "pack": "Example nodes",
    "url": "https://example.invalid/nodes",
    "search": "Example",
    "classes": ["ExampleNode"],
    "installation": {
      "action": "sample_nodes",
      "id": "sample.nodes",
      "version": "1.0.0",
      "folder": "sample_nodes",
      "url": "https://example.invalid/nodes-1.0.0.zip",
      "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "requirements": ["example-helper==1.2.3"],
      "archive_prefix": "nodes-1.0.0"
    }
  }]
}
```

The example uses nonfunctional addresses and a placeholder hash. Replace them
with a reviewed immutable archive, its actual SHA-256 and the exact dependencies
of that revision. Pin a release/commit archive; do not select a moving branch.

- `action` must be in this plugin's `owns.install_actions`. Each action, recipe
  `id` and destination `folder` is unique within its manifest.
- `installation.url` is the archive address; the outer `url` is a project link.
  The installer accepts no URL, command or destination override from a client.
- `classes` contains the exact node names expected from ComfyUI's `/object_info`.
  These are ComfyUI class names, which can differ from Python identifiers.
- `requirements` defaults to `[]`. Use exact `name==version` pins; flags, direct
  URLs, extras, markers, ranges and duplicate package names are refused. The pin
  is the version installed when the package is missing. A version already
  installed with the same major release is accepted as it is (see below).
- `python` defaults to `">=3.10"`. Declare the recipe's actual Python range,
  for example `">=3.12"`, using a nonempty packaging version specifier of at
  most 120 characters. This constrains ComfyUI's interpreter, separately from
  the plugin's host Python compatibility. The host checks its real version
  before pip resolution, staging wheels or downloading nodes; a mismatch names
  the current and required versions without changing the environment.
- `archive_prefix` defaults to `""`. Set it to the safe relative directory that
  contains the node package inside the archive. It is not an absolute path.
- Both `comfyui` and `filesystem` permissions and a positive
  `compatibility.api` floor of **1.20** are required. Older hosts must reject a
  package that relies on this field instead of silently ignoring its recipe.

The host validates the full recipe before importing the plugin. Legacy entries
without `installation` retain their inventory-only behavior and do not need the
new floor or permissions.

Receipts created before the optional `python` field are accepted only for its
unchanged `>=3.10` default, with all other recipe hashes and files verified. A
successful preparation rewrites the receipt with the new complete identity.
Changing the Python declaration changes the recipe identity and does not
silently adopt an older receipt.

## Author-supplied wheels

If an exact dependency has no suitable published wheel, build and qualify it
before publishing the plugin. Include it in the plugin ZIP and extend the
installation declaration:

```json
{
  "wheelhouse": "assets/wheels",
  "wheels": [{
    "filename": "example_helper-1.2.3+lds.1-py3-none-any.whl",
    "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  }]
}
```

These are example names and hashes. Supply the actual wheel, its license and
build provenance. `wheelhouse` is a safe relative subdirectory of this plugin;
each descriptor has only a wheel filename and SHA-256. Paths, links, duplicate
distributions and source archives are refused. Package inspection checks that
every declared wheel is present and matches its hash before admitting the ZIP.

The host snapshots only declared files into temporary storage, verifies their
metadata and uses them as constraints during pip's dry run. An unused wheel
does not cause an extra install. Existing dependency versions remain pinned;
the resolver selects supplied wheels only for missing dependencies. Filename,
metadata and report identities must agree. Local versions such as `1.2.3+lds.1`
follow Python packaging version semantics and can satisfy a `==1.2.3` pin.

Selected wheels pass the same GPU, destination and hash checks as downloaded
wheels. Raw wheel scripts/headers, interpreter hooks and dependencies expressed
as direct URLs are unsupported. Source files are rechecked before mutation;
pip installs only the verified snapshots with hashes and no dependency lookup.
The application never builds an sdist on the user's machine.

The recipe's identity includes wheel filenames and hashes, but not the absolute
plugin location or the relative wheelhouse. Independent plugins can carry the
same qualified artifacts and share the same node recipe. The host derives the
physical source solely from each plugin's own manifest directory; no client can
choose it.

## Registration and execution

```python
def register(ctx):
    ctx.register_node_pack('sample_nodes')
```

The method accepts only the declared action. Register it once; it cannot replace
another registered action or subsequently be replaced by a model/pip recipe.
The registry's action metadata carries `node_pack`, a copy of the exact
declaration, for the host's preparation plan and UI. Plugin code uses the public
method; it does not import the core's node-install service.

When preparation is explicitly started, the normal plugin admission rules apply:
the product must be active, its pending package changes applied, and its work
must not conflict with preparation. The worker plans the declared recipe and
passes the resulting bound `plan_id` into installation. Failures produce an
installer error. Success logs **prepared** and **restart_required=true**.

Prepared means that files and dependencies were checked on disk. It does not
mean that ComfyUI has imported them. The application must restart ComfyUI when
safe and then check every expected class before presenting the feature as ready.
This registration does not restart ComfyUI or submit a GPU render.

## Initial supported target and limits

A recipe without `requirements` or `wheels` is only files: it is prepared into
the `custom_nodes` folder of the configured ComfyUI whatever its layout
(portable, Desktop, manual, Linux), with the same archive, path and link checks.
A recipe that adds Python packages is prepared automatically only on the
configured local NVIDIA Windows portable ComfyUI layout, where LDS manages that
interpreter. Its own interpreter is validated; LDS's Python and a plugin's CPU
worker environment are not substitutes. On any other layout nothing is
installed: the plan reports `state: 'manual'` with `manual_steps` (the pinned
download, its SHA-256, where to extract it, and the `pip install` command with
the pinned requirements, GPU packages excepted), and the Setup action ends with
those steps for the user instead of a bare refusal.

The installer preserves the existing dependency versions and adds only missing,
compatible wheels from its reviewed plan. An installed package with the same
major release as the recipe's pin serves that pin and is kept: `numpy 2.3.1`
serves `numpy==2.5.2`, and `numpy 1.26.4` conflicts with it. Every OpenCV build
(`opencv-python`, `opencv-contrib-python` and their headless builds) provides
the same `cv2` module, so an installed build answers for any of them and a
second build is never added. Conflicting versions, unsupported GPU dependency
changes, foreign node directories and automatic managed-node upgrades are
refused. It does not execute an arbitrary repository `install.py`.

A compatible managed node can be shared by independent plugins without making
one plugin depend on the other. Package uninstall does not remove shared nodes
or model files. A plugin-code rollback is not a rollback of ComfyUI's Python
environment. Qualify each actual archive and runtime before offering it to users.

## Dependency presence hints

`lds_sdk.setup.missing_modules(names)` checks which top-level Python modules are
absent from the host interpreter, without importing the requested modules:

```python
from lds_sdk.setup import missing_modules

missing = missing_modules(['example_helper'])
```

Pass a list or tuple of at most 64 module names. Names must be ASCII Python
identifiers; paths and dotted submodules are rejected with `ValueError` before
inspection, because resolving a submodule could import its parent. The result
is a list of missing names. Discovery errors are treated as absence.

This is a preparation hint, not proof that a module or its native dependencies
can load. It neither installs packages nor validates a plugin worker or ComfyUI
environment. Keep the owning plugin's lifecycle checks around its preparation
UI and use the normal installation actions for any requested changes.
