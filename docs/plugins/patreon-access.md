# Patreon access

New LDS versions reach Patreon members (LDS Member) first, then everyone;
exclusive plugins stay members-only. Members-only plugins, including Creative
Library, require a connected active membership while in use. Disconnecting locks
their routes and screens immediately; installed files and saved data are kept.
The installed LDS core and public plugins remain usable.

The app never sees Patreon credentials. **Connect Patreon** asks the operator's
service for a one-time authorization link on `www.patreon.com`. After you
approve LDS there, the Patreon page shows a code (`XXXX-XXXX`): type it in the
Patreon card of LDS and press **Confirm**. Approving alone links nothing, so an
approval link sent by someone else cannot attach their installation to your
membership. The code expires after 10 minutes and allows five attempts; press
Connect Patreon again for a new one. A membership can connect a limited number
of new installations per 30 days.

Only the service's library answer counts as membership, and the service checks
it again for every download. A Patreon free trial does not include LDS access:
access starts with the first payment on Patreon.

## Shipped configuration

`store/patreon.json` ships with the app:

```json
{
  "format": "lds-patreon/1",
  "service_url": "https://store.example.invalid",
  "checkout_origin": "https://www.patreon.com",
  "join_url": "https://www.patreon.com/c/Loraperfectgf",
  "catalog": {
    "root": null,
    "first_party_ids": [],
    "plugin_ids": [],
    "plugin_namespaces": ["patreon"]
  }
}
```

- `service_url` is an origin without a path. HTTPS is required; plain HTTP is
  accepted only for a literal loopback address (`127.0.0.1`, `::1`). The
  reserved `.invalid` placeholder keeps Patreon access switched off.
- `checkout_origin` must be `https://www.patreon.com`. `join_url`, shown before
  the installation is connected, must be an HTTPS page on `www.patreon.com`.
- `catalog.root` embeds the independently verified TUF bootstrap root of the
  Patreon catalog, so the file ships alone in the Windows ZIP and both Docker
  images. `root_path` (relative to the file) is accepted instead, for test
  instances. Without a root the Patreon catalog and core channel stay off.
- The Patreon catalog provides the identifiers it lists, `first_party_ids`
  (LDS identifiers) and `plugin_ids` (`publisher.plugin`), and every external
  identifier whose publisher part is in `plugin_namespaces`. A members-only
  plugin published later as `patreon.<name>` therefore appears in every
  installation without a core update. A namespace uses the publisher grammar
  of plugin identifiers (2-32 lowercase letters, digits or `_`, starting with
  a letter); `lds` and Windows device names are refused.
- A namespace never grants an LDS identifier: those stay explicit
  `first_party_ids`, and a product of the public catalog cannot be listed.
- An identifier also listed by a local [private catalog](private-catalogs.md)
  stays with that source, inside a namespace too. No other source serves an
  identifier of these namespaces: not the public catalog, and not a private
  catalog that does not list it.

`LDS_STORE_PATREON_CONFIG` points a test instance at another file with the
same format. The TUF repository is read from `<service_url>/tuf/metadata/`
and `<service_url>/tuf/targets/`.

## Plugins

Patreon products appear in the Store marked **For Patreon members**. Their
releases must be `{"kind": "paid", "product": …, "provider": "patreon"}`, and
only the Patreon catalog may use that label. Installation and updates go
through the usual review; the archive comes through a download grant from the
Patreon service, checked against the signed length and SHA-256. Without an
active membership the review offers Connect Patreon and Join on Patreon, and
**Update all** leaves the members-only plugins out of its batch, naming them,
instead of blocking the other updates. Installed members-only plugins lock until
an active membership is connected again, without a reinstall or restart. Runtime
checks are cached for at most 15 seconds; a successful disconnect invalidates them
immediately. An unavailable verification service does not extend cached access.

## Core updates

The Patreon catalog signs `core-releases.json` (`lds-core-releases/1`). Each
release names its `commit`, its public `base`, a Git `bundle` of `base..commit`
whose only ref is `refs/heads/patreon`, a ZIP `archive`, and, for Docker, a
`source` archive of the commit's complete tree
(`core/lds-core-<version>.source.zip`; optional in older releases). Versions
follow the app's `YYYY.MM.DD[.N]` scheme. Lengths and hashes come only from the
TUF target metadata; bytes come only through grants for `lds_core`.

| Installation | Behavior |
| --- | --- |
| Git checkout | Downloads the verified bundle, reads it into a private ref, then fast-forwards only. Local commits, modified tracked files or a diverged history refuse, like `git pull --ff-only`, before any download; a checkout that already took public commits newer than the release waits for the next Patreon release. LDS restarts itself. |
| Pinokio | Same fast-forward. Pinokio keeps the restart: Stop, Update (installs changed requirements), Start. Until then LDS accepts no new work. |
| ZIP | Downloads the verified archive and applies it with the existing staged swap, backup and rollback, then restarts. |
| Docker | Windows installations from the ZIP only: installed from the host with `update-docker.bat patreon`, which the Updates card shows to members. Stop running work first. See below. |

In-app updates (Git checkout, Pinokio, ZIP) apply nothing while generation,
training, captioning, a bank or dataset operation, an installation or another
restart is in progress: the same idle check as plugin **Apply** runs before the
first file changes, and new work stays refused until LDS restarts. The Docker
host updater cannot see that work: stop it before running the command, because
the image rebuild restarts LDS without waiting.

A fast-forwarded checkout records the release in `refs/lds/patreon/installed`.
While HEAD contains that commit and the public branch does not, the public
update check reports no update and the public updater never resets the
checkout onto the public branch, even when a public commit shares its tree.
Pinokio's Update tab tolerates that state instead of failing.

A ZIP or Docker installation has no Git history, so it records the release in
`patreon-channel.json` in its data folder: `data/` for ZIP, the lane's
`data-docker-gpu/` or `data-docker/` for Docker. While that file exists, the
public update check offers no public release, and the public in-app update and
the `stable`, `v2` and `main` channels of `update-docker.bat` refuse to replace
the Patreon version, pointing to the Patreon channel instead. To return to
public releases anyway, choose **Install the public release instead** under
**Settings → Maintenance** (ZIP) or run `update-docker.bat stable force`
(Docker); the file goes once the public release is installed.

### Operator requirements

- Keep the Patreon line fast-forwardable: merge public commits into it rather
  than rebasing it. A checkout that took a public commit absent from the next
  release's history cannot fast-forward to it; set `base` to the latest public
  commit the release contains.
- Item statuses in the library keep the existing values (`active`, `revoked`,
  `verification_required`); an inactive membership must not use another value.

### Docker

A container cannot replace its own image, so the Windows host updater installs
Patreon releases: `update-docker.bat patreon`. The command exists only for an
LDS installation from the Windows ZIP archive; a Git checkout is refused
unchanged and cannot install Patreon releases in Docker yet. Stop generation,
training and other running work first: the host updater cannot see it, and
the image rebuild restarts LDS.

The Updates card shows the Patreon channel without plugin administration.
Connecting Patreon from a browser behind Docker's bridge needs the plugin admin
token, as installing plugins does (see the [Docker guide](../guide/docker.md)).

1. It starts a disposable container of the current image, never the running
   one: ComfyUI, plugins and the app do not start. That container runs
   `python -m app.services.patreon_core fetch-source`, the app's own client:
   TUF from the root shipped in the image, the membership grant, then the
   length and SHA-256 of the signed target metadata.
2. It mounts exactly two folders: the lane's data folder read-only, for the
   installation's Patreon connection, and a fresh folder of this update only,
   which no other container shares. Nothing the running container can write,
   plugins and ComfyUI custom nodes included, is ever applied on the host.
3. The host copies the archive out of that folder, checks it against the
   manifest written beside it, then uses the usual transaction: backup, code
   switch, image rebuild, health check and rollback.

When no newer Patreon release exists, nothing changes. The Docker updater never
installs an older `APP_VERSION` than the installed one, on any channel, and
after a Patreon install its public channels refuse a newer public release too
(see the channel marker above), so a double-click on `update-docker.bat` cannot
replace a Patreon installation. A release published without a `source` archive
stays unavailable to Docker; the installed version keeps working.
