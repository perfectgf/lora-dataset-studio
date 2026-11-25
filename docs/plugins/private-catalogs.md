# Private plugin update sources

An operator can connect a signed catalog for specific private plugins without
replacing the public catalog. Its updates use the usual Plugins review and
apply flow. Plugin code and archives stay on the configured server.

Create `plugin-store/sources.json` in the LDS data directory:

```json
{
  "sources": [{
    "metadata_url": "https://plugins.example.org/metadata/",
    "target_url": "https://plugins.example.org/targets/",
    "root_path": "private-root.json",
    "plugin_ids": ["studio.depth"]
  }]
}
```

Supply the independently verified TUF bootstrap root alongside this file.
Root paths are relative to `sources.json`. Sources must use HTTPS, except for
a server bound to a literal loopback address on the same machine. The client
does not send account credentials, follow redirects or accept trust settings
from a downloaded package. A private server must restrict access to its files;
signatures authenticate downloads, they do not hide publicly served files.

Only the exact listed plugin IDs are accepted from each source. External IDs
use `publisher.plugin` and receive no first-party privileges. Sources cannot
overlap one another or replace first-party products authorized by the primary
catalog. Updates from different catalogs can share one review, transaction and
restart. Each source keeps its own trusted root, permissions and verified
downloads; the whole selection must verify before any update is prepared.
An unavailable source does not prevent other sources from being used. Cached
listings are display-only: every installation requires fresh valid signatures,
expiry checks and archive hashes. Removing a source stops its update offers;
installed plugins and their data remain available.

For a private first-party product using a historical LDS identifier, the operator
must additionally authorize that exact identifier in `official_ids`:

```json
{
  "sources": [{
    "metadata_url": "https://plugins.example.org/metadata/",
    "target_url": "https://plugins.example.org/targets/",
    "root_path": "private-root.json",
    "plugin_ids": ["manga", "studio.depth"],
    "official_ids": ["manga"]
  }]
}
```

This permission belongs in the locally installed source configuration, never
in an archive or downloaded catalog. It grants the selected root first-party
rights for that product: verify the root independently before authorizing it.
Every historical identifier in `plugin_ids` must be explicitly authorized, and
`official_ids` cannot include any unselected identifier or a primary-catalog
product. The signed archive still goes through the normal review, provenance
receipt and installation transaction. Existing projects retain their identifier
and data. These sources load during normal startup/browsing; no alternate
launcher or `LDS_STORE_CONFIG` override is needed.

When preparing this configuration before updating an older client, leave its
external `plugin_ids` unchanged and put the additional historical identifiers in
`first_party_ids` instead:

```json
{
  "plugin_ids": ["studio.depth"],
  "first_party_ids": ["manga"]
}
```

These fields belong in the same source object with its URLs and root path.
Older clients ignore `first_party_ids` and retain the external plugin's updates;
the updated core applies the same explicit first-party permissions described
above. Do not combine `first_party_ids` and `official_ids` in one source.

The Patreon catalog shipped with the app is configured separately; see
[Patreon access](patreon-access.md). An identifier listed here stays with this
source even when the Patreon catalog is also authorized for it.

Use `store/tools/repository.py` to publish immutable package versions. Keep
signing keys outside the served directory. Private sources and their packages
do not belong in the public LDS checkout. The ordinary app updater continues
to use the public LDS repository.

## Adding products while preserving older clients

API 1.23 hosts can declare `additional_catalog_targets` in their trusted Store
configuration, for example `["catalog-v2-extra.json"]`. Each named JSON target
must be signed by the same TUF repository and follows the ordinary catalog
schema. The client combines their products with `catalog.json`; duplicate
identities or unauthorized publishers remain invalid. A catalog cannot enable
this option itself.

The public catalog keeps its original products in `catalog.json`. New first-party
identifiers live in additional targets, read only by hosts that authorize them:
Dataset Forge in `catalog-v2-extra.json`, Remove Background in
`catalog-v3-extra.json`. Each new identifier needs a new target, published before
the core that declares it: a host refuses a whole target that lists an identifier
it does not know, and its whole Store when a declared target is missing.
Publishers must preserve and renew every signed target and its archives; changing
one product must not drop another target. This lets older clients continue to
receive their existing plugin updates.
