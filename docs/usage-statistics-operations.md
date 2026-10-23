# Usage statistics: maintainer setup

This is an optional product feedback channel for participating **installations**.
It does not count identified people or all LDS users. Refusals, old versions,
offline operation, reinstalls, shared servers and cloned data folders affect
coverage. The dashboard must use that wording.

## Before connecting a public build

Create a dedicated LDS project in PostHog Cloud **EU**. Label disposable trial
installations with `LDS_USAGE_ENVIRONMENT=test`. Keep the product dashboard private and
leave paid upgrades, session replay, autocapture and automatic error capture off.
Only the three manual event types below are needed. The free plan lists one year
of event retention, but enforcement is a gradual rollout: while the project's
`events_retention_enforced` is false, no older events are removed. PostHog does
not offer a shorter custom period. The user notice discloses this limitation;
it does not promise a deletion deadline. See the
[provider's policy](https://posthog.com/docs/data/events-retention).

Review the processing agreement and the provider's IP/access-log handling.
Network receivers see the connection's IP even though LDS does not include an
IP address in event properties and requests that GeoIP enrichment be disabled.
Do not describe persistent random IDs as fully anonymous data.

`backend/app/usage_statistics_config.json` contains the approved **public ingestion
token** for the LDS EU project. This connects the optional consent flow without
requiring users to configure a service. Clearing `project_token` hides the consent
banner and prevents collection from being enabled. For a local integration trial,
set `LDS_USAGE_PROJECT_TOKEN` and start the isolated development installation.
Never use a personal/admin API key in LDS. The collector host is fixed to the
EU ingestion endpoint; application requests cannot override it.

A connected distribution needs its approved public ingestion configuration in
the release artifact. Do not ask end users to create PostHog accounts or enter
keys. Any change to the project, policy version, retention policy or environment invalidates
an earlier enabled choice; an explicit refusal stays remembered. The maintainer
must revise the policy and notice if the plan or retention terms change.

`LDS_USAGE_DISABLED=1` disables collection regardless of a saved choice. No
remote feature flags or tracking scripts are loaded. The application remains
usable without the analytics service.

## Events and exact meaning

| Event | Meaning | Specific properties |
|---|---|---|
| `lds_active_day` | One UTC day with a trusted visible browser interaction or a selected user operation | None |
| `lds_feature_used` | First interaction with a coarse core feature that UTC day | `feature` |
| `lds_action_result` | Outcome of an explicitly instrumented request, bank worker, training run or Studio result | `feature`, `action`, `result`, `error_code`, `error_reason`, `duration_bucket`, `failure_stage`, `execution`, `model_family`, `caption_backend` |

Common properties: `app_version`, `os` (windows/linux/macos/other),
`environment` (`production` by default, or `test` for isolated checks),
`first_active_day` (first activity of this consent/installation ID),
`schema_version`, `$process_person_profile: false`, `$geoip_disable: true`.
The random installation UUID is `distinct_id`. Each event also has its own
stable UUID for transport retries and a UTC timestamp. No names, addresses,
hardware fingerprints, URLs, dataset/job IDs or user-entered text are sent.

`result` is `accepted`, `completed`, `partial`, `failed` or `cancelled`.
**Accepted is not completed.** Local training requests distinguish `train_start`,
`train_resume` and `train_enqueue`; older builds used the shared action `train`.
Updated builds separately report `train_run` when the observed local process exits:
completed for exit code zero, failed for an error or failed exact-state startup,
and cancelled for a confirmed Stop. The consent generation is captured before
launch, so withdrawing and re-enabling sharing cannot attribute an old run to a
new identity. No run IDs, exit messages or logs are sent. A server restart loses
its in-memory watcher: recovered runs do not report a final outcome.
Cloud runs report `train_run` from committed core host states: `done` is completed,
`stopped` is cancelled, and `error`/`error_pod_kept` are failed. Studio reports
`generate` as completed only after its result image is saved, not when ComfyUI
finishes. Failed result retrieval/saving is distinguished from generation failure.
The core observes these committed database states without changing plugin workers.
Rolled-back transactions and released savepoints without an outer commit emit
nothing. Only new operations captured under the same consent generation qualify;
the bounded in-memory tracker does not backfill or persist run identifiers.
Bank requests report acceptance
and the selected bank worker separately reports its terminal result: filter out
accepted rows when calculating terminal failure rates. Caption and import skips
are marked partial/failed where the service exposes their counts.

`error_reason` is an additional closed category, not an exception message. It
distinguishes known refusals such as training already running, a caption backend
unavailable, GPU/dataset/bank occupied, or a missing/unusable bank folder. The
allowlist in `usage_statistics.py` is checked both when recording and before
sending. Older pending events without this property remain valid. Unknown reasons
stay unknown; the dashboard must not reinterpret historical `busy` or
`invalid_input` events as a proven cause. API status codes and messages stay
compatible; the statistics use typed refusals rather than matching free text.

Policy/schema version 2 requires a fresh opt-in for previously enabled installs;
saved refusals remain disabled. `failure_stage` is a closed phase category, such
as preparation, uploading, training, generation or saving, and is `none` for
non-failure results. It describes the last observed phase, not a proven root cause.
`execution` is local/cloud/unknown. `model_family` accepts only known family keys,
never a checkpoint name. `caption_backend` is ollama/lmstudio/unknown. Every
category is validated again at send time; missing or unrecognized context stays
unknown. No exception text, request body, engine URL or model path is transmitted.

Features cover the core datasets, bank, training, studio, gallery, setup,
plugins/settings and guide surfaces. A feature interaction does not prove a
successful operation. Arbitrary and private plugin routes/names are not captured;
the generic Plugins settings surface is only reported as `plugins`. The complete
request allowlist is `backend/app/usage_statistics_hooks.py`; committed Studio
and cloud state observation is in `backend/app/usage_lifecycle.py`. Its timestamps
measure request/worker duration, not GPU-only execution. Export completion means
the archive was built, not proof that the browser saved it.

## Dashboard definition

Create a private dashboard called **LDS — Product usage**, with these insights:

Every insight must filter `environment = production` to exclude synthetic checks.

1. **Active installations, last 7 days**: unique `distinct_id` with
   `lds_active_day` in the last seven days. Repeat over 30 days; never sum daily
   unique counts to calculate monthly uniques.
2. **Daily active installations**: daily unique `distinct_id` of `lds_active_day`.
3. **Features used**: unique installations of `lds_feature_used`, broken down by
   `feature`, over 30 days. Use unique installations rather than raw event volume.
4. **Operation outcomes**: `lds_action_result` broken down by `result` and action.
   Display accepted separately from terminal outcomes.
5. **Failures and affected installations**: `lds_action_result` filtered to
   failed/partial, broken down by `error_code`, `error_reason`, `feature`, `action`, `app_version` and `os`.
   Show affected installations as well as operation counts.
6. **Return on day 7**: cohort by `first_active_day`, returning event
   `lds_active_day` exactly seven UTC dates later. Include only cohorts old enough
   to have a fully completed seventh UTC day, with first activity in the last
   180 days. First observed activity is not necessarily
   the software's installation date. Use the same rule for day 30 once available.
7. **Operation durations**: counts by feature, action, outcome and duration bucket.
   Keep accepted request durations separate from terminal worker/run durations;
   these are approximate request/run times, not GPU-only timings.

The service sends no historical local activity at opt-in. Background polls or an
idle open server do not create active days. A later opt-out and opt-in generates
a fresh identity, so it starts a new cohort.

## Reliability and revocation

The outbox and preferences live in `data/usage-statistics.json` (under the
configured `LDS_DATA_DIR`). Writes atomically replace the file. The outbox is
limited to 500 events, seven days, five attempts per event and 50 events per batch.
The daemon wakes every minute and uses bounded exponential backoff, a three-second
connect timeout and five-second read timeout. No HTTPS is performed by business
operations. A failing collector cannot block imports, captions or rendering.

Turning sharing off clears the identifier, pending events and dedupe history.
An HTTPS send already in flight is allowed to finish before the opt-out response;
after that response no old batch can start. Existing remote events are not
deleted by opting out. Captured consent generations prevent an operation begun
before opt-in, or before an opt-out/opt-in cycle, from reporting its completion
under the new consent. A corrupt preference file fails closed.

The ingestion token is a public write capability, not authenticated proof of a
real installation. Keep administration keys out of distributed code. Monitor
unexpected volume and use service-side limits/filters; don't promise bot-proof
or exact population counts. Rate/queue limits mean delivery is best effort.

## Verification before release

The targeted consent/outbound-schema tests run with
`python -m pytest backend/tests/test_usage_statistics.py -q` and never contact
PostHog. TESTING app factories do not start the sender thread.

Before a connected release, use a separate consenting disposable installation to
verify actual event receipt, the retention disclosure, dashboard privacy and
the visible consent/withdrawal flow. Do not send the maintainer's real library
or capture real images as test data. Project/dashboard provisioning is separate
from these local tests; a successful local test is not proof of cloud ingestion.

Protocol references: [PostHog capture API](https://posthog.com/docs/api/capture),
[privacy controls](https://posthog.com/docs/privacy).
