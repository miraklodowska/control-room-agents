# Control Room for Long-Horizon Agents

Phase 0 is a local feasibility slice for bounded, fail-closed agent execution. It uses a
deterministic injected coordinator, an in-memory shared state backing, deterministic policy, a
separate verifier gate, metadata-only evidence and telemetry, and synthetic tasks.

Phase 1 adds an opt-in Firestore state adapter while preserving the memory store as the local
default. Store selection is fail-closed:

- unset `CONTROL_ROOM_STATE_STORE`, or set it to `memory`, for process-local memory;
- set `CONTROL_ROOM_STATE_STORE=firestore` and `GOOGLE_CLOUD_PROJECT=<project-id>` for Firestore;
- optionally set `CONTROL_ROOM_FIRESTORE_DATABASE`; it defaults to `(default)`.

Any other selector, a missing project, or a blank database prevents application startup. The
Firestore client is constructed only after explicit Firestore selection. Unit tests inject an
in-process client and do not use credentials, an emulator, or the network.

On Cloud Run (detected by `K_SERVICE`), the process refuses to start unless
`CONTROL_ROOM_STATE_STORE=firestore` and `GOOGLE_CLOUD_PROJECT` are present. Ordinary local runs
with no selector still use memory. `GET /health` reports the active `state_backend`.

Firestore uses independent bounded task documents, bounded evidence within each task, separate
cycle claim/result documents, evidence-ID markers, and a separate registry document. Retryable
transaction callbacks only claim or finalize durable state; coordinator, worker outcome creation,
and telemetry are outside callbacks that the Firestore library may replay. Public Firestore task
titles must use `synthetic_<token>` and required checks are restricted to the Phase 0 synthetic
check allowlist (`check`, `progress`, `quality`, `safety`).

Task creation is globally capped at 256 by a Firestore counter read and incremented in the same
transaction as the new task document; the in-memory store applies the same cap. Cap rejection is
atomic and is returned as HTTP 409. The public request model independently limits titles to 64
characters, required checks to eight items, and each check to 64 characters; excess input is a
sanitized HTTP 422 response.

Claims use 120-second expiring UTC leases and deterministic transactional takeover of abandoned
claims. This is strictly longer than the authorized deployed Cloud Run request timeout of 60
seconds: an unexpired owner cannot be taken over, takeover is allowed at expiry, and an expired owner
cannot finalize. Keep the lease strictly longer than the deployed request timeout if either value is
changed. This is a bounded ownership and idempotency mechanism, not a mathematically general
exactly-once guarantee. Any future action that calls an external system must pass the task/cycle
idempotency identity downstream and rely on that system's idempotency contract; a Firestore claim
alone cannot prevent a timed-out external effect from completing after lease takeover.

## Requirements

- Python 3.12
- `uv`

Install exactly the locked dependencies:

```bash
uv sync --frozen
```

Run the verification gates:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy control_room
```

### Real-semantics integration suite (opt-in)

`tests/test_firestore_emulator.py` replays the concurrency-critical paths
(claim/lease/takeover/fencing/idempotency/global task cap) against the real
Firestore emulator instead of the in-memory fake, which enforces genuine
snapshot-read transactions — any read placed after a write in a transaction
callback fails with `ReadAfterWriteError`. It needs the `gcloud` CLI with the
`cloud-firestore-emulator` component and a Java runtime on `PATH` (or an
already-running emulator advertised via `FIRESTORE_EMULATOR_HOST`), and skips
cleanly when they are unavailable:

```bash
CONTROL_ROOM_EMULATOR_TESTS=1 uv run pytest tests/test_firestore_emulator.py -q
```

## Local Docker build and run

Build the image from the repository root:

```bash
docker build -t control-room-agents:local .
```

Run the local in-memory configuration:

```bash
docker run --rm -p 8080:8080 \
  -e CONTROL_ROOM_STATE_STORE=memory \
  control-room-agents:local
```

Then request `http://127.0.0.1:8080/health`. The controller actually built the image and ran both
normal and breaker container smokes as external execution evidence. The packaging tests remain
static inspection and are not Docker runtime evidence.

Firestore mode requires these container environment variables:

- `CONTROL_ROOM_STATE_STORE=firestore` (required selector);
- `GOOGLE_CLOUD_PROJECT=<project-id>` (required project);
- `CONTROL_ROOM_FIRESTORE_DATABASE=<database-id>` (optional; defaults to `(default)`).

The container must also receive Application Default Credentials from its runtime identity or an
explicit read-only credential mount. Credentials must not be copied into the image. For a local
credential-file mount, set `GOOGLE_APPLICATION_CREDENTIALS` to its in-container path. Cloud Run
also sets `K_SERVICE`; when present, startup fails unless Firestore and the project are explicitly
configured.

## Local normal smoke

The examples use ports 8010/8011; substitute any free local ports (a fresh-clone
reproducibility run on 2026-08-20 found port 8000 occupied by an unrelated local
service, which makes the smoke fail against the wrong server).

Terminal 1:

```bash
uv run uvicorn control_room.api:app --host 127.0.0.1 --port 8010
```

Terminal 2:

```bash
uv run python scripts/smoke.py --base-url http://127.0.0.1:8010 --mode normal
```

Stop the server with Ctrl-C.

## Local breaker smoke

The breaker route is absent unless demo mode is explicitly enabled. Start a fresh server:

```bash
CONTROL_ROOM_DEMO_MODE=true uv run uvicorn control_room.api:app --host 127.0.0.1 --port 8011
```

Then run:

```bash
uv run python scripts/smoke.py --base-url http://127.0.0.1:8011 --mode breaker
```

Stop the server with Ctrl-C.

## Gemini coordinator (optional)

The service defaults to the provider-free deterministic coordinator. To run the
model-backed coordinator locally against Vertex AI (requires Application Default
Credentials with Vertex access):

```bash
CONTROL_ROOM_COORDINATOR=gemini \
GOOGLE_GENAI_USE_VERTEXAI=TRUE \
GOOGLE_CLOUD_PROJECT=<project-id> \
GOOGLE_CLOUD_LOCATION=global \
uv run uvicorn control_room.api:app --host 127.0.0.1 --port 8010
```

`gemini-3.5-flash` is served from Vertex's `global` endpoint (not from all
regions). Every model call is admitted by a transactional daily budget
(`CONTROL_ROOM_MODEL_CALLS_PER_DAY`, default 50); when the budget is exhausted or
unreadable the coordinator falls back to the deterministic proposal without
calling the model, and any model/transport error fails the cycle closed to
`NEEDS_OPERATOR`. `scripts/canary_v2.py` sends exactly one bounded probe request
through the same seam and asserts model version, finish reason, and a strict
content parse.

## Prior-work disclosure

Control Room is a new standalone repository created during the contest window.
It deliberately reuses design *concepts* from the author's private
`hermes-operator-controls` project — metadata-only telemetry, bounded tool/failure
budgets, and typed verification evidence — reimplemented here from scratch. No
code was copied from that repository. See `docs/devpost-draft.md` for the full
disclosure text.

## Independent SPEC review RED/GREEN ledger

The repository had no `HEAD` before this remediation. `uv.lock` was already present in the initial
untracked 22-file Phase 0 manifest, so no commit-order claim is made for the lockfile or framework
imports.

1. PASS authorization — `uv run pytest tests/test_state.py -q -k
   'worker_cycle_cannot_authorize_pass or pass_requires_typed_verification'`: RED, 2 failed because
   worker PASS and non-verification evidence were accepted; GREEN, 2 passed.
2. Terminal immutability — `uv run pytest tests/test_state.py tests/test_api.py -q -k
   'failure_outcome_cannot_mutate_terminal or breaker_rejects_terminal'`: RED, 3 failed because
   failure/breaker outcomes mutated terminal tasks; GREEN, 3 passed.
3. Service fail-closed persistence — `uv run pytest tests/test_api.py -q -k
   'malformed_and_policy_denied_cycles'`: RED, 4 failed because coordinator injection was absent and
   denials could not produce stored results; GREEN, 4 passed in the initial cycle and 8 parameter
   cases in the final suite.
4. Atomic failure accounting — `uv run pytest tests/test_state.py -q -k
   'any_execution_exception or post_execution_evidence_validation'`: RED, 2 failed because an
   arbitrary exception and invalid evidence escaped; GREEN via the same selection plus the PASS
   boundary regressions, 4 passed.
5. Canonical operation idempotency — `uv run pytest tests/test_api.py -q -k
   'same_key_for_normal_and_breaker_operations'`: RED, 2 failed because normal and breaker calls
   replayed each other; GREEN, 2 passed.
6. Registry-bound actors — `uv run pytest tests/test_state.py tests/test_api.py -q -k
   'unknown_actor_and_verifier or verifier_action_does_not_overwrite or
   malformed_and_policy_denied_cycles'`: RED, 3 failed and 5 passed because actor IDs were trusted;
   GREEN, 8 passed in the initial cycle and 10 selected cases in the final suite.
7. Evidence privacy/immutability — `uv run pytest tests/test_domain.py tests/test_state.py -q -k
   'content_like_key_variants or narrow_identifier_and_status or callers_cannot_mutate'`: RED, 7
   failed because key variants and nested reference mutation were accepted; GREEN, 7 passed.
8. Baton/registry versions — `uv run pytest tests/test_state.py -q -k
   'baton_write_requires or registry_write_requires'`: RED, 2 failed because the version contracts
   were absent; GREEN, 2 passed.
9. Inclusive intent expiry — `uv run pytest tests/test_policy.py -q -k
   'exactly_300_seconds'`: RED, 1 failed because age 300 seconds was accepted; GREEN, 1 passed.
10. Operator dashboard — `uv run pytest tests/test_api.py -q -k
    'home_exposes_minimal_operator_dashboard_contract'`: RED, 1 failed because the required controls
    and panels were absent; GREEN, 1 passed.
11. Verified-result semantics, direct terminal mutation guards, and untyped FAILED outcomes were
    exercised together with the exact offline command
    `UV_CACHE_DIR=/tmp/control-room-uv-cache uv run --offline pytest tests/test_state.py -q -k
    'pass_rejects_verification_without_exact_verified_status or
    direct_task_mutations_reject_terminal_without_any_mutation or
    failed_target_without_failure_code'`. RED output summary: `FFFFFFFFFF [100%]`, then `10 failed,
    22 deselected in 0.11s`. The ten failures were five non-`verified` verification references that
    incorrectly authorized PASS, four PASSED/FAILED direct evidence or baton mutations that did not
    raise, and one `target_state=FAILED` outcome without `failure_code` that persisted FAILED with
    zero failed actions. GREEN output summary for the exact same command: `.......... [100%]`, then
    `10 passed, 22 deselected in 0.06s`.

This second remediation was performed in the same untracked repository with no `HEAD`. The exact
terminal output above is contemporaneous execution evidence, but Git cannot independently prove or
reconstruct test-before-production edit ordering without a historical commit graph.

Supplemental adversarial classifications used `uv run pytest tests/test_api.py -q -k
'coordinator_exception_persists'` (RED: 1 failed due to an escaping exception) and `uv run pytest
tests/test_api.py -q -k 'non_mapping_model_output'` (RED: 1 failed due to misclassification); each
finished GREEN with 1 passed. Store-boundary revalidation used `uv run pytest tests/test_state.py -q
-k 'store_revalidates_copied_evidence'` (RED: 1 failed; GREEN: 1 passed), and normal terminal API
rejection used `uv run pytest tests/test_api.py -q -k 'normal_cycle_rejects_terminal'` (RED: 1
failed; GREEN: 1 passed). Final authoritative gate counts are recorded after rerunning the full
commands above, not inferred from this incremental ledger.

## Third strict remediation RED/GREEN ledger

This pass also ran in the same untracked 22-file repository with no `HEAD`. The output below is
exact contemporaneous filesystem-state evidence; without a commit graph it is not historical proof
of test-before-production edit ordering.

1. Caller/proposal cycle-key equality and two-dimensional store idempotency used
   `UV_CACHE_DIR=/tmp/control-room-uv-cache uv run --offline pytest tests/test_policy.py
   tests/test_state.py tests/test_api.py -q -k 'caller_and_proposed_cycle_keys_must_match or
   canonical_operation_never_replaces_canonical_intent or
   cycle_key_mismatch_fails_before_admission'`. RED: `FFF [100%]`, `3 failed, 65 deselected`;
   GREEN: `... [100%]`, `3 passed, 71 deselected`.
2. Pre-admission accounting, deep Intent revalidation, and constructed invalid outcomes used
   `UV_CACHE_DIR=/tmp/control-room-uv-cache uv run --offline pytest tests/test_policy.py
   tests/test_state.py tests/test_api.py -q -k 'malformed_and_policy_denied_cycles_persist_fail_closed
   or coordinator_exception_persists_idempotent or budget_policy_denial_persists or
   cycle_key_mismatch_fails_before_admission or existing_intent_payload_mutation or
   constructed_outcome_with_non_string_failure_code'`. RED: `FF.FFFFFFFFFFFF [100%]`, `14 failed,
   1 passed, 57 deselected`; GREEN: `............... [100%]`, `15 passed, 59 deselected`.
3. Scalar immutable telemetry and truthful breaker availability used
   `UV_CACHE_DIR=/tmp/control-room-uv-cache uv run --offline pytest tests/test_telemetry.py
   tests/test_api.py -q -k 'telemetry_rejects_container_content_and_wrong_scalar_types or
   telemetry_rejects_non_string_event_name or telemetry_reads_cannot_mutate or
   dashboard_reports_breaker_availability_only_from_health'`. RED: `FFFFFFFFF.FF [100%]`, `11
   failed, 1 passed, 25 deselected`; GREEN: `............ [100%]`, `12 passed, 25 deselected`.

Final third-pass gates: full pytest `99 passed, 2 warnings in 0.45s`; Ruff `All checks passed!`;
mypy `Success: no issues found in 9 source files`. The warnings are upstream ADK and
FastAPI/TestClient deprecations; no tests were skipped.

## Final single-blocker malformed `ActionFailed` RED/GREEN ledger

This pass ran in the same untracked repository with no `HEAD`. The output below is exact
contemporaneous filesystem-state evidence; without a commit graph it cannot independently prove or
reconstruct test-before-production edit ordering.

The exact targeted command was `UV_CACHE_DIR=/tmp/control-room-uv-cache uv run --offline pytest
tests/test_state.py -q -k
malformed_raised_action_failure_is_atomic_sanitized_and_idempotent`. RED: `F [100%]`, `1 failed, 34
deselected in 0.10s`; Pydantic `ValidationError` escaped while constructing `CycleOutcome` from the
dict code `{"raw_content": "must-not-persist"}`. Before deliberately failing the RED test, its
assertions confirmed that the callback ran once, the task and both action counters were unchanged,
and no idempotency record existed. GREEN, using the exact same command: `. [100%]`, `1 passed, 34
deselected in 0.03s`.

Final gates after the fix: full pytest `100 passed, 2 warnings in 0.58s`; Ruff `All checks passed!`;
mypy `Success: no issues found in 9 source files`. The warnings are the same upstream ADK and
FastAPI/TestClient deprecations; no tests were skipped.

## Quality-review RED/GREEN ledger

This pass ran in the same untracked exact 22-file Phase 0 manifest on unborn `main`; there was no
commit, push, cloud call, or deployment. The RED/GREEN results below are contemporaneous command
output. With no `HEAD`, Git cannot independently reconstruct the edit ordering.

1. Shared-backing concurrent idempotency used `UV_CACHE_DIR=/tmp/control-room-uv-cache uv run
   --offline pytest tests/test_api.py -q -k
   concurrent_same_cycle_across_shared_service_instances_executes_once`. RED: `F [100%]`, `1
   failed, 25 deselected, 2 warnings in 0.36s`; both services passed the replay lookup before either
   committed and the coordinator ran twice. GREEN with the exact same command: `. [100%]`, `1
   passed, 25 deselected, 2 warnings in 0.32s`. A supplemental two-app-instance selection later
   returned `2 passed, 31 deselected, 2 warnings in 0.37s` for the service and app concurrency
   regressions together.
2. Opaque caller idempotency used `UV_CACHE_DIR=/tmp/control-room-uv-cache uv run --offline pytest
   tests/test_api.py -q -k 'caller_idempotency_key_is_opaque_everywhere_and_replays or
   caller_idempotency_key_must_be_nonblank_and_bounded or
   caller_idempotency_key_accepts_exactly_256_characters'`. RED: `F.FFF [100%]`, `4 failed, 1
   passed, 26 deselected, 2 warnings in 0.39s`; raw keys appeared in evidence, baton, results, and
   backing records, while whitespace and 257-character keys were accepted. GREEN with the exact
   same command: `..... [100%]`, `5 passed, 26 deselected, 2 warnings in 0.35s`.
3. Atomic task/baton status snapshots used `UV_CACHE_DIR=/tmp/control-room-uv-cache uv run --offline
   pytest tests/test_state.py tests/test_api.py -q -k 'task_status_is_one_defensive_snapshot or
   task_status_endpoint_cannot_mix_task_and_baton_snapshots'`. RED: `FF [100%]`, `2 failed, 66
   deselected, 2 warnings in 0.39s`; the store method was absent and the endpoint deterministically
   returned pre-write task version 0 with a post-write baton. GREEN with the exact same command:
   `.. [100%]`, `2 passed, 66 deselected, 2 warnings in 0.33s`.

Final quality-review gates: full pytest `109 passed, 2 warnings in 0.49s`; Ruff `All checks
passed!`; mypy `Success: no issues found in 9 source files`. The two warnings remain the upstream
ADK and FastAPI/TestClient deprecations; no tests were skipped. A final post-ledger rerun returned
the same gate results, with pytest completing in `0.46s`.

## Final quality RED/GREEN remediation ledger

This pass remained on unborn `main` with the exact 22-file Phase 0 manifest (the 21 files returned
by `rg --files` plus hidden `.gitignore`). No commit, push, cloud call, or deployment occurred.
Because the repository still has no `HEAD`, the results below are contemporaneous command output;
Git cannot independently reconstruct test-before-production edit ordering.

The exact targeted selection was `UV_CACHE_DIR=/tmp/control-room-uv-cache uv run --offline pytest
tests/test_domain.py tests/test_state.py tests/test_api.py -q -k
'recursive_bounded_json_metadata or metadata_is_depth_and_size_bounded or
unsupported_recursive_container or tuple_wrapped_raw_content or
concurrent_breaker_across_shared_services or concurrent_service_initialization or
non_json_or_content_like_coordinator_payload'`. RED: `FFFFFF.FFFFFFFFFFFFFF [100%]`, `20 failed,
1 passed, 80 deselected, 2 warnings in 0.44s`. The failures proved that unsupported/non-finite
payload values and containers were accepted, tuple-wrapped evidence escaped recursive validation,
concurrent breaker calls emitted telemetry twice, concurrent service construction raised
`VersionConflict`, and malformed constructed coordinator intents either reached admission or
escaped as HTTP 409. GREEN with the exact same selection: `..................... [100%]`, `21
passed, 80 deselected, 2 warnings in 0.35s`.

The remediation uses an exact `dict`/`list`/JSON-scalar recursive allowlist with depth, node, key,
string, integer, and finite-float bounds; revalidates copied or constructed `Intent` values before
canonical serialization; runs the full breaker operation and telemetry emission inside the shared
backing transaction; and initializes all three built-in actors atomically and idempotently under
the backing lock. Deterministic `Barrier`-based tests prove same-key breaker serialization and
`Event`/`Barrier`-based prior concurrency regressions remain sleep-free.

Initial full gates after GREEN: pytest `130 passed, 2 warnings in 0.49s`; Ruff `All checks passed!`;
mypy `Success: no issues found in 9 source files`. Controller loopback reruns used ports 18000 and
18001 because port 8000 was occupied: normal smoke passed to `RUNNING`, breaker smoke passed to
`NEEDS_OPERATOR`, and both Uvicorn processes shut down cleanly. Final gate results are recorded
after the post-review exact-container hardening and import correction: pytest `130 passed, 2
warnings in 0.49s`; Ruff `All checks passed!`; mypy `Success: no issues found in 9 source files`.
The warnings remain the upstream ADK and FastAPI/TestClient deprecations; no tests were skipped.

## P1 field-specific metadata privacy RED/GREEN ledger

This single P1 pass remained on unborn `main` with the exact 22-file Phase 0 manifest. No commit,
push, cloud call, or deployment occurred. Because there is still no `HEAD`, the results below are
contemporaneous command output rather than independently reconstructable Git history.

The exact targeted command was `UV_CACHE_DIR=/tmp/control-room-uv-cache uv run --offline pytest
tests/test_api.py tests/test_state.py -q -k 'sensitive_status_from_untrusted_coordinator or
sensitive_prose_under_allowed_evidence_scalar_key or sensitive_prose_in_direct_baton or
sensitive_prose_in_outcome_baton'`. After correcting the coordinator fixture so the intended
payload boundary, rather than the earlier cycle-key guard, was exercised, RED was `FFFFFFF
[100%]`, `7 failed, 76 deselected, 2 warnings in 0.39s`. The coordinator status prose was accepted
with no failure, four allowed-looking Evidence fields persisted the prose, and both direct and
outcome baton paths persisted it. GREEN with the exact same command was `....... [100%]`, `7
passed, 76 deselected, 2 warnings in 0.40s`.

The remediation gives `status` the exact Phase 0 enum `ok`, `recorded`, `verified`, or `failed`;
requires identifier, reference, key, code, and step values to match a 1-to-128-character opaque
ASCII token grammar; rejects unknown keys per Intent, Evidence, and baton container; and permits
lists only for at most eight opaque `steps` tokens. Intent and Evidence models, copied/constructed
values revalidated by the store, `CycleOutcome` batons and failure codes, direct baton writes, and
`ActionResult` failure codes all use these field rules before canonical persistence.

The focused PASS regression `UV_CACHE_DIR=/tmp/control-room-uv-cache uv run --offline pytest
tests/test_state.py -q -k 'pass_rejects_verification_without_exact_verified_status'` returned `5
passed, 38 deselected in 0.03s`; in particular, `status=failed` still cannot authorize PASS. The
first full gates after GREEN returned pytest `148 passed, 2 warnings in 0.48s`, Ruff `All checks
passed!`, and mypy `Success: no issues found in 9 source files`. Controller loopback reruns used
ports 18010 and 18011: normal smoke passed to `RUNNING`, breaker smoke passed to `NEEDS_OPERATOR`,
and both Uvicorn processes shut down cleanly. The final post-ledger controller rerun returned pytest
`148 passed, 2 warnings in 0.48s`, Ruff `All checks passed!`, and mypy `Success: no issues found in
9 source files`; no tests were skipped.

## API summary

- `GET /health`
- `POST /tasks`
- `POST /tasks/{task_id}/cycles`
- `GET /tasks/{task_id}`
- `GET /tasks/{task_id}/evidence`
- `GET /registry`
- `POST /demo/tasks/{task_id}/breaker` only with `CONTROL_ROOM_DEMO_MODE=true`

Every cycle requires a caller-supplied `idempotency_key` and admits at most one action. The local
creation API always issues `syn_` task IDs with immutable `origin=LOCAL_API`; the prefix alone is
never accepted as provenance by the breaker.

## Safety and evidence boundaries

- Phase 0 makes no Gemini or other provider calls. The exported ADK `root_agent` is configured for
  `gemini-3.5-flash`, but runtime cycles use the injected deterministic fake coordinator.
- Memory mode is process-local and loses state on restart. Explicit Firestore mode persists bounded
  per-task, per-cycle, evidence-marker, and registry records; it is covered only by injected,
  network-free tests here, not by live cloud evidence.
- Tasks accepted by the public creation route are synthetic. Breaker mutation additionally requires
  stored `origin=LOCAL_API` and a `syn_` ID, and the breaker route is disabled by default.
- Evidence and telemetry contain identifiers, statuses, counters, and references only. Prompts,
  response bodies, content, credentials, and secrets are rejected or never recorded.
- No cloud setup, authentication, API enablement, deployment, publication, or live-secret dependency
  is part of Phase 0.

The container runs as a non-root user, pins the Linux/amd64 `python:3.12-slim` base image by
manifest digest, bounds idle HTTP keep-alive to five seconds, and the smoke client bounds every HTTP
request to five seconds. `TelemetrySink` retains only the most recent 1,000 metadata-only events.
Packaging does not select a state backend or bake credentials into the image.

## Top-level Evidence privacy RED/GREEN ledger

RED/GREEN: `UV_CACHE_DIR=/tmp/control-room-uv-cache uv run --offline pytest tests/test_domain.py tests/test_state.py -q -k 'evidence_top_level_identifiers_reject_prose_and_control_characters or evidence_kind_rejects_prose_outside_phase_zero_allowlist or evidence_kind_allows_only_current_phase_zero_kinds or append_evidence_revalidates_invalid_top_level_fields_before_mutation or apply_cycle_revalidates_invalid_top_level_evidence_atomically'` — RED: `13 failed, 3 passed, 79 deselected in 0.11s`; GREEN: `16 passed, 79 deselected in 0.04s`.
