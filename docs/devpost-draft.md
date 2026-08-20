# Devpost write-up draft — Control Room

Track: **Fortified Enterprise Fleet** · All Things Agentic Hackathon
Status: DRAFT — the operator pastes this into Devpost; no account, terms, or
submission actions are performed by automation.

## Inspiration

Long-horizon agents fail in one predictable way: they keep going. They retry
until something gives, they double-execute under crashes, and when a step
fails they narrate success anyway. Enterprises don't need a smarter agent as
much as they need a **control plane that makes those failure modes impossible**.

## What it does

Control Room runs institutional agent work as **bounded, policy-authorized,
exactly-once cycles** over durable state:

- A coordinator (Gemini 3.5 Flash via Google's ADK, behind a swappable seam
  with a provider-free deterministic default) proposes the next step for a task.
- A deterministic policy gate (`authorize_intent`) is the only authority: role
  and action allowlists, staleness and version fencing, one action per cycle,
  three tool actions and one failure per task — then the task parks in
  `NEEDS_OPERATOR` for a human.
- A transactional state layer (Firestore) makes each cycle exactly-once under
  real concurrency: claim documents with 120-second fenced leases, atomic
  takeover of expired claims, idempotent replay of a crashed worker's bound
  intent, append-only evidence, and a global task cap.
- Verification is separated from work: a `PASS` requires per-check verification
  evidence from a registered verifier distinct from the worker.
- The public demo endpoint serves synthetic data only, by construction
  (store-level validation of provenance and titles).

## How the fleet stays fortified (criteria mapping)

### Innovation & operational utility (40%)

The novel piece is the **cost-and-trust boundary around the model**:

- Model output is treated as untrusted input end-to-end. The Gemini
  coordinator strict-parses responses into a bounded choice, rebuilds every
  infrastructure field locally, and still routes the result through the same
  policy gate as deterministic proposals. A malformed or failed model response
  can only produce `NEEDS_OPERATOR` — never fabricated progress.
- A **transactional model-call budget document** admits every model call
  before it happens (default 50/day). Budget exhausted or unreadable → the
  coordinator answers deterministically without calling Vertex. An anonymous
  public endpoint that structurally cannot amplify model spend removes a real
  operational blocker to exposing agents at all.
- Idempotency keys make client retries free: replays return the recorded
  result without re-invoking the model.

### Architectural discipline & tech stack (30%)

- Python 3.12, FastAPI, Pydantic strict models everywhere; mypy `--strict`,
  ruff, gitleaks, and 250 unit tests green; locked dependencies (`uv.lock`).
- State machine semantics proven twice: against an in-memory fake for speed,
  and via an **opt-in integration suite against the real Firestore emulator**
  that replays the concurrency-critical paths (claim/lease/takeover/fencing/
  idempotency/caps) — the real client also machine-enforces the
  all-reads-before-all-writes transaction discipline the fake cannot check.
- Decoupling: coordinator seam (deterministic ↔ Gemini), state seam
  (memory ↔ Firestore), transport seam (fake ↔ ADK/Vertex) — each swap is one
  environment variable, and every seam fails closed.

### Demo & production readiness (30%)

- Live on Cloud Run (europe-west1), public invoker only, `maxScale 1`,
  dedicated runtime service account with least-privilege roles.
- Supply chain: images built by Cloud Build with **SLSA provenance**
  (`requestedVerifyOption: VERIFIED`); the provenance's source-material SHA-256
  matches the submitted `git archive` hash; Artifact Registry vulnerability
  scanning enabled; deploys are **by digest only** with prior revisions kept
  ready as instant rollbacks.
- Observability: `severity>=ERROR` log alert emailing the owner; billing
  export dataset in BigQuery; alert-only budget on the project.
- Reproducibility: the README spin-up was re-run from a fresh clone before
  submission (install → tests → lint → typecheck → live normal + breaker
  smokes). An audit ledger outside the repo records every cloud mutation and
  the metadata of every bounded model request (never response bodies).

## Technologies

Google ADK (`google-adk`), Gemini 3.5 Flash on Vertex AI (global endpoint),
Cloud Run, Firestore (native mode, delete protection + 7-day PITR), Cloud
Build + Artifact Registry (SLSA provenance, vulnerability scanning), Cloud
Monitoring/Logging, BigQuery, FastAPI, Pydantic, uv, pytest, mypy, ruff,
gitleaks.

## Data sources

Synthetic data only. Task titles are store-validated to a `synthetic_*`
pattern; evidence references are metadata-only opaque tokens. No real
institutional data touches the system.

## Prior-work disclosure (required reading for judges)

Control Room is a **new standalone repository created during the contest
window**. It deliberately builds on design concepts from the author's earlier
private project `hermes-operator-controls` — specifically metadata-only
telemetry, bounded tool/failure budgets, and typed verification evidence.
Those ideas were **reimplemented from scratch here; no code was copied** from
the prior repository. All Gemini usage, cloud spend, and deployments during
development were bounded by a written authorization contract and recorded in
an audit ledger.

## What we learned

- The emulator-backed suite caught what no fake could: real Firestore rejects
  reads placed after writes in a transaction, and real commits abort under
  contention — retry-purity is a design constraint, not a nicety.
- `gemini-3.5-flash` is not served from every Vertex region; model access
  belongs on the `global` endpoint while data and compute stay regional.
- A "one bounded canary request" discipline (retries=0, temperature 0,
  thinking disabled, metadata-only recording) turns model integration from a
  leap of faith into an auditable step.

## What's next

Multi-tenant actor registries, verifier pools with quorum policies, and a
first-class operator console for the `NEEDS_OPERATOR` queue.
