# Control Room — Architecture

Control Room is a fail-closed control plane for long-horizon agents: every cycle
of agent work is proposed, policy-authorized, executed once, and durably
recorded — or it does not happen at all. The public demo serves synthetic data
only.

## Runtime architecture

```mermaid
flowchart TB
    subgraph Public["Public surface (Cloud Run, allUsers -> run.invoker only)"]
        API["FastAPI app<br/>POST /tasks · POST /tasks/{id}/cycles<br/>GET status/evidence · /health · demo breaker"]
    end

    subgraph Service["ControlRoomService (one bounded cycle per call)"]
        SEAM{{"Coordinator seam<br/>CONTROL_ROOM_COORDINATOR"}}
        DET["DeterministicCoordinator<br/>(provider-free default)"]
        GEM["GeminiCoordinator<br/>ADK root_agent · gemini-3.5-flash<br/>temperature 0 · max 512 tokens · thinking off"]
        BUDGET["Transactional model budget<br/>model_budget doc · N calls/day<br/>exhausted or unreadable -> deterministic fallback,<br/>model never called"]
        POLICY["authorize_intent (deterministic policy gate)<br/>role/action allowlist · staleness · version fencing<br/>1 action/cycle · 3 tool actions · 1 failure -> NEEDS_OPERATOR"]
    end

    subgraph State["StateStore (memory | Firestore)"]
        TX["Per-task transactional documents<br/>claim/lease 120s · owner fencing · takeover<br/>idempotent cycle records · append-only evidence<br/>256-task global cap · reads-before-writes"]
        FS[("Firestore (default)<br/>delete protection · 7d PITR")]
    end

    VERTEX["Vertex AI (global endpoint)"]

    API --> Service
    SEAM --> DET
    SEAM --> GEM
    GEM -->|"admit first"| BUDGET
    GEM -->|"one bounded request"| VERTEX
    GEM -->|"untrusted output<br/>strict mechanical parse"| POLICY
    DET --> POLICY
    POLICY -->|"authorized intent only"| TX
    TX --> FS

    GEM -.->|"any model error<br/>PolicyDenied (fail closed)"| POLICY
```

Key properties:

- **Model output is untrusted input.** The Gemini coordinator strict-parses the
  response into a bounded choice (action + opaque token), rebuilds every
  infrastructure field locally, and the result still passes the same
  `authorize_intent` gate as the deterministic path. A malformed or failed model
  response persists `NEEDS_OPERATOR` — progress is never fabricated.
- **Anonymous callers cannot amplify cost.** Task creation is capped by a
  transactional 256-task counter; model calls are capped by a transactional
  daily budget document; every task is capped at 3 tool actions and 1 failure.
- **Exactly-once cycles under real concurrency.** Cycle claims carry a 120-second
  lease with owner fencing; expired claims are taken over atomically and a
  crashed worker's bound intent is replayed, not re-invented. An opt-in
  integration suite replays these paths against the real Firestore emulator,
  which also machine-enforces the reads-before-writes transaction discipline.
- **Privacy-minimized evidence.** Evidence and batons carry metadata-only
  references (opaque tokens, no bodies); verification of a `PASS` requires a
  distinct registered verifier with per-check verification evidence.

## Build and deployment provenance

```mermaid
flowchart LR
    GIT["git commit<br/>(clean tree)"] -->|"git archive<br/>SHA-256 recorded"| CB["Cloud Build<br/>docker step · CLOUD_LOGGING_ONLY<br/>requestedVerifyOption VERIFIED"]
    CB -->|"SLSA v0.1 in-toto provenance<br/>materials sha256 == archive sha256"| AR["Artifact Registry<br/>+ vulnerability scanning"]
    AR -->|"deploy by digest only"| RUN["Cloud Run control-room<br/>maxScale 1 · 512Mi · SA hackathon@<br/>prior revisions kept ready as rollback"]
    RUN --> OBS["Observability<br/>severity>=ERROR log alert -> owner email<br/>billing export dataset (BigQuery)"]
```

The audit ledger (kept outside the repository) records every cloud mutation:
build IDs, source-archive SHA-256s, image digests, deploy revisions, IAM state,
and metadata (never bodies) of each bounded model request.
