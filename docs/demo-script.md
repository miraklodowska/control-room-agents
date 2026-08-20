# Demo video script — Control Room (target 3:30, hard cap 3:40)

Rules for the recording session:

- **Fresh task IDs only.** Every task shown on camera is created live during the
  recording; `POST /tasks` mints a new `syn_<uuid>` every time, so nothing needs
  to be pre-staged. Task IDs that appeared in earlier recordings or in the audit
  ledger are burned and must not be reused on screen.
- All data on screen is synthetic (`synthetic_*` titles enforced by the store).
- Keep the Cloud Console pages pre-authenticated in separate tabs before
  recording; judges need visible Google Cloud backend proof.

Shell setup before recording (off camera):

```bash
BASE=https://control-room-231909520054.europe-west1.run.app
```

| # | Time | Shot | Script (voice) |
|---|------|------|----------------|
| 1 | 0:00–0:20 | Title slide: "Control Room — fail-closed control plane for long-horizon agents" over the runtime mermaid diagram from `docs/architecture.md`. | "Enterprise agents fail in one predictable way: they keep going. They retry, they double-spend, they invent progress. Control Room is a control plane where an agent cycle either passes a deterministic policy gate and is recorded exactly once — or the task stops and waits for an operator." |
| 2 | 0:20–0:50 | Terminal: `curl -s $BASE/health \| jq` → shows `coordinator: gemini`, `state_backend: firestore`. Then create a task live: `curl -s -X POST $BASE/tasks -H 'content-type: application/json' -d '{"title":"synthetic_demo_video","required_checks":["quality"]}' \| jq` — point at the fresh `syn_…` ID. | "This is the live Cloud Run service — public, anonymous, serving only synthetic data. The health probe discloses the active coordinator: a Gemini agent on Vertex. I'm creating a task now; note the fresh synthetic ID, and note task creation itself is capped by a transactional 256-task counter, so anonymous traffic can't flood the store." |
| 3 | 0:50–1:30 | Terminal: run one cycle with the fresh ID: `curl -s -X POST $BASE/tasks/<FRESH_ID>/cycles -d '{"idempotency_key":"video-1"}' … \| jq`. Show `state: RUNNING`, `tool_actions: 1`, the evidence entry. Re-run the same command; point out the byte-identical reply. | "One cycle: gemini-3.5-flash proposes the next bounded step through Google's ADK. The model's output is untrusted input — it's strict-parsed into a bounded choice and still has to pass the same deterministic policy gate as everything else. The write happens under a transactional claim with a 120-second fenced lease. And if I replay the same idempotency key — same result, no second model call, no double work." |
| 4 | 1:30–2:05 | Split screen: Firestore console (`control_room` collection — task doc, `model_budget` doc with `used` incremented) next to the terminal. | "Here's the same state in Firestore — delete protection and point-in-time recovery on. And this document is my favorite: a transactional model-call budget. Every model call must win a slot here first. Exhaust it and the coordinator degrades to a deterministic fallback without calling Vertex at all — a public endpoint that can never amplify model cost. Model errors don't fake progress either: they fail the cycle closed to NEEDS_OPERATOR." |
| 5 | 2:05–2:35 | Terminal: trip the breaker on the same task: `curl -s -X POST $BASE/demo/tasks/<FRESH_ID>/breaker -d '{"idempotency_key":"video-breaker"}' … \| jq` → `NEEDS_OPERATOR`, `failure_code: demo_missing_evidence`, empty evidence. Then `GET /tasks/<FRESH_ID>` read-back. | "Now the failure path, on purpose. The verifier refuses to pass work without per-check verification evidence from a distinct verifier identity. The task lands in NEEDS_OPERATOR — durably. Nothing retries behind your back; the read-back shows the failure is the recorded truth." |
| 6 | 2:35–3:10 | Cloud console tabs in sequence: Cloud Build history (build with VERIFIED provenance), Artifact Registry image digest + provenance/scanning tab, Cloud Run revisions list (current + ready rollbacks), Monitoring alert policy "Control Room ERROR logs". | "Production posture: every image is built by Cloud Build with SLSA provenance — the provenance's source hash matches the git archive we submitted. Deploys are by digest only, previous revisions stay ready as instant rollbacks, error logs page the owner by email, and billing exports to BigQuery. The whole loop — build, deploy, every bounded model request — is written to an audit ledger." |
| 7 | 3:10–3:35 | Back to the architecture diagram; zoom on the coordinator seam and budget box. | "Control Room treats the model as a brilliant, untrusted colleague: it may propose, never authorize. That's the Fortified Enterprise Fleet thesis — agents you can hand real institutional work to, because the blast radius is engineered to zero. Repo, reproducible spin-up, and the audit trail are in the submission." |
| 8 | 3:35–3:40 | End card: repo URL + "Fail closed. Prove everything." | (beat) |

Model-call budget for the recording: shots 2–5 spend exactly **one** Gemini
request (the single cycle in shot 3; the replay is served from the idempotency
record and the breaker path makes no model call). Rehearsals against production
count against the A5 ledger — rehearse locally (`CONTROL_ROOM_COORDINATOR`
unset) instead.
