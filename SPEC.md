# Spec: Self-Healing Technical Documentation

## 1. Overview

A GitHub Action that monitors a codebase, detects when code changes make documentation inaccurate, identifies the specific stale sections, and either:

- **auto-generates a PR** with corrected docs (high confidence), or
- **flags the discrepancies** for human review (low confidence).

**Problem:** every team's docs are perpetually stale. **Goal:** an installable, CI-native tool that keeps docs in sync with code, demonstrating embeddings, retrieval, LLM generation, and production deployment.

## 2. Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Best ecosystem for embeddings (`sentence-transformers`), ChromaDB, PyGithub |
| Embeddings | Hugging Face `BAAI/bge-small-en-v1.5` via `sentence-transformers` | Free, runs locally on CPU in the Action, no API key |
| Vector store | ChromaDB (file-based) | No server; persists to disk |
| LLM | Primary: Groq (free tier, e.g. `gpt-oss-120b`). Fallback: Gemini (free tier). Paid alternatives: Claude Sonnet, GPT-4o | Free, fast, good code understanding; fallback keeps runs completing when the primary is rate-limited. Providers are swappable. User supplies the API keys |
| Git integration | `git` CLI + GitHub REST over `httpx` | PR creation and diff parsing; one HTTP client already used for the LLM, easy to test with a mock transport |
| CI/CD | GitHub Actions | Native integration, free tier |
| Classifier (optional) | Default: the configured LLM with structured JSON output. Experimental: a System One model such as Laya (open source, local) | Fast, calibrated labels for narrow decisions; swappable and evaluated in Phase 5 |
| Packaging | Docker (prebuilt image on GHCR, embedding model baked in) | Reproducible runs; avoids per-run model download and image build |

## 3. Architecture

```
PR opened/updated
      │
      ▼
[Diff Parser] ──► changed code chunks
      │
      ▼
[Change Filter] ──► meaningful changes only
      │
      ▼
[Link Graph Lookup] ──► suspect doc sections
      │
      ▼
[LLM Staleness Verifier] ──► confirmed stale + diagnosis
      │
      ▼
[Repair Engine: generate → validate] ──► corrected section + confidence
      │
      ├── high confidence ──► new branch + fix PR
      └── low confidence  ──► draft with TODOs + review comment
      │
      ▼
[PR Summary Comment]
```

## 4. Functional Requirements

Status legend: `[x]` implemented and tested, `[ ]` pending. Update a box in the same commit that completes the requirement.

### Phase 1: Code-to-Docs Mapping (Days 1–4)

- [x] **FR-1.1 Code chunking.** Parse the codebase into semantic chunks: function signatures with docstrings, class definitions, API endpoint definitions, configuration schemas, CLI command definitions. Each chunk has a stable ID (`file path + function/class name`). *Status: Python implemented; JS/TS pending.*
- [x] **FR-1.2 Doc sectioning.** Split markdown docs by heading. Each section records: heading path (e.g. `Configuration > Environment Variables`), raw content, and a list of code references it mentions (function names, class names, config keys, CLI commands).
- [x] **FR-1.3 Link graph.**
  - Heuristic links: a doc section mentioning a function/class/key that exists in the codebase is linked to that chunk.
  - Embedding links: embed both code chunks and doc sections with the local embedding model; link pairs with cosine similarity above a configurable threshold. The model is configurable (e.g. `jinaai/jina-embeddings-v2-base-code` if code-to-doc matching quality is poor).
- [x] **FR-1.4 Persistence.** Store the code-to-docs graph as a JSON file in the repo (the index). It is queried to find docs affected by code changes.

### Phase 2: Change Detection Pipeline (Days 4–7)

- [x] **FR-2.1 Diff parsing.** On every PR, extract changed files and added/removed/modified lines; map each change to the code chunks it affects. *Implemented by chunking the base and head versions of each changed file and comparing chunks, rather than mapping diff lines, so removed code and renames are handled exactly.*
- [x] **FR-2.2 Meaningful-change filter.** *(Rules implemented; ambiguous body edits go to the FR-2.2a classifier, and are kept when none is configured.)*
  - Skip: comment-only changes, whitespace changes, internal refactors that don't change behavior, test file changes.
  - Keep: API signature changes, configuration changes, new/removed features, behavioral changes to existing functions.
- [x] **FR-2.2a Classifier backend.** Narrow classification decisions go through a single `Classifier` interface returning a label plus a probability:
  - *Meaningful change?* (the FR-2.2 filter): comment/whitespace/test-only/internal refactor vs. API, config or behavior change.
  - *Simple or complex?* (FR-3.3 routing): renamed parameter or new default vs. new feature or removed capability.
  - Default backend: the configured LLM with structured JSON output. Optional backends: a non-generative "System One" model (open-source Laya, run locally; or hosted Jev). Backends are chosen by config, and the pipeline must work with the default alone. *Status: interface, LLM default backend and `build_classifier` selection implemented (rules decide added/removed as complex; code is sent to the model as delimited untrusted data; if the model is unavailable or replies badly, the change is kept at confidence 0.5). Laya and Jev raise `NotImplementedError` until the Phase 5 evaluation (FR-5.1a). LLM probabilities are self-reported and weakly calibrated.* Classifier probabilities may feed the confidence score (see FR-3.3).
- [x] **FR-2.3 Suspect identification.** For each meaningful change, query the link graph for linked doc sections ("suspects"). Suspects are grouped per section, ranked by link score x change weight (to prioritize LLM calls under rate limits), and sections the PR also edited are reported separately instead of flagged. Added chunks have no links in the base index, so they flag nothing yet.
- [x] **FR-2.4 LLM staleness verification.** For each suspect, send the LLM the old code, new code, and doc section. It determines whether the doc is still accurate and, if not, explains specifically what is wrong. This filters false positives. *Status: implemented. One call per suspect, most likely stale first. The model must quote each wrong statement; quotes are checked against the section text, and a stale verdict with no findable quote is marked ungrounded and its confidence capped. Out-of-capacity or malformed replies leave suspects `unverified` for human review. Run it with `doc-healer check --verify` (settings via `LLM_*` environment variables). Prompt quality is untested against live models.*

### Phase 3: Doc Repair Engine (Days 7–10)

- [x] **FR-3.1 Targeted corrections.** For each confirmed stale section, send the LLM the current section, the new code, and the diagnosis. It rewrites only the stale parts, preserving original style, tone, and structure, and must not touch accurate parts. *Status: implemented. The model returns `find`/`replace` edits that are applied mechanically, so text outside an edit is preserved byte for byte. Edits are rejected if the `find` text is missing or ambiguous, the edit is a no-op or overlaps another, or (when the diagnosis quoted the wrong statements) it touches text the diagnosis did not call wrong. Failed and skipped corrections carry the unchanged original. Applying corrections to files is FR-4.2. Prompt quality is untested against live models.*
- [x] **FR-3.2 Validation pass.** A second LLM pass checks that the corrected doc (a) accurately describes the new code, (b) preserved already-correct parts, (c) matches the document's writing style. This is the quality gate before PR creation. *Status: implemented in two stages. A free structural lint rejects corrections that change the number of code fences or headings or introduce new URLs, with no LLM call. Otherwise one LLM review judges accuracy, preservation and style; all three must pass. A correction that could not be checked (capacity, budget, malformed reply) is `unchecked` and must go to a human, never to an auto-fix. The reviewer's confidence is self-reported; combining it with the verifier's confidence is FR-3.3. Prompt quality is untested against live models.*
- [x] **FR-3.3 Correction modes.** Confidence determines the mode:
  - *Simple* (renamed parameter, updated default value) → auto-fix at high confidence.
  - *Complex* (new feature, removed capability) → draft with clear `TODO` markers and request human review.
  - *Status: implemented (`modes.py`).* Each stale section gets one of three modes. **auto_fix**: the change is simple, the correction is validated, and confidence meets the threshold (default 0.80). **draft**: a usable correction that a person must review; its text carries `<!-- TODO(doc-healer): ... -->` markers above each edited region (never inside a code block or table). Used when the change is complex or its complexity is unknown, confidence is below the threshold, the diagnosis quoted nothing findable in the section, or some proposed edits were rejected. **flag**: no usable correction (none generated, validation failed, or it could not be checked); reported with the diagnosis and no text to apply.
  - *Confidence* is the lowest of the verifier's, the validator's and (for simple changes) the classifier's confidence, so any one doubtful signal blocks an auto-fix. Added and removed chunks are complex by rule; with no classifier configured, modified chunks count as unknown, so nothing is auto-fixed.

### Phase 4: GitHub Action (Days 10–12)

*Progress: the pipeline driver is done (`driver.py`, `doc-healer run`). It reads the link index as it was at the PR's base (the committed one, else one built from a temporary checkout of base), runs detect, filter, suspects, verify, repair, validate and mode selection, and returns a JSON-ready result with a summary (accurate, auto-fixed, drafts, flagged, unverified, already updated) and LLM usage. It writes nothing to the repository and does not talk to GitHub; FR-4.1 to FR-4.3 build on it.*

- [ ] **FR-4.1 Packaging.** `Dockerfile` + `action.yml` defining:
  - **Inputs:** LLM API key, confidence threshold, auto-merge for high-confidence fixes.
  - **Outputs:** list of stale sections found, corrections generated.
  - **Trigger:** PRs that modify code files.
- [x] **FR-4.2 PR workflow.**
  - High-confidence: create a new branch, apply corrections, open a PR with a clear description of what changed and why.
  - Low-confidence: comment on the original PR listing sections needing review, with links to them.
  - *Status: implemented as a library (`workflow.py`, `apply.py`, `render.py`, `github.py`); the Action entry point that feeds it the GitHub event is FR-4.1.* Auto-fixes are written to the doc files only where a section's text still matches the analyzed version, committed on `doc-healer/fix-<PR number>` with an explicit bot author, pushed, and opened as a PR into the original PR's branch (merging it updates the original PR). Drafts, flags, unverified sections and fixes that could not be applied are listed in one comment on the original PR with links to each section. Re-runs refresh the branch, the fix PR and the comment instead of duplicating them. LLM-written text is neutralized (no pings, links or HTML) before it reaches GitHub. The checkout is restored afterward; `dry_run` writes, pushes and posts nothing.
- [x] **FR-4.3 PR summary comment.** On every triggering PR, post a summary, e.g. "Doc Check Results: 3 sections verified accurate, 1 auto-fixed (see PR #42), 2 flagged for review." Include links to everything. *Status: implemented (`render.py`, `workflow.py`).* One comment per PR, edited in place on re-runs (hidden marker). It opens with the specified headline, then lists what needs a person (links, status, why, suggested edit), and collapsible lists of auto-fixed, verified-accurate and already-updated sections, each linking to its heading line. A footer reports the providers used, and tags each section with its provider when a fallback was involved. It is cut back in steps to stay under GitHub's comment size limit. `comment_mode="needed"` skips the comment when nothing was fixed or needs a person.

### Phase 5: Real-Repository Evaluation (Days 12–13)

- [ ] **FR-5.1** Fork a well-documented open-source project (FastAPI, Pydantic, or similar), install the Action, and deliberately make code changes that should invalidate docs; verify correct detection and reasonable fixes.
- [ ] **FR-5.1a Classifier evaluation.** Run the LLM classifier and Laya on the same labeled cases (meaningful vs. not, simple vs. complex) and compare accuracy, calibration and latency. Adopt a System One backend only if it clearly beats the LLM default; skip Jev unless it wins by enough to justify a hosted, proprietary dependency and an extra API key.
- [ ] **FR-5.2** Measure and report in the README:
  - True positives (stale docs correctly identified)
  - False positives (accurate docs flagged as stale)
  - False negatives (missed staleness)
  - Correction quality (are the fixes actually right?)

### Phase 6: Portfolio Polish (Days 13–14)

- [ ] **FR-6.1** Publish to the GitHub Actions Marketplace (free).
- [ ] **FR-6.2** Record a demo under 3 minutes: code change pushed → Action runs → PR comment appears → auto-generated doc-fix PR. Lead with the problem ("every team's docs are perpetually stale"); end with the result.

### Non-Functional Requirements: LLM Usage

- [x] **NFR-1 Rate-limit resilience.** All LLM calls retry with exponential backoff on HTTP 429 and transient errors, honoring `Retry-After` when present.
- [ ] **NFR-2 Batching and budgeting.** Batch and prioritize suspect sections so large PRs stay within per-minute and per-day token limits; cap the number of LLM calls per run. *Status: per-run call cap and suspect ranking exist; batching of suspects into prompts is pending.*
- [ ] **NFR-3 Provider fallback.** If the primary provider (Groq) is still rate-limited after retries, switch to the configured fallback (Gemini) for the remaining calls. Fallback is opt-in; the PR summary notes which provider handled which sections. *Status: client-side fallback implemented (opt-in, stays on the fallback once the primary is exhausted); the PR summary now names the providers used and tags sections verified via a fallback; the private-repo warning belongs to the Action.*
- [ ] **NFR-3a Graceful degradation.** If all providers are exhausted, do not fail the check: post a PR comment listing the sections that could not be verified, flagged for human review. *Status: the client raises `LLMUnavailable`; posting the "could not verify" comment is pending the Action.*
- [ ] **NFR-4 Provider abstraction.** A single interface wraps providers so switching (e.g. Groq → Claude Sonnet) requires only config changes. *Status: one interface with an OpenAI-compatible implementation (Groq, Gemini, OpenAI presets, or any compatible endpoint); a native Anthropic adapter is pending.*
- [ ] **NFR-5 Data privacy.** Document the data-handling policy of each supported provider; default to a provider that does not use API data for training, and warn users before using free tiers that may do so on private repos. Because Gemini's free tier may use requests to improve Google products, the Gemini fallback must be explicitly enabled and the Action logs a warning when it is used on a private repo.

## 5. Data Model

**CodeChunk**
| Field | Description |
|---|---|
| `id` | Stable: `<file path>::<function/class name>` |
| `kind` | function / class / endpoint / config / cli |
| `signature`, `docstring`, `source` | Extracted content |
| `embedding` | Stored in ChromaDB |

**DocSection**
| Field | Description |
|---|---|
| `id` | `<doc path>#<heading path>` |
| `heading_path` | e.g. `Configuration > Environment Variables` |
| `content` | Raw markdown |
| `code_refs` | Names of functions, classes, config keys, CLI commands mentioned |
| `embedding` | Stored in ChromaDB |

**Link** (in the JSON index): `{ code_chunk_id, doc_section_id, method: "heuristic" | "embedding", score }`

## 6. Action Interface (draft)

```yaml
inputs:
  llm_provider:         # primary; default groq; also anthropic, openai, gemini
  llm_model:            # provider-specific model name
  llm_api_key:          # required
  classifier_backend:   # default llm; experimental: laya, jev
  fallback_provider:    # optional; recommended gemini; unset = no fallback
  fallback_model:       # provider-specific model name
  fallback_api_key:     # required if fallback_provider is set
  embedding_model:      # default BAAI/bge-small-en-v1.5
  confidence_threshold: # default 0.8; at or above → auto-fix PR
  auto_merge:           # default false
  similarity_threshold: # default 0.70; embedding link cutoff (cosine). Tune in Phase 5
outputs:
  stale_sections:       # list of stale doc sections found
  corrections:          # list of corrections generated
on: pull_request (paths: code files)
```

Values marked "default" are suggestions and may be tuned.

## 7. Success Criteria

- Correctly detects stale sections on a real OSS repo with reported precision/recall.
- Auto-fix PRs are accurate, minimal, and style-consistent.
- Low-confidence cases never auto-merge; they surface as drafts/comments.
- Installable from the Marketplace with a single workflow file.
- Demo video under 3 minutes.

## 8. Self-Healing Loop

1. **Detect:** diff → meaningful change → link graph → suspects.
2. **Diagnose:** LLM verifies staleness and explains the discrepancy.
3. **Repair:** minimal targeted rewrite → validation gate.
4. **Route:** confidence decides auto-fix PR vs. human review.
5. **Re-index:** after merge, update the link graph so the index stays current.

## 9. Open Questions

- Does Groq's free tier, with the Gemini fallback, hold up on large PRs, and when should the Action recommend a paid provider?
- Should the Gemini fallback be disabled by default on private repos?
- Does Groq's data policy suit private repos? (Verify before documenting.)
- Does a System One classifier (Laya, locally; or Jev) beat the LLM default on the "meaningful change" and "simple vs. complex" decisions? Evaluate in Phase 5 (FR-5.1a). Also verify its APIs, input limits, and how it handles code diffs, since current coverage is thin and benchmark claims are mostly company-reported.
- Is an open model (e.g. `gpt-oss-120b`) reliable enough at minimal, style-preserving rewrites? Evaluate in Phase 5.
- Validate the default thresholds (confidence 0.8, similarity 0.70) against Phase 5 results and adjust.
- How and when is the link-graph index rebuilt and committed?
