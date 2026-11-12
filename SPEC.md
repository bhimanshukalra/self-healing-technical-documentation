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
| Git integration | PyGithub + `git diff` | PR creation and diff parsing |
| CI/CD | GitHub Actions | Native integration, free tier |
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

### Phase 1: Code-to-Docs Mapping (Days 1–4)

- **FR-1.1 Code chunking.** Parse the codebase into semantic chunks: function signatures with docstrings, class definitions, API endpoint definitions, configuration schemas, CLI command definitions. Each chunk has a stable ID (`file path + function/class name`).
- **FR-1.2 Doc sectioning.** Split markdown docs by heading. Each section records: heading path (e.g. `Configuration > Environment Variables`), raw content, and a list of code references it mentions (function names, class names, config keys, CLI commands).
- **FR-1.3 Link graph.**
  - Heuristic links: a doc section mentioning a function/class/key that exists in the codebase is linked to that chunk.
  - Embedding links: embed both code chunks and doc sections with the local embedding model; link pairs with cosine similarity above a configurable threshold. The model is configurable (e.g. `jinaai/jina-embeddings-v2-base-code` if code-to-doc matching quality is poor).
- **FR-1.4 Persistence.** Store the code-to-docs graph as a JSON file in the repo (the index). It is queried to find docs affected by code changes.

### Phase 2: Change Detection Pipeline (Days 4–7)

- **FR-2.1 Diff parsing.** On every PR, extract changed files and added/removed/modified lines; map each change to the code chunks it affects.
- **FR-2.2 Meaningful-change filter.**
  - Skip: comment-only changes, whitespace changes, internal refactors that don't change behavior, test file changes.
  - Keep: API signature changes, configuration changes, new/removed features, behavioral changes to existing functions.
- **FR-2.3 Suspect identification.** For each meaningful change, query the link graph for linked doc sections ("suspects").
- **FR-2.4 LLM staleness verification.** For each suspect, send the LLM the old code, new code, and doc section. It determines whether the doc is still accurate and, if not, explains specifically what is wrong. This filters false positives.

### Phase 3: Doc Repair Engine (Days 7–10)

- **FR-3.1 Targeted corrections.** For each confirmed stale section, send the LLM the current section, the new code, and the diagnosis. It rewrites only the stale parts, preserving original style, tone, and structure, and must not touch accurate parts.
- **FR-3.2 Validation pass.** A second LLM pass checks that the corrected doc (a) accurately describes the new code, (b) preserved already-correct parts, (c) matches the document's writing style. This is the quality gate before PR creation.
- **FR-3.3 Correction modes.** Confidence determines the mode:
  - *Simple* (renamed parameter, updated default value) → auto-fix at high confidence.
  - *Complex* (new feature, removed capability) → draft with clear `TODO` markers and request human review.

### Phase 4: GitHub Action (Days 10–12)

- **FR-4.1 Packaging.** `Dockerfile` + `action.yml` defining:
  - **Inputs:** LLM API key, confidence threshold, auto-merge for high-confidence fixes.
  - **Outputs:** list of stale sections found, corrections generated.
  - **Trigger:** PRs that modify code files.
- **FR-4.2 PR workflow.**
  - High-confidence: create a new branch, apply corrections, open a PR with a clear description of what changed and why.
  - Low-confidence: comment on the original PR listing sections needing review, with links to them.
- **FR-4.3 PR summary comment.** On every triggering PR, post a summary, e.g. "Doc Check Results: 3 sections verified accurate, 1 auto-fixed (see PR #42), 2 flagged for review." Include links to everything.

### Phase 5: Real-Repository Evaluation (Days 12–13)

- **FR-5.1** Fork a well-documented open-source project (FastAPI, Pydantic, or similar), install the Action, and deliberately make code changes that should invalidate docs; verify correct detection and reasonable fixes.
- **FR-5.2** Measure and report in the README:
  - True positives (stale docs correctly identified)
  - False positives (accurate docs flagged as stale)
  - False negatives (missed staleness)
  - Correction quality (are the fixes actually right?)

### Phase 6: Portfolio Polish (Days 13–14)

- **FR-6.1** Publish to the GitHub Actions Marketplace (free).
- **FR-6.2** Record a demo under 3 minutes: code change pushed → Action runs → PR comment appears → auto-generated doc-fix PR. Lead with the problem ("every team's docs are perpetually stale"); end with the result.

### Non-Functional Requirements: LLM Usage

- **NFR-1 Rate-limit resilience.** All LLM calls retry with exponential backoff on HTTP 429 and transient errors, honoring `Retry-After` when present.
- **NFR-2 Batching and budgeting.** Batch and prioritize suspect sections so large PRs stay within per-minute and per-day token limits; cap the number of LLM calls per run.
- **NFR-3 Provider fallback.** If the primary provider (Groq) is still rate-limited after retries, switch to the configured fallback (Gemini) for the remaining calls. Fallback is opt-in; the PR summary notes which provider handled which sections.
- **NFR-3a Graceful degradation.** If all providers are exhausted, do not fail the check: post a PR comment listing the sections that could not be verified, flagged for human review.
- **NFR-4 Provider abstraction.** A single interface wraps providers so switching (e.g. Groq → Claude Sonnet) requires only config changes.
- **NFR-5 Data privacy.** Document the data-handling policy of each supported provider; default to a provider that does not use API data for training, and warn users before using free tiers that may do so on private repos. Because Gemini's free tier may use requests to improve Google products, the Gemini fallback must be explicitly enabled and the Action logs a warning when it is used on a private repo.

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
  fallback_provider:    # optional; recommended gemini; unset = no fallback
  fallback_model:       # provider-specific model name
  fallback_api_key:     # required if fallback_provider is set
  embedding_model:      # default BAAI/bge-small-en-v1.5
  confidence_threshold: # default 0.8; at or above → auto-fix PR
  auto_merge:           # default false
  similarity_threshold: # default 0.65; embedding link cutoff (cosine). Tune in Phase 5
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
- Is an open model (e.g. `gpt-oss-120b`) reliable enough at minimal, style-preserving rewrites? Evaluate in Phase 5.
- Validate the default thresholds (confidence 0.8, similarity 0.65) against Phase 5 results and adjust.
- How is "confidence" computed (LLM self-report, validation-pass score, or both)?
- How and when is the link-graph index rebuilt and committed?
