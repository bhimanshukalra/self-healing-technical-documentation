# doc-healer

A GitHub Action that keeps documentation in step with code. When a pull request changes code, doc-healer
finds the documentation sections that may now be wrong, checks each one with an LLM, and then either

- opens a small fix pull request into your PR's branch (simple, high-confidence cases), or
- lists the section in a comment, with a link and a suggested edit, for a person to handle.

Every PR it runs on gets one summary comment, e.g.

> **Doc Check Results:** 3 sections verified accurate, 1 auto-fixed (see #42), 2 flagged for review.

## Use it

Copy [`examples/doc-healer.yml`](examples/doc-healer.yml) to `.github/workflows/doc-healer.yml` and add your LLM
key as a secret. The checkout step must use `fetch-depth: 0` and the PR's head commit, so doc-healer can compare the
PR with where it branched off.

For the best first run, commit a link index with `doc-healer build .` (it writes `.doc-healer/index.json`).
Without one, doc-healer builds it from the PR's base commit on every run.

### Inputs

| Input | Default | Purpose |
|---|---|---|
| `llm_api_key` | | Key for the LLM provider. Without it, suspect sections are listed but not verified or fixed. |
| `llm_provider` / `llm_model` | `groq` / preset | Provider (`groq`, `gemini`, `openai`) and model. |
| `fallback_provider`, `fallback_api_key`, `fallback_model` | off | A second provider used if the first is rate limited. Check its data-use terms. |
| `max_llm_calls` | unlimited | Cap on LLM calls per run. |
| `confidence_threshold` | `0.8` | Confidence a simple fix needs to be applied automatically. |
| `auto_merge` | `false` | Merge the fix PR immediately. It targets the PR's own branch. |
| `classifier_backend` | `llm` | `llm`, or `none` (rules only; nothing modified is auto-fixed). |
| `comment_mode` | `always` | `needed` comments only when something was fixed or needs review. |
| `index_path` | `.doc-healer/index.json` | Where the committed link index lives. |
| `embeddings` | `true` | When building the index, also link by meaning, not only by name. |
| `github_token` | `github.token` | Pushes the fix branch, opens the PR, comments. |
| `fail_on_error` | `false` | Fail the step on unexpected errors. By default doc-healer only warns. |

### Outputs

`stale_sections` (JSON list), `corrections` (JSON list: section, file, mode, confidence, applied), `summary`
(JSON counts), `fix_pr_url`.

## How it works

1. **Link graph.** Code (functions, classes, endpoints, config, CLI commands) is linked to the documentation sections
   that mention it, by name and by embedding similarity. The index is read as it was at the PR's base.
2. **Change detection.** The PR's changed code is compared chunk by chunk with its base. Comment-only, whitespace,
   docstring-only and private changes are skipped; ambiguous edits are judged by the LLM.
3. **Suspects.** Linked sections that the PR did not already edit are the suspects, most likely stale first.
4. **Verification.** The LLM must quote each statement that is now wrong; quotes are checked against the section.
5. **Repair.** The LLM proposes find/replace edits that are applied mechanically, so text outside an edit is
   preserved byte for byte.
6. **Validation.** A structural lint and a second LLM review check accuracy, preservation and style.
7. **Routing.** Simple, validated, high-confidence fixes are applied; anything else becomes a draft with TODO
   markers or a flag. Confidence is the lowest of the verifier's, validator's and classifier's.

## Behavior to know about

- **It never blocks your CI.** A missing key, rate limits, a failed push or an API error degrade to a comment or the run
  summary. Only invalid inputs fail the step.
- **Fork PRs** get the comment (where the token allows) but no fix PR, because the token is read-only.
- **Your code is sent to the LLM provider.** Some free tiers may use requests to improve their products; check the
  provider's terms before using it on private code.
- **Python code only** for now; documentation must be Markdown.

## Status

Built test-first against scripted LLM replies and a fake GitHub API. It has not yet been run against live LLM providers,
real GitHub, or a large real repository, so treat model prompts, thresholds and the published image as unproven.
See [SPEC.md](SPEC.md) for the design and what is done.
