# chat/cli.py manual walkthrough

A repeatable, real-world check for the end-to-end loop: retrieval ->
perception -> guidance, against your real ingested OpenShot docs and a
real running OpenShot window, with local Ollama doing the thinking.

## Prerequisites

- `.env` filled in with real Supabase credentials; `sql/001_init.sql` and
  `sql/002_match_knowledge_chunks.sql` applied.
- OpenShot docs already ingested for `app_name="OpenShot"` (see
  `ingestion/pipeline.py`).
- Ollama running locally, with `OLLAMA_MODEL` and `OLLAMA_EMBED_MODEL`
  pulled.
- OpenShot itself open and visible on screen.

## Scenario: exporting a video

1. Start the loop:
   ```
   python -m chat.cli --app-name OpenShot
   ```
2. Ask: `how do I export a video`
   - Expect: intent resolves to a procedural workflow; step 1 prints with
     an instruction and (if grounding succeeds) a screenshot path. Open
     that PNG and confirm the highlight is actually on the right control
     for step 1.
3. Reply `done` (or `ok`, `yes`, `next`) to advance.
   - Expect: step 2 prints with a *new* screenshot highlighting step 2's
     target control. Confirm it's positioned correctly.
4. Reply something ambiguous, e.g. `hmm not sure` or `I think so?`
   - Expect: since this doesn't clearly land in either rule-based bucket,
     it falls through to the local Ollama classifier -- you should notice
     a longer pause here than steps 2-3 had.
5. Reply `stuck` (or `no`, `didn't work`).
   - Expect: the *same* step re-prints (index doesn't advance), with a
     "here's that step again" note ahead of the instruction.
6. Continue through 2-3 steps total, confirming each screenshot's
   highlight lands on the correct control.
7. Ask something unrelated mid-workflow, e.g. `what does resolution mean`.
   - Expect: classified as a new question (not "stuck" or "advance"),
     routed to a reference answer, and the workflow resets -- a follow-up
     like `done` afterward should NOT resume the old export workflow.

## What to look for

- Does the highlighted box in each screenshot land on the actual
  button/menu item described by that step's instruction?
- When grounding fails (`image_path: None`), is the `note` actually useful
  as a words-only fallback, or is the step instruction too vague without
  a picture?
- Do steps ever advance when you didn't mean "done" (a rule-based false
  positive), or stay stuck when you clearly said "next" (a false
  negative)? Either is a sign the keyword lists in `chat/session.py`
  (`ADVANCE_WORDS`/`STUCK_WORDS`/`*_PHRASES`) need tuning against how you
  actually phrase things.

## Latency

**Not benchmarked in the session that wrote this file** -- there's no real
Supabase project, ingested data, or running OpenShot window available in
that environment, so the loop above couldn't actually be exercised end to
end there. Time it yourself and fill in the blanks below; this is the main
signal for whether more reply classification needs to move from the Ollama
fallback back into the rule-based path.

- Time from asking a *new* question to the first step appearing (query
  embed + pgvector search + UI tree walk + matching + screenshot render):
  ___
- Time from replying `done`/`next`/etc. (rule-based path, no LLM call) to
  the next step appearing: ___
- Time from an ambiguous reply (LLM fallback path) to the next step or
  re-prompt appearing: ___

If the rule-based-path number is close to the LLM-fallback number, the
`generate_json` retry logic (up to 3 attempts on invalid/mismatched JSON
-- `ingestion/ollama_client.py`) may be firing on the classification calls
too, which is worth checking directly (e.g. temporary print/log around
`_llm_classify_reply`) before assuming it's raw model latency.
