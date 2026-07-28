# Instructly

An AI guidance tool that teaches users how to use desktop software (starting
with OpenShot Video Editor) by ingesting the app's documentation and,
later, guiding the user step-by-step through real tasks.

All five core modules are built: `ingestion/` (PDF -> chunks -> classify ->
extract -> embed -> Supabase), `retrieval/` (query -> embedding -> pgvector
search -> routed result), `perception/` (window -> UI Automation tree ->
element matching), `guidance/` (screenshot -> highlight annotation -> saved
PNG), and `chat/` (ties all of the above into a conversational loop via
`GuidanceSession` + a CLI). This is the full core MVP loop; a real GUI on
top of `chat/session.py` is a separate future session.

**Caveat on `perception/`**: this project's perception session was
originally meant to be built from a real audit script + a live JSON run
against OpenShot, to pin down exact field shapes from observed data. That
input was never provided (twice asked for, never arrived), so
`perception/uia_tree.py`'s field shapes (`control_type`, `name`,
`automation_id`, `class_name`, `bounding_box`, `is_enabled`, `depth`)
follow the names originally specified, using the `uiautomation` package's
Control properties directly with no reshaping — a reasonable default, not
a verified-against-real-data shape. True it up against a real audit run if
you get one.

All LLM inference (classification, extraction, chat) and embeddings run
**locally via [Ollama](https://ollama.com)** — there is no Anthropic or
OpenAI dependency anywhere in this app. Supabase + pgvector is used as the
vector store.

## Project layout

```
instructly/
  ingestion/
    chunker.py       # PDF -> text -> sections, via a configurable heading regex
    ollama_client.py # schema-constrained generation (validate + retry) and embeddings
    classifier.py     # content_type / grounding_confidence classification
    extractor.py       # type-specific structured payload extraction
    embedder.py         # embeddings, with a hard dimension check
    pipeline.py           # orchestrates the above; CLI entry point
  retrieval/
    search.py   # embed query (reuses ingestion/embedder.py) -> pgvector RPC search; CLI entry point
    router.py   # organizes search() results by intent: procedural/reference/unknown + ui_glossary context
  perception/
    uia_tree.py     # get_ui_tree(): flatten a window's UI Automation tree (uiautomation package)
    matcher.py       # find_element(): ui_hint -> live control, exact -> fuzzy -> glossary-assisted retry
    live_state.py      # get_active_window_ui_tree(): whichever window currently has OS focus
    manual_test.py        # live-GUI check; run directly, not via pytest
  guidance/
    capture.py    # window screenshot (PIL.ImageGrab + DWM extended frame bounds)
    annotate.py    # draws the highlight box + label callout on a screenshot
    render.py        # capture -> annotate -> save to a temp PNG; returns the path
    manual_test.py     # live-GUI visual check; run directly, not via pytest
  chat/
    session.py    # GuidanceSession: retrieval + perception + guidance tied into one conversation
    cli.py          # command-line chat loop; python -m chat.cli --app-name OpenShot
    manual_test.md    # written walkthrough for a real end-to-end check (not automatable)
  sql/          # schema migrations
  tests/
  config.py     # env var loading
  requirements.txt
  .env.example
```

## Prerequisites

- **Python 3.11+**
- **Ollama** installed and running locally (tested against server 0.32.4,
  `ollama` Python client 0.6.1):
  ```bash
  ollama serve          # if not already running as a service
  ollama pull llama3.1          # chat / classification / extraction
  ollama pull nomic-embed-text  # embeddings
  ```
  Verify the server is up: `curl http://localhost:11434/api/version`
- A **Supabase** project with the `pgvector` extension available (enabled
  automatically by `sql/001_init.sql`).

## Setup

```bash
# from the instructly/ directory
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
# then edit .env with real SUPABASE_URL / SUPABASE_KEY
```

## Database setup

Run the `sql/` migrations, in order, against your Supabase project (SQL
editor in the Supabase dashboard, or `psql`/`supabase db push`):

- **`001_init.sql`** — enables the `vector` extension, creates
  `knowledge_chunks`, an ivfflat index on `embedding` for cosine similarity
  search, and a `pgvector_enabled()` helper used by the connection test.
- **`002_match_knowledge_chunks.sql`** — creates `match_knowledge_chunks(...)`,
  a Postgres RPC function used by `retrieval/search.py`. PostgREST (what
  supabase-py's `.table()`/`.select()` calls go through) can't express a
  `<=>` vector-distance `order by`, so similarity search goes through this
  RPC instead of a plain table query.

## Embedding dimension

`knowledge_chunks.embedding` is `vector(768)`, matching the **actual**
output dimension of `nomic-embed-text` as run locally via Ollama on this
machine (confirmed with a throwaway script calling `ollama.embed(...)`,
not assumed from documentation). **If `OLLAMA_EMBED_MODEL` is ever changed
to a model with a different output dimension, this column must be updated
to match exactly and the table recreated** — existing embeddings from a
different model/dimension are not compatible.

## `knowledge_chunks.payload` shapes

`payload` is `jsonb`; its shape depends on `content_type`:

- **`procedural`**
  ```json
  {
    "goal": "string",
    "steps": [
      { "instruction": "string", "ui_hint": { "...": "..." } }
    ],
    "prerequisites": ["string"]
  }
  ```
- **`ui_glossary`**
  ```json
  {
    "element_name": "string",
    "context": "string",
    "description": "string"
  }
  ```
- **`reference`**
  ```json
  {
    "entity_name": "string",
    "properties": [
      { "name": "string", "description": "string", "range": "string | null" }
    ]
  }
  ```
- **`conceptual`**
  ```json
  { "summary": "string" }
  ```

`embedding` is computed from a short goal/title string, **not** the raw
`payload`.

## Running the ingestion pipeline

```bash
# Quick smoke test on the first 10 sections — review output before committing
# to a full run against a large PDF, since local model quality needs
# eyeballing early:
python -m ingestion.pipeline --pdf path/to/openshot_docs.pdf --app-name "OpenShot" --limit 10 --dry-run

# Full run once the sample output looks right:
python -m ingestion.pipeline --pdf path/to/openshot_docs.pdf --app-name "OpenShot"
```

Notes:

- `--dry-run` prints what would be inserted instead of writing to Supabase.
- `--limit N` processes only the first N sections.
- `--heading-pattern` overrides the section-heading regex for docs that
  don't use OpenShot's `X.Y.Z Title` numbering (see
  `ingestion/chunker.py:DEFAULT_HEADING_PATTERN`). The default requires at
  least two numeric levels (`1.6`, not bare `1.`) so it doesn't collide
  with numbered UI-glossary lists inside a section body.
- Each section is classified into one of the four `content_type`s, then
  extracted with a type-specific prompt into the matching payload shape.
  `procedural` chunks are additionally classified for
  `grounding_confidence` (`menu` vs `canvas`).
- Local models are less reliable at strict JSON than hosted frontier
  models: every generation call is schema-validated with pydantic and
  retried up to twice with a stricter prompt on failure
  (`ingestion/ollama_client.py`).
- If a chunk still fails after retries, it's logged to `failed_chunks.jsonl`
  (section number, title, and reason) and the run continues — one bad
  chunk never kills the whole ingestion.

## Running retrieval

```bash
python -m retrieval.search "how do I export a video" --app-name OpenShot
```

Prints the routed result: the query's intent (`procedural` / `reference` /
`unknown`), the primary match formatted for that intent, and `ui_glossary`
context fetched alongside it.

Notes:

- `search(query, app_name, top_k=5, content_types=None)`
  (`retrieval/search.py`) embeds `query` with **the same**
  `ingestion/embedder.py` / `ingestion/ollama_client.py` used at ingestion
  time — query and stored vectors must live in the same embedding space —
  then calls the `match_knowledge_chunks` Postgres RPC (see
  `sql/002_match_knowledge_chunks.sql`), optionally filtered by
  `content_type`.
- `route(query, app_name, top_k=5)` (`retrieval/router.py`) organizes one
  `search()` call by the top result's `content_type`: a `procedural` top
  match is returned with its full ordered steps; a `reference` top match is
  condensed into a short snippet instead of a step sequence; anything else
  is `"unknown"` intent but the top match is still surfaced. `ui_glossary`
  matches for the same query are always fetched too (for the not-yet-built
  grounding module), in a clean `{element_name, context, description, ...}`
  shape.
- This is single-shot: one query in, one routed result out. No "current
  step" or conversation memory yet — that's chat orchestration, for a
  later session.
- **Local embeddings need eyeballing**: nomic-embed-text (or whatever
  `OLLAMA_EMBED_MODEL` is set to) may cluster similarity differently than a
  hosted model. Run a handful of real queries against your ingested
  OpenShot data with the CLI above and sanity-check the results before
  trusting this for anything downstream.

## Running perception

Windows-only. No LLM — this module talks to the Windows UI Automation API
directly via the `uiautomation` package.

```bash
python -m perception.manual_test
```

Prompts you to open OpenShot, walks its UI Automation tree, then tries to
resolve a couple of hardcoded example `ui_hint`s (e.g. matching "Export
Video") against that tree and prints what it found — name, bounding box,
confidence, and which matching stage resolved it (`exact` / `fuzzy` /
`glossary_exact` / `glossary_fuzzy`), or `NOT FOUND`.

Notes:

- `uia_tree.get_ui_tree(window_title_substring, max_depth=6)` flattens a
  window's control tree depth-first; each node is
  `{control_type, name, automation_id, class_name, bounding_box, is_enabled, depth}`,
  with `bounding_box` screen-absolute — matching what `guidance/render.py`
  expects. Individual controls that raise on property access (stale/
  disappearing COM elements — a real risk on a live, possibly-animating
  UI) are skipped rather than aborting the whole walk.
- `matcher.find_element(ui_hint, ui_tree, glossary_context=None)` tries,
  in order: an exact case-insensitive name match, then a fuzzy match
  (`rapidfuzz`) above `SIMILARITY_THRESHOLD`, then — only if
  `glossary_context` is given and both of those fail — resolves the hint
  against glossary entries' `element_name` first, and retries exact/fuzzy
  matching against the tree using that canonical name. Returns `None`
  rather than a low-confidence guess when nothing clears the bar; callers
  (`chat/session.py`) must handle `None` explicitly.
- `live_state.get_active_window_ui_tree(max_depth=6)` returns
  `(window_title, ui_tree)` for whichever window currently has OS focus,
  walking directly from the resolved control rather than re-searching by
  title (titles aren't guaranteed unique). `chat/session.py` doesn't use
  this — it already knows the target app by name — but it's there for
  callers that don't.
- **Caveat**: built without a real audit-script run to verify field shapes
  against (see the top-level caveat above). Sanity-check
  `perception/manual_test.py`'s output against what you'd expect from
  OpenShot's actual controls.

## Running guidance

Windows-only. No LLM, no RAG — this module just takes a window title and a
bounding box and produces an annotated screenshot.

```bash
python -m guidance.manual_test
```

Prompts you to open OpenShot, then captures it, draws a highlight + label
around a hardcoded example box, saves a PNG to a temp file, and opens it
with the default image viewer so you can visually confirm the highlight is
positioned correctly. **That example box is still a placeholder guess**
(see the NOTE at the top of `guidance/manual_test.py`) — it predates
`perception/` existing and hasn't been swapped for a real
`perception.matcher.find_element()` result yet.

Notes:

- `capture.capture_window_screenshot(window_title_substring)` uses
  `PIL.ImageGrab` (a direct `PIL.Image` return, no conversion step) over
  `mss`, finding the window via `pywin32` and reading its true visible
  bounds via `DwmGetWindowAttribute(..., DWMWA_EXTENDED_FRAME_BOUNDS)` —
  plain `GetWindowRect` includes a few pixels of invisible resize-border
  padding on Windows 10/11 that would otherwise offset every annotation
  from where perception's UI Automation bounding boxes say it should be.
  Falls back to a full-screen capture (with a `warnings.warn`) if no
  matching window is found.
- `annotate.annotate_target(image, bounding_box, label=None)` expects
  `bounding_box` **relative to the image's own pixel grid**, not
  screen-absolute — it only ever sees the pixels it's handed, with no way
  to know where on screen that image came from. The label callout
  position is clamped so it always stays fully within the image, flipping
  from above the box to below it when there isn't room above.
- `render.render_guidance(window_title_substring, bounding_box, label)` is
  the orchestrator: it expects `bounding_box` in **screen-absolute**
  coordinates (matching what perception's UI Automation output will use),
  converts it to image-relative using the window origin from
  `capture.get_window_bounds`, then calls `annotate_target` and saves the
  result to a temp PNG, returning its path.

## Running chat

Windows-only (perception + guidance are). This is the full loop: retrieval
-> perception -> guidance, tied together by `GuidanceSession`, with the
local Ollama model doing reply classification when needed.

```bash
python -m chat.cli --app-name OpenShot
```

A plain command-line REPL, not a polished UI — it exists to prove the loop
works end to end, not as the intended user experience (that's a real GUI,
in a future session, built on top of `chat/session.py`). See
`chat/manual_test.md` for a specific scenario to walk through and what to
check at each step, including how step-transition latency should feel.

Notes:

- `GuidanceSession.ask(user_message)`: with no active workflow, a message
  is a new question — `procedural` intent starts walking that workflow
  from step 0, `reference` intent is answered directly with no step-
  walking. With an active workflow, a message is a reply to "did that
  work?": a cheap keyword classifier (`_rule_based_classify`) handles the
  common cases ("done"/"next"/"ok" vs "stuck"/"help"/"no", tokenized to
  avoid false positives like "no" matching inside "know") and only an
  ambiguous reply falls back to an Ollama call
  (`ingestion/ollama_client.py:generate_json`) to classify it as advance /
  stuck / a new question entirely. This split exists because local
  inference is slow enough that classifying every single turn with it
  would make the guide feel laggy.
- `GuidanceSession.present_current_step()` resolves the current step's
  `ui_hint` against the live UI tree — via `perception.uia_tree.get_ui_tree`
  targeting the known app name directly, not
  `perception.live_state.get_active_window_ui_tree` (which would grab
  whatever window currently has OS focus, easily the terminal this CLI is
  running in instead of the target app). If grounding resolves, it renders
  a highlighted screenshot and returns its path; if not, it still returns
  the step's instruction text with `image_path: None` and an explanatory
  `note` — never a silent failure.

## Running tests

```bash
# Unit tests — mocked Ollama/Supabase/win32/UI-Automation calls, no
# external services or live GUI needed:
pytest tests/ --ignore=tests/test_connection.py
```

`tests/test_connection.py` is the exception: it hits real services and
requires real values in `.env`, both SQL migrations already applied to
your Supabase project, Ollama running locally, and both `OLLAMA_MODEL` and
`OLLAMA_EMBED_MODEL` pulled. Every other test (ingestion, retrieval,
perception, guidance, and chat) mocks the Ollama/Supabase/win32/UI
Automation clients and runs offline — `pytest tests/` runs everything,
including `test_connection.py`, if you have real credentials set up.

`perception/manual_test.py`, `guidance/manual_test.py`, and
`chat/cli.py` (walked through via `chat/manual_test.md`) are all separate
from the pytest suite on purpose: they need a real GUI on screen and are
meant to be run and eyeballed directly, not executed automatically.
