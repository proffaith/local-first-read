# Local first read

A first read of student writing that runs on your own computer. Point it at a
folder of submissions and, for each paper, it:

1. **extracts the text** from .docx (including tracked changes, tables and
   footnotes), .pdf, .txt or .md;
2. **scrubs student names and identifiers** before anything else sees the text;
3. **checks every citation.** Does the case or statute exist (CourtListener,
   Cornell LII)? Does it say what the student says it says (judged by the local
   model against the opinion text)? Has a higher court reversed or vacated it?
   Academic sources in the reference list or footnotes are looked up in
   Semantic Scholar, OpenAlex and doi.org;
4. **gives a preliminary rubric read** with a local model (gemma4 through
   Ollama): a suggested level per criterion, the reason, and what to check
   before you agree.

Everything it produces is labelled **PRELIMINARY**. It is a triage draft for
the instructor, not a grade, and not AI detection.

This is a snapshot of the local evaluation tool in
[ProfFaith](https://proffaith.com), published so others can see how it works
and adapt it. It is not kept in sync with ProfFaith. The prompts are the same
ones ProfFaith uses.

## What leaves your computer

| What | Where it goes |
|---|---|
| The student's paper | Nowhere. The scrubbed text goes only to the Ollama model on `localhost`. `firstread/local_llm.py` refuses any other host. |
| Case and statute citations | The citation itself (volume, reporter, page, party names, year) goes to CourtListener and Cornell LII so the authority can be found and its opinion fetched. |
| Academic sources | A DOI or similar identifier, or the source's title and year, goes to Semantic Scholar, OpenAlex and doi.org. Author names and the sentence the source is cited for are never sent. |

Each run writes `_emissions.jsonl` in its results folder, listing exactly what
went where for every file. The judgment calls ("does this case support the
proposition?", "how did the higher court treat it?") run only on the local
model; this snapshot has no code path that sends student text to a cloud
model.

The reports quote (scrubbed) student text. Keep the results folder wherever
your institution keeps student records.

## Prerequisites

- **A computer that can run a 26-billion-parameter model.** Tested on an
  Apple-silicon Mac with 32 GB of memory. The model alone is about 17 GB. With
  less memory it will be very slow or fail to load. Linux with a comparable GPU
  should work but has not been tested.
- **Python 3.12 or newer** (tested on 3.12, 3.13 and 3.14).
- **[Ollama](https://ollama.com/download)**, running, with the model pulled
  (about a 17 GB download):

  ```bash
  ollama pull gemma4:26b
  ```

- **A free CourtListener account and API token** for case lookups:
  <https://www.courtlistener.com/help/api/rest/>.
- Optional: a free **Semantic Scholar API key** (by request at
  <https://www.semanticscholar.org/product/api>). Without one, academic-source
  lookups share a public pool that is often rate-limited, so expect some "not
  checked" rows.
- Optional: an **OpenAlex API key** for a daily budget of your own.

The folder picker button in the web UI is macOS-only. On other systems, type
the folder path.

## Install

```bash
git clone <this repository's URL> local-first-read
cd local-first-read
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env`:

- `PII_PSEUDONYM_SECRET`: a long random value, for example from
  `python3 -c "import secrets; print(secrets.token_hex(32))"`. It keys the
  pseudonyms that replace student names. Keep it private.
- `COURT_LISTENER_API_KEY`: your CourtListener token.
- The optional keys, if you have them.

## Try it on the sample

`examples/sample_memo.md` is a short, made-up predictive memo by a made-up
student. Four of its five citations are two real cases with their parallel
citations (*Palsgraf* and *MacPherson*). The fifth, *Hartwell v. Simmons
Grocery Co.*, 412 F.3d 1187, is invented.

```bash
.venv/bin/python -m firstread run --pack packs/example-memo.json \
    --names examples/roster.txt examples/sample_memo.md
```

On a 32 GB Mac this took about a minute once the model was loaded (the first
call of a session adds up to a minute while Ollama loads the model). The run
finds all five citations, confirms the four real ones and flags *Hartwell* as
"could not be found (possibly inaccurate or fabricated)". The student's name
appears nowhere in the outputs. Results land in `results/<timestamp>-<pack>/`:

| File | What it is |
|---|---|
| `_summary.md` | One line per paper: citation problems and suggested levels. Start here. |
| `<paper>.md` / `.json` | The full report for one paper. |
| `<paper>.txt` | Exactly what the model saw, after scrubbing. Read it when a result surprises you. |
| `_next_class.md` | Class-level patterns across the folder: what many students missed. |
| `_emissions.jsonl` | The audit log of what left the computer. |

## The web UI

```bash
.venv/bin/python -m firstread ui        # opens http://127.0.0.1:5077
```

It runs the same pipeline behind a local page: choose a folder, watch
progress, read one report per paper, record where you agree or disagree with
the suggested levels, and keep a grading queue. It binds to 127.0.0.1 only.

## Assignment packs

Without a pack the tool checks citations only. With one, it also gives a rubric
read. A pack is a JSON file in `packs/` (or in a folder you name in
`FIRSTREAD_PACKS`). `packs/example-memo.json` is a complete example. The fields:

| Field | Required | What it is |
|---|---|---|
| `title` | yes | The assignment's name. Also used to match a folder to its pack. |
| `course_title` | | Groups packs in the UI and keeps calibration per course. |
| `purpose`, `task`, `criteria` | | The instructions (Transparency in Learning and Teaching, TILT, layout). |
| `ai_policy`, `objectives`, `sources` | | More context for the model; `objectives` and `sources` are lists of strings. |
| `rubric` | for a rubric read | Either the simple shape `{"criteria": [{"name": ..., "levels": {"Exemplary": "...", ...}}]}` or the grouped shape in the example, with your own level names. |
| `readings` | | What students were given: `[{"title": ..., "file": "readings/week2.md"}]` or `{"title": ..., "text": "..."}`. Files are relative to the pack. Only include student-facing material, never an answer key. |
| `intentional_errors` | | For find-the-flaws exercises: `{"document_title": ..., "planted_errors": [{"id", "severity", "category", "locator", "whats_wrong", "correct_fix"}]}`, plus `defective_document_text` (the flawed original) at the top level. The model checks which planted errors each student found and fixed. |

You can also keep an **expected moves** checklist beside a pack
(`<pack>.moves.json`, editable from the UI's Packs page) for open writing: the
moves a strong answer makes, scored for presence in every paper.

## Brightspace downloads and names

A Brightspace (D2L) bulk download names each file
`<id>-<id> - Last, First - date - filename` and includes an `index.html`. The
tool reads both, pairs multiple files from the same student, and scrubs those
names automatically. For files named any other way, pass a roster (`--names
roster.txt` or the UI's roster field), one `Name, email` per line.

No scrubber is perfect. Read `<paper>.txt` when it matters.

## Calibration

When you record your own level for a criterion in the UI, it is saved in the
results folder (`_calibration/`, no student text) and shown to the model as
an example of your judgment in later runs on the same course. The Calibration
page shows how often you and the model agree.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

The tests stub every network and model call.

## Layout

```
firstread/
  local_eval.py          the pipeline: extract, scrub, check citations, rubric read, write reports
  local_llm.py           Ollama client (localhost only, JSON-schema output)
  pii_gate.py            the name and identifier scrubber
  citation_verifier.py   citation extraction and CourtListener / Cornell LII lookups
  academic_citations.py  reference lists, footnotes, DOIs
  academic_lookup.py     Semantic Scholar / OpenAlex / doi.org lookups
  rubric_prompts.py      the rubric-read prompts
  local_eval_cohort.py   pairing files by student, the class view
  local_eval_context.py  readings → prompt context
  expected_moves.py, intentional_errors.py, local_eval_calibration.py
  cli.py, ui.py, templates/
packs/                   assignment packs (example included)
examples/                a sample memo and roster
```

## License

Copyright 2026 Tim Faith. Licensed under
[Creative Commons Attribution-ShareAlike 4.0 International](LICENSE)
(CC BY-SA 4.0). `firstread/courts.json` comes from Free Law Project's
[courts-db](https://github.com/freelawproject/courts-db) under its BSD 2-Clause
license; see [NOTICE](NOTICE). Dependencies keep their own licenses (PyMuPDF,
for example, is AGPL-3.0).
