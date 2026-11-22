#!/usr/bin/env python3
"""
Local first read: a preliminary, local-only reading of student writing.

  run   — offline except for citation identifiers sent to CourtListener /
          Cornell LII (cases, statutes) and Semantic Scholar / OpenAlex /
          doi.org (academic sources: a DOI or the title and year only).
          Extracts each submission, scrubs it through firstread/pii_gate,
          checks every citation (exists? says what the student says it
          says? any subsequent appellate history? is the source cited
          accurately?) and, when a pack is given, asks the LOCAL model for
          a preliminary rubric read.

      python -m firstread run --pack packs/example-memo.json --folder ~/Downloads/memo1
      python -m firstread run --folder ~/Downloads/memo1            # citations only
      python -m firstread run --pack packs/example-memo.json examples/sample_memo.md

  ui    — the same pipeline behind a localhost web page: point it at a
          folder, watch progress, read one report per file.

      python -m firstread ui            # http://127.0.0.1:5077

Student writing never leaves this machine except as the citation
identifiers above; the model runs on localhost through Ollama. Each run
writes a local _emissions.jsonl listing exactly what was sent where.
Output is labelled PRELIMINARY: a triage draft for the instructor, not a
grade.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env (PII_PSEUDONYM_SECRET, COURT_LISTENER_API_KEY, ...).
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True) or (REPO_ROOT / ".env"))
except ImportError:
    pass

from firstread.local_eval import RESULTS_DIR  # noqa: E402
from firstread.local_llm import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_NUM_CTX  # noqa: E402


def _print_progress(ev):
    stage = ev.get("stage")
    if stage == "file":
        print(f"\n[{ev['index']}/{ev['total']}] {ev['file']}")
    elif stage == "file_done":
        print(f"  => {ev['status']}: {ev.get('message', '')}")
    elif stage == "citations":
        print(f"  cite {ev.get('index')}/{ev.get('total')}: {ev.get('message')}")
    elif stage in ("cohort", "cohort_done"):
        print(f"\n{ev.get('message')}")
    else:
        print(f"  {stage}: {ev.get('message', '')}")


def cmd_run(args):
    from firstread.local_eval import (EvalOptions, load_pack, read_roster,
                                  run_folder)
    from firstread.local_llm import LocalLLMError

    from firstread.local_eval import detect_assignment_title, match_pack
    pack = load_pack(args.pack) if args.pack else None
    pack_path = args.pack or ""
    if not pack and not args.no_pack and args.folder:
        det = detect_assignment_title(Path(args.folder).expanduser())
        m = match_pack(det["title"])
        if m["confident"] and m["best"]:
            pack_path = m["best"]["path"]
            pack = load_pack(pack_path)
            print(f"detected assignment {det['title']!r} ({det['source']}) -> "
                  f"pack {Path(pack_path).name} {pack['title']!r}")
        elif m["best"]:
            sys.exit(f"detected {det['title']!r}; closest pack is "
                     f"{Path(m['best']['path']).name} {m['best']['title']!r} "
                     f"(score {m['score']}). Pass --pack to use it, or --no-pack.")
        else:
            print(f"detected {det['title']!r}: no local pack matches — "
                  "citations only (write one from packs/example-memo.json)")
    opts = EvalOptions(
        model=args.model, host=args.host, num_ctx=args.num_ctx,
        temperature=args.temperature, with_review=args.with_review,
        think=args.think, check_citations=not args.no_citations,
        check_history=not args.no_history,
        max_history_checks=args.max_history_checks,
        check_sources=not args.no_sources, max_sources=args.max_sources,
        support_reads=not args.no_support_reads,
        scholarly_fallback=not args.no_openalex,
        doi_registry_check=not args.no_doi_check,
        refresh_sources=args.refresh_sources,
        identities=read_roster(args.names) if args.names else [],
        calibration_root=args.calibration_root or "",
        use_calibration=not args.no_calibration,
    )
    if args.folder:
        folder, files = Path(args.folder).expanduser(), None
        if not folder.is_dir():
            sys.exit(f"not a folder: {folder}")
    elif args.files:
        files = [Path(f).expanduser() for f in args.files]
        folder = files[0].parent
    else:
        sys.exit("give --folder DIR or submission files")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = re.sub(r"[^\w.-]+", "_", str(pack["assignment_id"]) if pack
                   else folder.name)[:40]
    run_id = f"{stamp}-{label}"
    outdir = Path(args.output or f"{RESULTS_DIR}/{run_id}")
    what = ("citations + rubric" if pack and opts.check_citations else
            "rubric only" if pack else "citations only")
    print(f"{folder} -> {opts.model} (local): {what}")
    try:
        manifest = run_folder(folder, pack, opts, outdir,
                              progress=_print_progress, run_id=run_id,
                              pack_path=pack_path, files=files)
    except LocalLLMError as e:
        sys.exit(str(e))
    ok = sum(1 for f in manifest["files"] if f["status"] == "ok")
    print(f"\n{ok}/{len(manifest['files'])} evaluated. Results in {outdir}/ "
          "(start with _summary.md; class view in _next_class.md)")


# ── ui ──────────────────────────────────────────────────────────────────────

def cmd_ui(args):
    import logging
    from firstread import ui
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ui.serve(port=args.port, open_browser=not args.no_browser)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(prog="python -m firstread", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="evaluate submissions locally")
    pr.add_argument("--pack", help="assignment pack JSON (omitted: detected "
                    "from the folder / index.html and matched to a local pack)")
    pr.add_argument("--no-pack", action="store_true",
                    help="citations only — do not detect or use a pack")
    pr.add_argument("--folder", help="folder of submissions (docx/pdf/txt/md)")
    pr.add_argument("files", nargs="*", help="submission files (alternative "
                    "to --folder)")
    pr.add_argument("-o", "--output",
                    help=f"results dir (default {RESULTS_DIR}/<timestamp>-<label>/)")
    pr.add_argument("--model", default=DEFAULT_MODEL)
    pr.add_argument("--host", default=DEFAULT_HOST)
    pr.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    pr.add_argument("--temperature", type=float, default=1.0,
                    help="gemma degenerates below ~0.7 under JSON constraints")
    pr.add_argument("--names", help="optional roster file for name scrubbing "
                    "(one 'Name, email' per line)")
    pr.add_argument("--with-review", action="store_true",
                    help="also run the stage-1 formative review per paper")
    pr.add_argument("--think", action="store_true",
                    help="enable gemma4 thinking mode (~10x slower; can "
                    "exhaust the output budget on short judgments)")
    pr.add_argument("--no-citations", action="store_true",
                    help="skip the citation check entirely")
    pr.add_argument("--no-history", action="store_true",
                    help="skip the subsequent-appellate-history check")
    pr.add_argument("--max-history-checks", type=int, default=3,
                    help="citing opinions judged per citation (default 3)")
    pr.add_argument("--no-sources", action="store_true",
                    help="skip the academic-source check (Semantic Scholar / "
                    "OpenAlex / doi.org)")
    pr.add_argument("--max-sources", type=int, default=40,
                    help="academic sources checked per paper (default 40)")
    pr.add_argument("--no-support-reads", action="store_true",
                    help="don't read found sources' abstracts against the "
                    "point they're cited for")
    pr.add_argument("--no-openalex", action="store_true",
                    help="Semantic Scholar only: no OpenAlex fallback")
    pr.add_argument("--no-doi-check", action="store_true",
                    help="don't ask doi.org whether an unknown DOI is registered")
    pr.add_argument("--refresh-sources", action="store_true",
                    help="ignore the results root's saved source lookups")
    pr.add_argument("--calibration-root",
                    help="folder holding _calibration/ (default: the parent "
                    "of the results dir, i.e. the results root)")
    pr.add_argument("--no-calibration", action="store_true",
                    help="do not inject past instructor decisions")
    pr.set_defaults(func=cmd_run)

    pu = sub.add_parser("ui", help="localhost web UI for the same pipeline")
    pu.add_argument("--port", type=int, default=5077)
    pu.add_argument("--no-browser", action="store_true")
    pu.set_defaults(func=cmd_ui)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
