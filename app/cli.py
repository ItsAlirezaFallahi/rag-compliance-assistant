"""CLI for Phase 1.

  python -m app.cli init-db
  python -m app.cli inspect sr-11-7 data/sr1107a1.pdf     # tune regexes first!
  python -m app.cli ingest sr-11-7 data/sr1107a1.pdf
  python -m app.cli ingest nist-ai-rmf data/NIST.AI.100-1.pdf
  python -m app.cli query "What are the three core elements of an effective validation framework?"
  python -m app.cli query "..." --doc sr-11-7 -k 3
"""

import argparse

from app.db import init_schema
from app.ingest.chunk import chunk_document
from app.ingest.parse import PROFILES, parse_pdf
from app.ingest.run_ingest import ingest
from app.retrieve import search


def cmd_inspect(args):
    """Dry run: show detected sections + chunk counts, no API calls."""
    sections = parse_pdf(args.pdf, args.slug)
    doc_title = PROFILES[args.slug].title
    chunks = chunk_document(sections, doc_title)
    print(f"{len(sections)} sections, {len(chunks)} chunks\n")
    for s in sections:
        n = sum(1 for c in chunks if c.section_path == s.section_path)
        preview = s.text[:90].replace("\n", " ")
        print(f"[{n} chunk(s)] {s.section_path}")
        print(f"    {preview}...\n")


def cmd_query(args):
    results = search(args.query, k=args.k, doc_slug=args.doc)
    if not results:
        print("No results.")
        return
    for i, r in enumerate(results, 1):
        print(f"--- {i}. sim={r.similarity:.3f}  {r.doc_slug} > {r.section_path} "
              f"(chunk {r.chunk_index})")
        print(r.content[:400])
        print()


def main():
    parser = argparse.ArgumentParser(prog="rag-compliance")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db")

    p = sub.add_parser("inspect")
    p.add_argument("slug", choices=PROFILES.keys())
    p.add_argument("pdf")

    p = sub.add_parser("ingest")
    p.add_argument("slug", choices=PROFILES.keys())
    p.add_argument("pdf")

    p = sub.add_parser("query")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--doc", choices=PROFILES.keys(), default=None)

    args = parser.parse_args()
    if args.command == "init-db":
        init_schema()
        print("Schema created.")
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "ingest":
        ingest(args.slug, args.pdf)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
