"""Command line entry point.

    python -m medscribe                          # built-in sample transcript
    python -m medscribe visit.txt                # a .txt or .docx transcript
    python -m medscribe visit.txt --patient P-1001 --save
    python -m medscribe visit.txt --baseline     # the original single-call version
    python -m medscribe --trace                  # print the agent's steps
    python -m medscribe --json                   # machine-readable output
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from . import config
from .samples import SAMPLE_TRANSCRIPT


def read_transcript(path: str | None) -> str:
    if not path:
        print("No transcript given, using the sample transcript.", file=sys.stderr)
        return SAMPLE_TRANSCRIPT
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    if path.lower().endswith(".docx"):
        try:
            from docx import Document
        except ImportError:
            sys.exit("Reading .docx needs python-docx: pip install python-docx")
        return "\n".join(p.text.strip() for p in Document(path).paragraphs if p.text.strip())
    with open(path, encoding="utf-8") as f:
        return f.read()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="medscribe", description="Summarise a doctor-patient consultation.")
    parser.add_argument("transcript", nargs="?", help="Path to a .txt or .docx transcript")
    parser.add_argument("--patient", help="Patient id, enables stored history (try P-1001)")
    parser.add_argument("--save", action="store_true", help="Write the verified note to patient memory")
    parser.add_argument("--baseline", action="store_true", help="Use the single-call baseline instead of the agent")
    parser.add_argument("--mcp", action="store_true", help="Run the tools through the MCP server")
    parser.add_argument("--trace", action="store_true", help="Print the agent's steps")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of markdown")
    parser.add_argument("--out", help="Also write the markdown note to this path")
    args = parser.parse_args(argv)

    load_dotenv()
    from .llm import MissingAPIKey, OpenAIChat
    from .render import to_markdown

    transcript = read_transcript(args.transcript)
    if not transcript.strip():
        sys.exit("Empty transcript.")
    llm = OpenAIChat()

    try:
        if args.baseline:
            from .baseline import analyze_baseline
            note = analyze_baseline(llm, transcript)
            payload = {"status": "baseline", "note": note}
            markdown = to_markdown(note)
        else:
            from .agent import ConsultationAgent
            from .mcp_backend import make_backend
            backend = make_backend("mcp" if args.mcp else config.TOOL_BACKEND)
            try:
                result = ConsultationAgent(llm, backend).run(transcript, args.patient, save=args.save)
            finally:
                backend.close()
            payload = result.to_dict(include_trace=args.trace)
            markdown = to_markdown(result.note, verification=result.verification)
            if result.status == "failed":
                print(f"Agent failed: {result.error}", file=sys.stderr)
    except MissingAPIKey as e:
        sys.exit(str(e))

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(markdown)
        if args.trace:
            for e in payload.get("trace", []):
                print(f"[{e['at_ms']:>6} ms] {e['type']:<7} {e.get('name', '')} "
                      f"{'ok' if e.get('ok', True) else 'ERROR'}", file=sys.stderr)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(markdown)


if __name__ == "__main__":
    main()
