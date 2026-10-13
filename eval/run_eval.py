"""Compare three systems on the synthetic cases and write a results table.

    python -m eval.run_eval                       # needs OPENAI_API_KEY, uses MEDSCRIBE_MODEL
    python -m eval.run_eval --repeats 3           # average over repeated runs (LLMs are not deterministic)
    python -m eval.run_eval --only c03 c05        # a subset of cases
    python -m eval.run_eval --arms baseline agent

Arms
  baseline           the original single LLM call (no tools, no history, no verification)
  agent_no_verifier  the agent loop and tools, but submit_note is accepted without grounding checks
  agent              the full system

The numbers in the README must come from running this script. Nothing here is precomputed.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from medscribe import tools
from medscribe.agent import ConsultationAgent
from medscribe.baseline import analyze_baseline
from medscribe.llm import OpenAIChat
from medscribe.verifier import Report

from .metrics import aggregate, score_case

HERE = Path(__file__).parent


class NullVerifier:
    """Ablation: accepts anything, so the agent loop runs without grounding checks."""

    def verify(self, note, *a, **k):
        return Report(n_items=0)


def run_baseline(llm, case, backend):
    t0 = time.perf_counter()
    note = analyze_baseline(llm, case["transcript"])
    return note, [], {"llm_calls": 1, "tool_calls": 0, "tool_errors": 0, "revisions": 0,
                      "elapsed_ms": round((time.perf_counter() - t0) * 1000)}, []


def _agent_runner(verifier: bool):
    def run(llm, case, backend):
        agent = ConsultationAgent(llm, backend)
        if not verifier:
            agent.verifier = NullVerifier()
        res = agent.run(case["transcript"], case.get("patient_id"))
        pruned = (res.verification or {}).get("pruned", [])
        return res.note, res.alerts, res.trajectory, pruned
    return run


ARMS = {"baseline": run_baseline, "agent_no_verifier": _agent_runner(False), "agent": _agent_runner(True)}


def load_cases(path: Path, only: list[str] | None) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if only:
        cases = [c for c in cases if any(c["id"].startswith(o) for o in only)]
    return cases


def evaluate(llm, cases: list[dict], arms: list[str], repeats: int = 1, backend=None) -> dict:
    backend = backend or tools.LocalBackend()
    results: dict = {}
    for arm in arms:
        scores, scores_nohist, runs, details = [], [], [], []
        for case in cases:
            for rep in range(repeats):
                try:
                    note, alerts, run, pruned = ARMS[arm](llm, case, backend)
                except Exception as e:  # one bad run should not abort a long eval
                    print(f"  ! {arm} {case['id']} run {rep + 1}: {type(e).__name__}: {e}", file=sys.stderr)
                    note, alerts, run, pruned = {}, [], {"status": "error"}, []
                sc = score_case(case, note, alerts)
                scores.append(sc)
                if not sc.has_history:
                    scores_nohist.append(sc)
                runs.append(run)
                details.append({"case": case["id"], "rep": rep, "violations": sc.violations, "pruned": pruned,
                                "note": note, "alerts": [a["key"] for a in alerts], "run": run})
        n = max(1, len(runs))
        results[arm] = {
            "all_cases": aggregate(scores),
            "cases_without_stored_history": aggregate(scores_nohist),
            "agent_stats": {
                "avg_llm_calls": round(sum(r.get("llm_calls", 0) for r in runs) / n, 2),
                "avg_tool_calls": round(sum(r.get("tool_calls", 0) for r in runs) / n, 2),
                "tool_success_rate": round(1 - sum(r.get("tool_errors", 0) for r in runs)
                                           / max(1, sum(r.get("tool_calls", 0) for r in runs)), 3)
                if any(r.get("tool_calls") for r in runs) else None,
                "avg_revisions": round(sum(r.get("revisions", 0) for r in runs) / n, 2),
                "failed_runs": sum(1 for r in runs if r.get("status") in ("failed", "error")),
                "avg_seconds": round(sum(r.get("elapsed_ms", 0) for r in runs) / n / 1000, 1),
                "avg_pruned_items": round(sum(len(d["pruned"]) for d in details) / n, 2),
            },
            "details": details,
        }
    return results


def to_markdown(results: dict, meta: dict) -> str:
    arms = list(results)

    def row(label, getter, fmt="{}"):
        cells = []
        for a in arms:
            try:
                v = getter(results[a])
                cells.append("n/a" if v is None else fmt.format(v))
            except KeyError:
                cells.append("n/a")
        return f"| {label} | " + " | ".join(cells) + " |"

    lines = [
        f"# Eval results ({meta['date']})", "",
        f"Model: `{meta['model']}`. Cases: {meta['cases']}. Repeats per case: {meta['repeats']}. "
        f"Synthetic transcripts only, so treat these as a regression benchmark, not clinical evidence.", "",
        "| Metric | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms),
        row("Extraction F1 (all cases)", lambda r: r["all_cases"]["extraction"]["f1"], "{:.2f}"),
        row("Extraction precision", lambda r: r["all_cases"]["extraction"]["precision"], "{:.2f}"),
        row("Extraction recall", lambda r: r["all_cases"]["extraction"]["recall"], "{:.2f}"),
        row("Extraction F1 (transcript-only cases)",
            lambda r: r["cases_without_stored_history"]["extraction"]["f1"], "{:.2f}"),
        row("Hallucination trap violations", lambda r: (
            f"{r['all_cases']['hallucination']['violations']}/{r['all_cases']['hallucination']['traps_checked']}")),
        row("Hallucination rate", lambda r: r["all_cases"]["hallucination"]["violation_rate"], "{:.1%}"),
        row("Safety alert recall", lambda r: (
            r["all_cases"]["safety_alerts"]["recall"] if r["agent_stats"]["tool_calls"] else None), "{:.2f}"),
        row("Tool-call success rate", lambda r: r["agent_stats"]["tool_success_rate"], "{:.1%}"),
        row("Avg tool calls per note", lambda r: r["agent_stats"]["avg_tool_calls"]),
        row("Avg revisions per note", lambda r: r["agent_stats"]["avg_revisions"] if r["agent_stats"]["avg_tool_calls"] else None),
        row("Avg items pruned per note", lambda r: r["agent_stats"]["avg_pruned_items"] if r["agent_stats"]["avg_tool_calls"] else None),
        row("Failed runs", lambda r: r["agent_stats"]["failed_runs"]),
        row("Avg seconds per note", lambda r: r["agent_stats"]["avg_seconds"]),
        "",
        "Notes",
        "- Hallucination rate = trap violations / traps checked. A trap is something that must not appear: an invented "
        "dose, an allergy nobody mentioned, a denied symptom, a diagnosis the doctor withheld, a drug the patient said "
        "they do not take.",
        "- Safety alert recall is n/a for the baseline because it has no interaction or allergy tools.",
        "- Some gold items (stored medications, allergies) only exist in patient history, which the baseline never "
        "sees. The transcript-only row removes that advantage.",
        "- Per-field F1 and every note are in the JSON file next to this one.",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(HERE / "cases.jsonl"))
    ap.add_argument("--only", nargs="*", help="case id prefixes, e.g. c03 c05")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--model")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args(argv)

    load_dotenv()
    llm = OpenAIChat(model=args.model)
    cases = load_cases(Path(args.cases), args.only)
    print(f"Running {len(cases)} cases x {args.repeats} repeat(s) x arms {args.arms} with {llm.model}...")
    results = evaluate(llm, cases, args.arms, args.repeats)

    meta = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": llm.model, "cases": len(cases),
            "repeats": args.repeats}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"{stamp}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2, ensure_ascii=False))
    md = to_markdown(results, meta)
    (out / "latest.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved {out / (stamp + '.json')} and {out / 'latest.md'}")


if __name__ == "__main__":
    main()
