"""Compare models on the same compliance reviews, as LangSmith experiments on one dataset.

Each case in `model_cases.jsonl` is a made-up code change, with the references a correct review should cite, or none if the
change is fine. The whole core runs on it: triage, retrieval, the review and the summary.

It uses the real rules, as the agent does. Start the Handbook RAG first (FCA_MCP_URL, by default the local server). The
Consumer Credit Act is read from CCA_MCP_URL if set, or else from a local server over stdio. Then, from the repo root:

    uv run --no-sync python 04_fca_compliance_agent/evals/model_eval.py --models openai:gpt-6-luna openai:gpt-6.2-sol

It uses LANGSMITH_API_KEY and the gateway settings in `.env`. Each case makes up to three model calls for each model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

from fca_compliance_agent.adapters.github_pr import split_diff
from fca_compliance_agent.adapters.llm import LangChainComplianceAnalyzer, load_repo_env
from fca_compliance_agent.adapters.regulatory import (
    MCPConsumerCreditActAdapter,
    MCPFCAHandbookAdapter,
    MCPServerConfig,
)
from fca_compliance_agent.core import build_core_analysis_subgraph
from fca_compliance_agent.domain import AnalysisInput
from fca_compliance_agent.mcp_server.bootstrap import DEFAULT_FCA_MCP_URL

CASES = [json.loads(line) for line in (Path(__file__).parent / "model_cases.jsonl").read_text("utf-8").splitlines()]
SCOPE = ["CONC", "PRIN 2A", "SYSC", "Consumer Credit Act 1974"]
BREACH = ("non_compliant", "potential_breach")


def _ports() -> tuple[Any, Any]:
    fca, cca = os.environ.get("FCA_MCP_URL") or DEFAULT_FCA_MCP_URL, os.environ.get("CCA_MCP_URL")
    cca_config = MCPServerConfig(url=cca) if cca else MCPServerConfig(
        transport="stdio", command=sys.executable, args=("-m", "fca_compliance_agent.mcp_server.server"),
    )  # fmt: skip
    return (
        MCPFCAHandbookAdapter(server_config=MCPServerConfig(url=fca)),
        MCPConsumerCreditActAdapter(server_config=cca_config),
    )


async def review(diff: str, model: str) -> dict[str, Any]:
    """Run the core on one change. Returns references and counts only, not rule text."""
    fca, cca = _ports()
    core = build_core_analysis_subgraph(fca_port=fca, cca_port=cca, analyzer_port=LangChainComplianceAnalyzer(model))
    given = AnalysisInput(
        target_id="eval", source_type="pr_review", title="Change to a regulated consumer lending app",
        summary="A UK consumer credit firm's codebase.", regulatory_scope=SCOPE, units=split_diff(diff),
    )  # fmt: skip
    started = time.perf_counter()
    report = (await core.ainvoke({"analysis_input": given}))["findings_report"]
    seconds = time.perf_counter() - started
    findings, quotes, valid = [], 0, 0
    for found in report.findings:
        for citation in found.fca_citations:
            if citation.excerpt:
                quotes += 1
                valid += (await fca.check_citation(citation.reference, citation.excerpt)).ok
        findings.append({
            "title": found.title, "status": found.status, "severity": found.severity,
            "references": [c.reference for c in found.fca_citations + found.cca_citations],
        })  # fmt: skip
    return {
        "verdict": report.overall_verdict, "findings": findings, "quotes": quotes, "quotes_valid": valid,
        "seconds": round(seconds, 1),
    }  # fmt: skip


def _cites(reference: str, expected: str) -> bool:
    """"CONC 5.2A.4R" is in "CONC 5.2A", but "CONC 5.2AB.1R" is not."""
    reference, expected = reference.upper(), expected.upper()
    return reference.startswith(expected) and not reference[len(expected) :][:1].isalnum()


# Each evaluator returns no score where it does not apply, so that LangSmith leaves the case out of its average.
def problem_found(outputs: dict, reference_outputs: dict) -> dict:
    if not reference_outputs["expected_references"]:
        return {"key": "problem_found", "score": None}
    return {"key": "problem_found", "score": float(any(f["status"] in BREACH for f in outputs["findings"]))}


def right_rule_cited(outputs: dict, reference_outputs: dict) -> dict:
    expected = reference_outputs["expected_references"]
    if not expected:
        return {"key": "right_rule_cited", "score": None}
    cited = [r for f in outputs["findings"] if f["status"] in BREACH for r in f["references"]]
    return {"key": "right_rule_cited", "score": float(any(_cites(r, e) for r in cited for e in expected))}


def no_false_alarm(outputs: dict, reference_outputs: dict) -> dict:
    if reference_outputs["expected_references"]:
        return {"key": "no_false_alarm", "score": None}
    return {"key": "no_false_alarm", "score": float(not any(f["status"] in BREACH for f in outputs["findings"]))}


def quotes_exact(outputs: dict) -> dict:
    share = outputs["quotes_valid"] / outputs["quotes"] if outputs["quotes"] else None
    return {"key": "quotes_exact", "score": share}


EVALUATORS = [problem_found, right_rule_cited, no_false_alarm, quotes_exact]


def _dataset(client: Any, name: str) -> None:
    if client.has_dataset(dataset_name=name):
        print(f"Using the existing dataset {name!r}")
        return
    client.create_dataset(name, description="Made-up code changes, and the rules a compliance review should cite.")
    client.create_examples(
        dataset_name=name,
        examples=[
            {
                "inputs": {"id": c["id"], "diff": c["diff"]},
                "outputs": {"expected_references": c["expected_references"], "why": c["why"]},
            }
            for c in CASES
        ],
    )
    print(f"Created the dataset {name!r} with {len(CASES)} cases")


def _row(model: str, rows: list[dict]) -> str:
    def share(key: str) -> str:
        scores = [s for r in rows if (s := r["scores"].get(key)) is not None]
        return f"{sum(scores):.0f}/{len(scores)}"

    quotes = sum(r["outputs"]["quotes"] for r in rows if r["outputs"])
    valid = sum(r["outputs"]["quotes_valid"] for r in rows if r["outputs"])
    seconds = [r["outputs"]["seconds"] for r in rows if r["outputs"]]
    errors = sum(not r["outputs"] for r in rows)
    return (f"| `{model}` | {share('problem_found')} | {share('right_rule_cited')} | {share('no_false_alarm')} | "
            f"{valid}/{quotes} | {median(seconds) if seconds else 0:.0f} s | {errors} |")  # fmt: skip


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    load_repo_env()
    from langsmith import Client, aevaluate

    if not os.environ.get("LANGSMITH_PROJECT"):
        raise SystemExit("LANGSMITH_PROJECT is not set: copy .env.example to .env and fill it in")
    client = Client()
    name = f"{os.environ['LANGSMITH_PROJECT']}-compliance-model-eval"
    _dataset(client, name)
    lines = []
    for model in args.models:

        async def target(inputs: dict, model: str = model) -> dict:
            return await review(inputs["diff"], model)

        results = await aevaluate(
            target, data=name, evaluators=EVALUATORS, experiment_prefix=model.split(":")[-1],
            metadata={"model": model, "fca_source": "mcp"},
            max_concurrency=args.concurrency, client=client,
        )  # fmt: skip
        rows = []
        async for result in results:
            outputs = result["run"].outputs if not result["run"].error else None
            scores = {e.key: e.score for e in result["evaluation_results"]["results"]}
            rows.append({"id": result["example"].inputs["id"], "outputs": outputs, "scores": scores})
            missed = [k for k in ("problem_found", "right_rule_cited", "no_false_alarm") if scores.get(k) == 0]
            if missed or not outputs:
                cited = [r for f in (outputs or {}).get("findings", []) for r in f["references"]]
                print(f"  {model} {rows[-1]['id']}: {missed or result['run'].error} {cited}")
        lines.append(_row(model, rows))
        print(f"{model}: experiment {results.experiment_name}", flush=True)
    print("\n| Model | Problem found | Right rule cited | No false alarm | Exact quotes | Median time | Errors |")
    print("|---|---|---|---|---|---|---|\n" + "\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
