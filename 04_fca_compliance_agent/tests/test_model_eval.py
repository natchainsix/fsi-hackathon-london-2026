"""The model eval's cases and evaluators. No network and no model."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import model_eval as ev  # noqa: E402


def outputs(*found: tuple[str, list[str]]) -> dict:
    return {"findings": [{"status": s, "references": r} for s, r in found], "quotes": 0, "quotes_valid": 0}


EXPECTED, CLEAN = {"expected_references": ["CONC 5.2A", "CCA 1974 s.87"]}, {"expected_references": []}


class DefaultModelTests(unittest.TestCase):
    def test_the_compliance_agent_has_its_own_model(self) -> None:
        from fca_compliance_agent.adapters.llm import langchain_llm

        def resolved(**env: str) -> str:
            with mock.patch.object(langchain_llm, "load_repo_env"), mock.patch.dict(os.environ, env, clear=True):
                return langchain_llm.resolve_default_model_name()

        self.assertEqual(resolved(), "openai:gpt-6.2-sol")
        self.assertEqual(resolved(MODEL_NAME="openai:gpt-6-luna"), "openai:gpt-6.2-sol")
        self.assertEqual(resolved(COMPLIANCE_MODEL_NAME=""), "openai:gpt-6.2-sol")
        self.assertEqual(resolved(COMPLIANCE_MODEL_NAME="anthropic:other"), "anthropic:other")


class ModelEvalTests(unittest.TestCase):
    def test_cases_are_usable(self) -> None:
        self.assertEqual(len({c["id"] for c in ev.CASES}), len(ev.CASES))
        self.assertTrue(all(len(ev.split_diff(c["diff"])) == 1 for c in ev.CASES))
        self.assertGreaterEqual(sum(not c["expected_references"] for c in ev.CASES), 5)

    def test_reference_matching_respects_boundaries(self) -> None:
        self.assertTrue(ev._cites("CONC 5.2A.4R", "CONC 5.2A"))
        self.assertTrue(ev._cites("CCA 1974 s.87", "CCA 1974 s.87"))
        self.assertFalse(ev._cites("CCA 1974 s.87A", "CCA 1974 s.87"))
        self.assertFalse(ev._cites("CONC 5.2AB.1R", "CONC 5.2A"))

    def test_problem_cases(self) -> None:
        right = outputs(("potential_breach", ["CONC 5.2A.4R"]))
        wrong = outputs(("non_compliant", ["CONC 9.1.1R"]))
        unsure = outputs(("needs_review", ["CONC 5.2A.4R"]))
        self.assertEqual([ev.problem_found(o, EXPECTED)["score"] for o in (right, wrong, unsure)], [1, 1, 0])
        self.assertEqual([ev.right_rule_cited(o, EXPECTED)["score"] for o in (right, wrong, unsure)], [1, 0, 0])
        self.assertIsNone(ev.no_false_alarm(right, EXPECTED)["score"])

    def test_clean_cases(self) -> None:
        self.assertEqual(ev.no_false_alarm(outputs(("observation", [])), CLEAN)["score"], 1)
        self.assertEqual(ev.no_false_alarm(outputs(("non_compliant", [])), CLEAN)["score"], 0)
        self.assertIsNone(ev.problem_found(outputs(), CLEAN)["score"])

    def test_quotes(self) -> None:
        self.assertIsNone(ev.quotes_exact({"quotes": 0, "quotes_valid": 0})["score"])
        self.assertEqual(ev.quotes_exact({"quotes": 4, "quotes_valid": 3})["score"], 0.75)


if __name__ == "__main__":
    unittest.main()
