"""
Unit tests for the rigorous error analysis subsystem, failure case suite, and report exporters.
"""

import json
from pathlib import Path

from football_cv.analytics.error_analysis import (
    BENCHMARK_CASES,
    CaseEvaluationResult,
    ErrorAnalysisExporter,
    ErrorReport,
    FailureCase,
    FailureCaseEvaluator,
    FailureCategory,
    FailureMode,
)


class TestFailureTaxonomy:
    def test_failure_categories(self):
        expected = {"detection", "tracking", "perspective_motion", "analytics"}
        actual = {c.value for c in FailureCategory}
        assert actual == expected

    def test_failure_modes(self):
        expected = {
            "ball_dropout",
            "possession_flicker",
            "track_swap_pass",
            "camera_cut_disruption",
            "superhuman_velocity",
        }
        actual = {m.value for m in FailureMode}
        assert actual == expected

    def test_benchmark_cases_catalog(self):
        assert len(BENCHMARK_CASES) == 5
        case_ids = [c.case_id for c in BENCHMARK_CASES]
        assert "TS_PASS_01" in case_ids
        assert "SH_VEL_02" in case_ids
        assert "POSS_FLK_03" in case_ids
        assert "CAM_CUT_04" in case_ids
        assert "BALL_DROP_05" in case_ids

        for c in BENCHMARK_CASES:
            assert isinstance(c, FailureCase)
            assert c.category in FailureCategory
            assert c.mode in FailureMode
            assert len(c.description) > 0
            assert len(c.root_cause) > 0
            assert len(c.mitigation_strategy) > 0


class TestEvaluationModels:
    def test_case_evaluation_result_serialization(self):
        result = CaseEvaluationResult(
            case_id="TEST_01",
            name="Test Failure Case",
            category="tracking",
            mode="track_swap_pass",
            metric_name="false_pass_count",
            unit="count",
            baseline_metric=1.0,
            mitigated_metric=0.0,
            delta=-1.0,
            improvement_percent=100.0,
            passed=True,
            details={"note": "passed"},
        )
        d = result.to_dict()
        assert d["case_id"] == "TEST_01"
        assert d["passed"] is True
        assert d["improvement_percent"] == 100.0
        assert d["details"]["note"] == "passed"

    def test_error_report_serialization(self):
        res = CaseEvaluationResult(
            case_id="TEST_01",
            name="Test",
            category="tracking",
            mode="track_swap_pass",
            metric_name="test",
            unit="count",
            baseline_metric=1.0,
            mitigated_metric=0.0,
            delta=-1.0,
            improvement_percent=100.0,
            passed=True,
            details={},
        )
        report = ErrorReport(
            timestamp="2026-09-18T00:00:00Z",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            pass_rate_percent=100.0,
            results=[res],
        )
        d = report.to_dict()
        assert d["total_cases"] == 1
        assert d["pass_rate_percent"] == 100.0
        assert len(d["results"]) == 1


class TestFailureCaseEvaluator:
    def test_evaluate_track_swap(self):
        evaluator = FailureCaseEvaluator(fps=25.0)
        res = evaluator.evaluate_track_swap()
        assert res.case_id == "TS_PASS_01"
        assert res.category == "tracking"
        assert res.baseline_metric == 1.0
        assert res.mitigated_metric == 0.0
        assert res.delta == -1.0
        assert res.improvement_percent == 100.0
        assert res.passed is True

    def test_evaluate_superhuman_velocity(self):
        evaluator = FailureCaseEvaluator(fps=25.0)
        res = evaluator.evaluate_superhuman_velocity()
        assert res.case_id == "SH_VEL_02"
        assert res.category == "tracking"
        assert res.baseline_metric == 1.0
        assert res.mitigated_metric == 0.0
        assert res.delta == -1.0
        assert res.improvement_percent == 100.0
        assert res.passed is True

    def test_evaluate_possession_flicker(self):
        evaluator = FailureCaseEvaluator(fps=25.0)
        res = evaluator.evaluate_possession_flicker()
        assert res.case_id == "POSS_FLK_03"
        assert res.category == "analytics"
        assert res.baseline_metric == 2.0
        assert res.mitigated_metric == 0.0
        assert res.delta == -2.0
        assert res.improvement_percent == 100.0
        assert res.passed is True

    def test_evaluate_camera_cut(self):
        evaluator = FailureCaseEvaluator(fps=25.0)
        res = evaluator.evaluate_camera_cut()
        assert res.case_id == "CAM_CUT_04"
        assert res.category == "perspective_motion"
        assert res.baseline_metric > 0.0
        assert res.mitigated_metric == 0.0
        assert res.improvement_percent == 100.0
        assert res.passed is True

    def test_evaluate_ball_dropout(self):
        evaluator = FailureCaseEvaluator(fps=25.0)
        res = evaluator.evaluate_ball_dropout()
        assert res.case_id == "BALL_DROP_05"
        assert res.category == "detection"
        assert res.baseline_metric == 2.0
        assert res.mitigated_metric == 1.0
        assert res.delta == -1.0
        assert res.improvement_percent == 50.0
        assert res.passed is True

    def test_run_suite(self):
        evaluator = FailureCaseEvaluator(fps=25.0)
        report = evaluator.run_suite()
        assert report.total_cases == 5
        assert report.passed_cases == 5
        assert report.failed_cases == 0
        assert report.pass_rate_percent == 100.0
        assert len(report.results) == 5


class TestErrorAnalysisExporter:
    def test_export_json_and_csv(self, tmp_path: Path):
        evaluator = FailureCaseEvaluator(fps=25.0)
        report = evaluator.run_suite()

        json_file = tmp_path / "report.json"
        csv_file = tmp_path / "report.csv"

        out_json = ErrorAnalysisExporter.export_json(report, json_file)
        out_csv = ErrorAnalysisExporter.export_csv(report, csv_file)

        assert out_json.is_file()
        assert out_csv.is_file()

        with open(out_json, encoding="utf-8") as f:
            data = json.load(f)
            assert data["total_cases"] == 5
            assert data["pass_rate_percent"] == 100.0

        with open(out_csv, encoding="utf-8") as f:
            lines = f.readlines()
            # Header + 5 cases = 6 lines
            assert len(lines) == 6
            assert "TS_PASS_01" in lines[1]

    def test_format_terminal_table(self):
        evaluator = FailureCaseEvaluator(fps=25.0)
        report = evaluator.run_suite()
        table = ErrorAnalysisExporter.format_terminal_table(report)

        assert "FOOTBALL-CV RIGOROUS ERROR ANALYSIS" in table
        assert "TS_PASS_01" in table
        assert "SH_VEL_02" in table
        assert "POSS_FLK_03" in table
        assert "CAM_CUT_04" in table
        assert "BALL_DROP_05" in table
        assert "[PASS]" in table
