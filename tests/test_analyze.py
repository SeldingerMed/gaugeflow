import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments_v2.common.analyze import analyze


ANALYZE_PATH = Path(__file__).resolve().parents[1] / "experiments_v2" / "common" / "analyze.py"


def _paired_rows(delta: float = 0.08):
    rows = []
    for cluster_idx in range(8):
        base = 0.50 + (cluster_idx * 0.01)
        for seed in range(2):
            rows.extend(
                [
                    {
                        "arm": "baseline",
                        "seed": seed,
                        "cluster": f"case-{cluster_idx}",
                        "metric": base,
                    },
                    {
                        "arm": "gaugeflow",
                        "seed": seed,
                        "cluster": f"case-{cluster_idx}",
                        "metric": base + delta,
                        "leakage": 0.02,
                    },
                    {
                        "arm": "negctrl",
                        "seed": seed,
                        "cluster": f"case-{cluster_idx}",
                        "metric": base + 0.003,
                    },
                ]
            )
    return rows


def test_analyze_passes_when_paired_delta_and_controls_meet_gates():
    gates = {
        "min_delta_vs_baseline": 0.05,
        "require_delta_ci_excludes_0": True,
        "max_leakage": 0.035,
        "max_negctrl_abs_delta": 0.01,
    }

    result = analyze(_paired_rows(), gates, direction="higher", B=200, seed=123)

    assert result["PASS"] is True
    assert result["arms"]["baseline"]["n_clusters"] == 8
    assert result["arms"]["gaugeflow"]["n_clusters"] == 8
    assert result["gaugeflow_minus_baseline"]["n_paired_clusters"] == 8
    assert result["gaugeflow_minus_baseline"]["signed_delta"] == pytest.approx(0.08)
    assert result["gaugeflow_minus_baseline"]["ci95"][0] > 0
    assert result["gate_checks"] == {
        "delta_meets_min": True,
        "delta_ci_excludes_0": True,
        "leakage_under_cap": True,
        "negctrl_is_null": True,
    }


def test_analyze_lower_direction_treats_smaller_metrics_as_better():
    gates = {
        "min_delta_vs_baseline": 0.05,
        "require_delta_ci_excludes_0": True,
    }
    rows = _paired_rows(delta=-0.08)

    result = analyze(rows, gates, direction="lower", B=200, seed=123)

    assert result["PASS"] is True
    assert result["direction"] == "lower"
    assert result["gaugeflow_minus_baseline"]["signed_delta"] == pytest.approx(0.08)
    assert result["gaugeflow_minus_baseline"]["ci95"][0] > 0


def test_analyze_cli_writes_json_and_sets_success_exit_code(tmp_path):
    results_path = tmp_path / "results.jsonl"
    gates_path = tmp_path / "gates.json"
    output_path = tmp_path / "analysis.json"

    results_path.write_text("\n".join(json.dumps(row) for row in _paired_rows()) + "\n")
    gates_path.write_text(
        json.dumps(
            {
                "min_delta_vs_baseline": 0.05,
                "require_delta_ci_excludes_0": True,
                "max_leakage": 0.035,
                "max_negctrl_abs_delta": 0.01,
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ANALYZE_PATH),
            "--results",
            str(results_path),
            "--gates",
            str(gates_path),
            "--bootstrap",
            "200",
            "--out",
            str(output_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )

    stdout_result = json.loads(completed.stdout)
    file_result = json.loads(output_path.read_text())
    assert stdout_result == file_result
    assert file_result["PASS"] is True
    assert file_result["gate_checks"]["delta_ci_excludes_0"] is True
