import json

import pytest

from aegisframe.calibrate import calibrate, calibrate_file
from aegisframe.cli import main
from aegisframe.config import CouncilConfig
from aegisframe.council import build_council

HARMFUL = [0.9] * 95 + [0.2] * 5
BENIGN = [0.05] * 90 + [0.3] * 9 + [0.95]


def test_crc_picks_the_largest_threshold_meeting_the_target():
    assert calibrate(HARMFUL, BENIGN, alpha=0.10, method="crc")["review"] == pytest.approx(0.9)
    assert calibrate(HARMFUL, BENIGN, alpha=0.05, method="crc")["review"] == pytest.approx(0.2)


def test_high_probability_bound_is_more_conservative():
    crc = calibrate(HARMFUL, BENIGN, alpha=0.10, method="crc")
    ucb = calibrate(HARMFUL, BENIGN, alpha=0.10, delta=0.05, method="ucb")
    assert ucb["review"] <= crc["review"]


def test_infeasible_target_falls_back_to_reviewing_everything():
    r = calibrate([0.9] * 10, BENIGN, alpha=0.001, method="crc")
    assert r["feasible"] is False and r["review"] == 0.0


def test_block_threshold_is_never_below_review():
    r = calibrate(HARMFUL, BENIGN, alpha=0.10)
    assert r["block"] >= r["review"]
    assert r["empirical"]["harmful_allow"] <= 0.10


def write_results(path, harmful, benign):
    with path.open("w", encoding="utf-8") as f:
        for p in harmful:
            f.write(json.dumps({"expected": "halt", "posterior": p}) + "\n")
        for p in benign:
            f.write(json.dumps({"expected": "allow", "posterior": p}) + "\n")
        f.write(json.dumps({"expected": "unlock", "posterior": 0.9}) + "\n")  # ignored
        f.write(json.dumps({"expected": "halt", "posterior": None}) + "\n")  # no council score: ignored


def test_calibrate_file_writes_a_file_the_council_can_load(tmp_path):
    results = tmp_path / "r.jsonl"
    write_results(results, HARMFUL, BENIGN)
    out = tmp_path / "calibration.json"
    r = calibrate_file(results, alpha=0.10, out=out)
    assert r["n_harmful"] == 100 and r["n_benign"] == 100
    council = build_council(CouncilConfig(calibration_file=str(out)), providers={})
    assert council.thresholds["review"] == pytest.approx(r["review"])


def test_cli_calibrate(tmp_path, capsys):
    results = tmp_path / "r.jsonl"
    write_results(results, HARMFUL, BENIGN)
    assert main(["calibrate", "--results", str(results), "--alpha", "0.1", "--out", str(tmp_path / "c.json")]) == 0
    assert json.loads(capsys.readouterr().out)["alpha"] == 0.1


# ---- bench integration -----------------------------------------------------------------------


def test_runner_records_council_posterior_and_agents_used():
    from reframebench.cases import build_cases
    from reframebench.run import AegisSystem, run_bench

    cases = [c for c in build_cases() if c.scenario == "upi-amount"][:3]
    results = run_bench(AegisSystem(), cases)
    assert all(r.posterior is not None and r.agents_used >= 1 for r in results)


def test_bench_accepts_a_config_for_ablations(tmp_path, capsys):
    from reframebench.run import main as bench_main

    cfg = tmp_path / "ablation.yaml"
    cfg.write_text("council:\n  enabled: false\n", encoding="utf-8")
    bench_main(["--systems", "aegis", "--config", str(cfg), "--limit", "8", "--out", str(tmp_path / "out")])
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert "aegisframe" in summary
    lines = (tmp_path / "out" / "aegisframe.jsonl").read_text(encoding="utf-8").splitlines()
    assert all(json.loads(x)["posterior"] is None for x in lines)


def test_block_threshold_with_identical_benign_scores():
    r = calibrate([0.9] * 100, [0.03] * 60, alpha=0.05)
    assert 0.85 <= r["block"] < 0.999
    assert r["empirical"]["benign_block"] == 0.0


def test_block_threshold_sits_just_above_the_benign_tail():
    benign = [0.05] * 98 + [0.9, 0.95]
    r = calibrate([0.99] * 100, benign, alpha=0.05, benign_block=0.01)
    assert r["empirical"]["benign_block"] <= 0.01
    assert r["block"] > 0.9
