"""The solution procedure of MultiEchelonPipeline.run_instance end to end on a small budget."""
import contextlib
import io
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
pytest.importorskip("highspy")
import MultiEchelonPipeline as P  # noqa: E402
import Repair_heuristcs as RH  # noqa: E402


def test_run_instance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(P, "ITS_ITERATIONS", 2)
    monkeypatch.setattr(P, "ITS_STAGNATION", 2)
    monkeypatch.setattr(P, "ALNS_ITERATIONS", 15)
    monkeypatch.setattr(P, "ALNS_TIME_LIMIT", None)
    monkeypatch.setattr(P, "FE_CANDIDATES", 3)
    reports = {k: P.create_report_file(k) for k in ("main", "initial", "multi")}
    with contextlib.redirect_stdout(io.StringIO()):
        problem, final, row = P.run_instance("c208.txt", reports)
    assert final is not None and final.is_feasible
    served = sorted(n.index for r in final.routes_sevs for n in r.sequence)
    assert served == sorted(n.index for n in problem.customers)
    # each step does not lose ground: initial >= local search >= ITS (two-way cost)
    assert row["initial_two_way"] >= row["ls_two_way"] >= row["its_two_way"]
    assert 1 <= row["its_rounds"] <= 2
    assert len(problem.checked_solutions) <= 3
    assert row["objective"] == pytest.approx(final.objective, abs=0.01)
    # report files
    text = (tmp_path / f"run_c208_{P.date.today()}.txt").read_text()
    for step in ("step 1:", "step 1+:", "step 2:", "step 3:", "final two-echelon solution"):
        assert step in text
    lines = (tmp_path / f"summary_{P.date.today()}.csv").read_text().splitlines()
    assert lines[0].split(",") == P.SUMMARY_COLUMNS and lines[1].startswith("c208,")
