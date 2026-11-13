"""The ITS of MultiEchelonPipeline.run_instance end to end on a small budget."""
import contextlib
import io
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
pytest.importorskip("highspy")
import MultiEchelonPipeline as P  # noqa: E402


def test_run_instance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, value in (("N_INIT", 2), ("N_ITS", 2), ("R_LS", 2), ("N_LS", 5), ("N_ALNS", 15), ("FE_CANDIDATES", 2)):
        monkeypatch.setattr(P, name, value)
    reports = {k: P.create_report_file(k) for k in ("main", "initial", "multi")}
    with contextlib.redirect_stdout(io.StringIO()):
        problem, final, row = P.run_instance("c208.txt", reports)
    assert final is not None and final.is_feasible
    served = sorted(n.index for r in final.routes_sevs for n in r.sequence)
    assert served == sorted(n.index for n in problem.customers)
    # R* never gets worse: fleet first, then two-way cost
    assert (row["init_sevs"], row["init_two_way"]) >= (row["init_ls_sevs"], row["init_ls_two_way"]) \
        >= (row["its_sevs"], row["its_two_way"])
    assert [log["its_iter"] for log in problem.its_log] == [1, 2]
    assert "init" in problem.its_log[0] and "init" not in problem.its_log[1]
    assert all(log["alns_report"]["iterations"] == 15 for log in problem.its_log)
    # FE only on min-fleet SE solutions
    assert 1 <= row["fe_candidates"] <= 2
    assert all(len(s) == row["its_sevs"] for s in problem.checked_solutions)
    assert row["objective"] == pytest.approx(final.objective, abs=0.01)
    text = (tmp_path / f"run_c208_{P.date.today()}.txt").read_text()
    for part in ("ITS iteration 1", "ITS iteration 2", "init (2 starts)", "ALNS 15 iterations",
                 "FE optimization", "final two-echelon solution"):
        assert part in text
    lines = (tmp_path / f"summary_{P.date.today()}.csv").read_text().splitlines()
    assert lines[0].split(",") == P.SUMMARY_COLUMNS and lines[1].startswith("c208,")
