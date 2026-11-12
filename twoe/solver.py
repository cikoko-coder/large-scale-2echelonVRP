"""
Thin MILP adapter so the two-echelon models do not depend on a single solver.

Backends (chosen automatically, or forced with ``MILP(backend=...)``):

* ``"highs"``  – HiGHS through the ``highspy`` package (default, open source, MIT)
* ``"scipy"``  – HiGHS through ``scipy.optimize.milp`` (fallback when ``highspy`` is
                 not installed; no warm start, no objective cutoff)
* ``"gurobi"`` – optional, for cross-checking results with a commercial solver

The modelling surface is deliberately small and mirrors the handful of Gurobi
features the original code used: binary/integer/continuous variables, linear
constraints, a linear objective, time limit, relative MIP gap, an objective
cutoff, warm starts, and reading back status, objective, gap and values.

Example
-------
>>> m = MILP("test")
>>> x = m.add_var(vtype="B", name="x")
>>> y = m.add_var(lb=0, ub=10, name="y")
>>> m.add_constr(m.expr([(x, 2.0), (y, 1.0)]) <= 7)
>>> m.set_objective(m.expr([(x, -3.0), (y, -1.0)]))
>>> m.optimize()
>>> m.status, m.obj_value, m.value(x), m.value(y)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

INF = math.inf

# --------------------------------------------------------------------------- #
# Linear expressions
# --------------------------------------------------------------------------- #


class Var:
    """A decision variable; only an index into the model plus metadata."""

    __slots__ = ("idx", "name", "lb", "ub", "vtype", "start")

    def __init__(self, idx: int, name: str, lb: float, ub: float, vtype: str):
        self.idx = idx
        self.name = name
        self.lb = lb
        self.ub = ub
        self.vtype = vtype  # "C", "B", "I"
        self.start: Optional[float] = None

    # arithmetic sugar -> LinExpr
    def __mul__(self, c):
        return LinExpr({self.idx: float(c)})

    __rmul__ = __mul__

    def __add__(self, other):
        return LinExpr({self.idx: 1.0}) + other

    __radd__ = __add__

    def __sub__(self, other):
        return LinExpr({self.idx: 1.0}) - other

    def __rsub__(self, other):
        return (-1.0) * LinExpr({self.idx: 1.0}) + other

    def __neg__(self):
        return LinExpr({self.idx: -1.0})

    def __le__(self, rhs):
        return LinExpr({self.idx: 1.0}) <= rhs

    def __ge__(self, rhs):
        return LinExpr({self.idx: 1.0}) >= rhs

    def __eq__(self, rhs):  # type: ignore[override]
        return LinExpr({self.idx: 1.0}) == rhs

    __hash__ = object.__hash__

    def __repr__(self):
        return f"Var({self.name})"


class LinExpr:
    """Sparse linear expression: sum(coef * var) + const."""

    __slots__ = ("coefs", "const")

    def __init__(self, coefs: Optional[Dict[int, float]] = None, const: float = 0.0):
        self.coefs: Dict[int, float] = dict(coefs) if coefs else {}
        self.const = float(const)

    @staticmethod
    def _as_expr(other) -> "LinExpr":
        if isinstance(other, LinExpr):
            return other
        if isinstance(other, Var):
            return LinExpr({other.idx: 1.0})
        return LinExpr({}, float(other))

    def copy(self) -> "LinExpr":
        return LinExpr(self.coefs, self.const)

    def add_term(self, var: Var, coef: float) -> "LinExpr":
        self.coefs[var.idx] = self.coefs.get(var.idx, 0.0) + float(coef)
        return self

    def __add__(self, other):
        o = self._as_expr(other)
        r = self.copy()
        for k, v in o.coefs.items():
            r.coefs[k] = r.coefs.get(k, 0.0) + v
        r.const += o.const
        return r

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-1.0) * self._as_expr(other)

    def __rsub__(self, other):
        return self._as_expr(other) - self

    def __mul__(self, c):
        c = float(c)
        return LinExpr({k: v * c for k, v in self.coefs.items()}, self.const * c)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    # comparisons build constraints
    def __le__(self, rhs):
        return Constraint(self - self._as_expr(rhs), "<=")

    def __ge__(self, rhs):
        return Constraint(self - self._as_expr(rhs), ">=")

    def __eq__(self, rhs):  # type: ignore[override]
        return Constraint(self - self._as_expr(rhs), "==")

    __hash__ = object.__hash__

    def __repr__(self):
        return f"LinExpr({self.coefs}, {self.const})"


@dataclass
class Constraint:
    """expr (sense) 0, stored as  lhs - rhs  so the bound is -const."""

    expr: LinExpr
    sense: str  # "<=", ">=", "=="
    name: str = ""

    def bounds(self) -> Tuple[float, float]:
        rhs = -self.expr.const
        if self.sense == "<=":
            return (-INF, rhs)
        if self.sense == ">=":
            return (rhs, INF)
        return (rhs, rhs)


def quicksum(terms: Iterable) -> LinExpr:
    """Sum of Vars / LinExprs / numbers (same role as gurobipy.quicksum)."""
    r = LinExpr()
    for t in terms:
        if isinstance(t, Var):
            r.coefs[t.idx] = r.coefs.get(t.idx, 0.0) + 1.0
        elif isinstance(t, LinExpr):
            for k, v in t.coefs.items():
                r.coefs[k] = r.coefs.get(k, 0.0) + v
            r.const += t.const
        else:
            r.const += float(t)
    return r


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

OPTIMAL = "OPTIMAL"
FEASIBLE = "FEASIBLE"  # stopped (time limit / gap) with an incumbent
INFEASIBLE = "INFEASIBLE"
NO_SOLUTION = "NO_SOLUTION"  # stopped without an incumbent, feasibility unknown
CUTOFF = "CUTOFF"  # proven worse than the cutoff
ERROR = "ERROR"


def _detect_backend() -> str:
    try:
        import highspy  # noqa: F401

        return "highs"
    except Exception:
        pass
    try:
        import scipy.optimize  # noqa: F401

        return "scipy"
    except Exception:
        pass
    try:
        import gurobipy  # noqa: F401

        return "gurobi"
    except Exception:
        pass
    raise ImportError("No MILP backend available: install highspy (pip install highspy).")


@dataclass
class SolveStats:
    status: str = NO_SOLUTION
    obj_value: Optional[float] = None
    obj_bound: Optional[float] = None
    mip_gap: Optional[float] = None
    runtime: float = 0.0
    n_vars: int = 0
    n_constrs: int = 0
    backend: str = ""
    extra: dict = field(default_factory=dict)


class MILP:
    def __init__(self, name: str = "model", backend: Optional[str] = None):
        self.name = name
        self.backend = backend or _detect_backend()
        self.vars: List[Var] = []
        self.constrs: List[Constraint] = []
        self.objective = LinExpr()
        self.minimize = True
        # parameters
        self.time_limit: float = INF
        self.mip_gap: float = 1e-4
        self.cutoff: Optional[float] = None
        self.verbose: bool = False
        self.threads: Optional[int] = None
        # results
        self.stats = SolveStats(backend=self.backend)
        self._x: Optional[List[float]] = None

    # ---- building -------------------------------------------------------- #
    def add_var(self, lb: float = 0.0, ub: float = INF, vtype: str = "C", name: str = "") -> Var:
        if vtype == "B":
            lb, ub = max(0.0, lb), min(1.0, ub)
        v = Var(len(self.vars), name or f"v{len(self.vars)}", float(lb), float(ub), vtype)
        self.vars.append(v)
        return v

    def add_constr(self, c: Constraint, name: str = "") -> Constraint:
        if not isinstance(c, Constraint):
            raise TypeError("add_constr expects an expression compared with <=, >= or ==")
        if name:
            c.name = name
        self.constrs.append(c)
        return c

    def expr(self, terms: Sequence[Tuple[Var, float]] = (), const: float = 0.0) -> LinExpr:
        e = LinExpr({}, const)
        for v, c in terms:
            e.add_term(v, c)
        return e

    def set_objective(self, e, minimize: bool = True):
        self.objective = LinExpr._as_expr(e).copy()
        self.minimize = minimize

    def set_start(self, var: Var, value: float):
        var.start = float(value)

    # ---- results --------------------------------------------------------- #
    @property
    def status(self) -> str:
        return self.stats.status

    @property
    def has_solution(self) -> bool:
        return self._x is not None

    @property
    def obj_value(self) -> Optional[float]:
        return self.stats.obj_value

    @property
    def gap(self) -> Optional[float]:
        return self.stats.mip_gap

    def value(self, v: Var) -> float:
        if self._x is None:
            raise RuntimeError("no solution available")
        return self._x[v.idx]

    def expr_value(self, e: LinExpr) -> float:
        return sum(c * self._x[k] for k, c in e.coefs.items()) + e.const

    # ---- solving --------------------------------------------------------- #
    def optimize(self) -> str:
        t0 = time.time()
        self.stats = SolveStats(backend=self.backend, n_vars=len(self.vars), n_constrs=len(self.constrs))
        self._x = None
        if self.backend == "highs":
            self._solve_highs()
        elif self.backend == "scipy":
            self._solve_scipy()
        elif self.backend == "gurobi":
            self._solve_gurobi()
        else:
            raise ValueError(f"unknown backend {self.backend}")
        self.stats.runtime = time.time() - t0
        if self._x is not None and self.stats.obj_value is None:
            self.stats.obj_value = self.expr_value(self.objective)
        return self.stats.status

    # .................................................................... #
    def _matrix(self):
        """Return (c, A_rows, A_cols, A_vals, row_lb, row_ub, col_lb, col_ub, integrality)."""
        n = len(self.vars)
        c = [0.0] * n
        for k, v in self.objective.coefs.items():
            c[k] += v
        if not self.minimize:
            c = [-ci for ci in c]
        rows, cols, vals, rlb, rub = [], [], [], [], []
        for r, con in enumerate(self.constrs):
            lo, hi = con.bounds()
            rlb.append(lo)
            rub.append(hi)
            for k, v in con.expr.coefs.items():
                if v != 0.0:
                    rows.append(r)
                    cols.append(k)
                    vals.append(v)
        clb = [v.lb for v in self.vars]
        cub = [v.ub for v in self.vars]
        integ = [0 if v.vtype == "C" else 1 for v in self.vars]
        return c, rows, cols, vals, rlb, rub, clb, cub, integ

    # .................................................................... #
    def _solve_highs(self):
        import highspy
        import numpy as np

        c, rows, cols, vals, rlb, rub, clb, cub, integ = self._matrix()
        n, m = len(self.vars), len(self.constrs)
        h = highspy.Highs()
        h.setOptionValue("output_flag", bool(self.verbose))
        if self.time_limit != INF:
            h.setOptionValue("time_limit", float(self.time_limit))
        h.setOptionValue("mip_rel_gap", float(self.mip_gap))
        if self.cutoff is not None:
            h.setOptionValue("objective_bound", float(self.cutoff))
        if self.threads:
            h.setOptionValue("threads", int(self.threads))

        lp = highspy.HighsLp()
        lp.num_col_ = n
        lp.num_row_ = m
        lp.col_cost_ = np.array(c, dtype=float)
        lp.col_lower_ = np.array([(-highspy.kHighsInf if x == -INF else x) for x in clb], dtype=float)
        lp.col_upper_ = np.array([(highspy.kHighsInf if x == INF else x) for x in cub], dtype=float)
        lp.row_lower_ = np.array([(-highspy.kHighsInf if x == -INF else x) for x in rlb], dtype=float)
        lp.row_upper_ = np.array([(highspy.kHighsInf if x == INF else x) for x in rub], dtype=float)
        # column-wise CSC
        order = sorted(range(len(vals)), key=lambda i: (cols[i], rows[i]))
        starts = [0] * (n + 1)
        for i in order:
            starts[cols[i] + 1] += 1
        for j in range(n):
            starts[j + 1] += starts[j]
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = np.array(starts, dtype=np.int32)
        lp.a_matrix_.index_ = np.array([rows[i] for i in order], dtype=np.int32)
        lp.a_matrix_.value_ = np.array([vals[i] for i in order], dtype=float)
        lp.integrality_ = [highspy.HighsVarType.kInteger if t else highspy.HighsVarType.kContinuous for t in integ]
        h.passModel(lp)

        starts_given = [v for v in self.vars if v.start is not None]
        if starts_given:
            sol = highspy.HighsSolution()
            sol.col_value = [v.start if v.start is not None else 0.0 for v in self.vars]
            sol.value_valid = True
            h.setSolution(sol)

        h.run()
        ms = h.getModelStatus()
        info = h.getInfo()
        name = h.modelStatusToString(ms)
        has_incumbent = info.primal_solution_status == 2  # kSolutionStatusFeasible
        if has_incumbent:
            self._x = list(h.getSolution().col_value)
            self.stats.obj_value = float(info.objective_function_value) * (1 if self.minimize else -1)
            try:
                self.stats.mip_gap = float(info.mip_gap)
                self.stats.obj_bound = float(info.mip_dual_bound)
            except Exception:
                pass
        if name == "Optimal":
            self.stats.status = OPTIMAL
        elif name == "Infeasible":
            self.stats.status = INFEASIBLE
        elif name in ("Objective bound", "Objective target"):
            self.stats.status = CUTOFF if not has_incumbent else FEASIBLE
        elif has_incumbent:
            self.stats.status = FEASIBLE
        else:
            self.stats.status = NO_SOLUTION
        self.stats.extra["highs_status"] = name

    # .................................................................... #
    def _solve_scipy(self):
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix

        c, rows, cols, vals, rlb, rub, clb, cub, integ = self._matrix()
        n, m = len(self.vars), len(self.constrs)
        A = csr_matrix((vals, (rows, cols)), shape=(m, n)) if m else None
        constraints = [LinearConstraint(A, np.array(rlb), np.array(rub))] if m else None
        options = {"disp": bool(self.verbose), "mip_rel_gap": float(self.mip_gap)}
        if self.time_limit != INF:
            options["time_limit"] = float(self.time_limit)
        res = milp(
            c=np.array(c),
            constraints=constraints,
            integrality=np.array(integ),
            bounds=Bounds(np.array(clb), np.array(cub)),
            options=options,
        )
        self.stats.extra["scipy_status"] = int(res.status)
        self.stats.extra["scipy_message"] = str(res.message)
        if res.x is not None:
            self._x = [float(v) for v in res.x]
            self.stats.obj_value = float(res.fun) * (1 if self.minimize else -1)
            self.stats.mip_gap = float(getattr(res, "mip_gap", float("nan")))
            self.stats.obj_bound = getattr(res, "mip_dual_bound", None)
        # status codes: 0 optimal, 1 iteration/time limit, 2 infeasible, 3 unbounded, 4 other
        if res.status == 0:
            self.stats.status = OPTIMAL
        elif res.status == 2:
            self.stats.status = INFEASIBLE
        elif self._x is not None:
            self.stats.status = FEASIBLE
        else:
            self.stats.status = NO_SOLUTION
        # scipy has no cutoff; emulate the CUTOFF outcome for callers
        if self.cutoff is not None and self._x is not None and self.minimize and self.stats.obj_value > self.cutoff:
            self.stats.status = CUTOFF

    # .................................................................... #
    def _solve_gurobi(self):
        import gurobipy as gp
        from gurobipy import GRB

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1 if self.verbose else 0)
        env.start()
        g = gp.Model(self.name, env=env)
        gv = []
        for v in self.vars:
            vt = {"C": GRB.CONTINUOUS, "B": GRB.BINARY, "I": GRB.INTEGER}[v.vtype]
            lb = -GRB.INFINITY if v.lb == -INF else v.lb
            ub = GRB.INFINITY if v.ub == INF else v.ub
            x = g.addVar(lb=lb, ub=ub, vtype=vt, name=v.name)
            if v.start is not None:
                x.Start = v.start
            gv.append(x)
        for con in self.constrs:
            e = gp.LinExpr([(c, gv[k]) for k, c in con.expr.coefs.items()])
            rhs = -con.expr.const
            if con.sense == "<=":
                g.addConstr(e <= rhs, name=con.name)
            elif con.sense == ">=":
                g.addConstr(e >= rhs, name=con.name)
            else:
                g.addConstr(e == rhs, name=con.name)
        obj = gp.LinExpr([(c, gv[k]) for k, c in self.objective.coefs.items()]) + self.objective.const
        g.setObjective(obj, GRB.MINIMIZE if self.minimize else GRB.MAXIMIZE)
        if self.time_limit != INF:
            g.Params.TimeLimit = self.time_limit
        g.Params.MIPGap = self.mip_gap
        if self.cutoff is not None:
            g.Params.Cutoff = self.cutoff
        if self.threads:
            g.Params.Threads = self.threads
        g.optimize()
        if g.SolCount > 0:
            self._x = [x.X for x in gv]
            self.stats.obj_value = g.ObjVal
            try:
                self.stats.mip_gap = g.MIPGap
                self.stats.obj_bound = g.ObjBound
            except Exception:
                pass
        st = g.Status
        if st == GRB.OPTIMAL:
            self.stats.status = OPTIMAL
        elif st == GRB.INFEASIBLE:
            self.stats.status = INFEASIBLE
        elif st == GRB.CUTOFF:
            self.stats.status = CUTOFF
        elif self._x is not None:
            self.stats.status = FEASIBLE
        else:
            self.stats.status = NO_SOLUTION
        self.stats.extra["gurobi_status"] = st
        g.dispose()
        env.dispose()
