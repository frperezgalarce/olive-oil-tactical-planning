
# ==========================================================
#  MODEL CLASS
# ==========================================================

# olive_model_service_level.py
# ----------------------------------------------------------
# Formulation #2 (Service Level) — adapted to Option A parsing:
#   - GP is OL[j][t] (loss curve) with length |T| or |T|+1 (t=0..T)
#   - sigma[j] is a placeholder (default 0.0 unless you set it)
# ----------------------------------------------------------


from .model_data import OliveHarvestDataFromJSON
import pulp as pl
import math
from typing import Optional, Dict, Any


class OliveHarvestModel_ServiceLevel:
    """
    Formulation #2:
      - Maximize oil yield.
      - Aggregate service-level constraint: sum_{j,t,k} x[j,t,k] >= mu_agg + z*sigma_agg
      - Initial inventory per lot: o[j,0] = mu[j] + z*sigma[j]
      - Inventory balance uses OL[j][t] (loss curve from GP Option A).
      - Budget is NOT constrained; required budget is computed as KPI after solve.

    Notes:
      - sigma_agg computed assuming independence: sqrt(sum_j sigma_j^2)
      - RequiredBudget uses certainty-equivalent denominator mu[j] for K1 terms
    """

    def __init__(self, data, sigma, mu):
        #data.validate()
        self.data = data
        self.mu = mu
        self.sigma = sigma
        self.J = data.J
        self.T = data.T
        self.T0 = data.T0

        self.K = data.K
        self.K1 = data.K1
        self.K2 = data.K2
        self.K3 = data.K3
        self.K4 = data.K4
        self.K1_comp = data.K1_comp

        self.model: Optional[pl.LpProblem] = None
        self.x = None
        self.o = None
        self.h = None  # only used if you want shaker productivity constraints

        self.req_budget_expr = None

    @staticmethod
    def z_value(alpha: float) -> float:
        """
        One-sided Normal quantile.
        Returns z such that P(Z <= z) = alpha, Z ~ N(0,1).
        Valid for 0 < alpha < 1.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")

        # Coefficients for the approximation
        a = [-3.969683028665376e+01,  2.209460984245205e+02,
            -2.759285104469687e+02,  1.383577518672690e+02,
            -3.066479806614716e+01,  2.506628277459239e+00]

        b = [-5.447609879822406e+01,  1.615858368580409e+02,
            -1.556989798598866e+02,  6.680131188771972e+01,
            -1.328068155288572e+01]

        c = [-7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00,  2.938163982698783e+00]

        d = [ 7.784695709041462e-03,  3.224671290700398e-01,
            2.445134137142996e+00,  3.754408661907416e+00]

        plow = 0.02425
        phigh = 1.0 - plow

        if alpha < plow:
            q = math.sqrt(-2.0 * math.log(alpha))
            return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

        if alpha > phigh:
            q = math.sqrt(-2.0 * math.log(1.0 - alpha))
            return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                    ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

        q = alpha - 0.5
        r = q*q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)

    def build(self):
        d = self.data
        z = self.z_value(d.alpha)

        # Aggregate mu and sigma (service level RHS)


        mu_agg = sum(self.mu[j-1] for j in self.J)

        sigma_agg = math.sqrt(sum((self.sigma[j-1] ** 2) for j in self.J))  # independence assumption

        self.model = pl.LpProblem("Olive_Harvest_Model_ServiceLevel", pl.LpMaximize)

        # --------------------
        # Decision variables
        # --------------------
        self.x = pl.LpVariable.dicts("x", (self.J, self.T, self.K), lowBound=0, cat="Continuous")
        self.o = pl.LpVariable.dicts("o", (self.J, self.T0), lowBound=0, cat="Continuous")

        # Optional: shaker hours (if you want to keep the linearization style)
        # Only create if you actually use K2 constraints with hours.
        # if getattr(d, "HMB", 0.0) > 0 and self.K2:
        #    self.h = pl.LpVariable.dicts("h", (self.J, self.T, self.K2), lowBound=0, cat="Continuous")

        # --------------------
        # Objective
        # --------------------
        self.model += pl.lpSum(
            self.x[j][t][k] * d.OC[j][t]
            for j in self.J for t in self.T for k in self.K
        ), "Oil_Maximization"

        # --------------------
        # Constraints
        # --------------------

        
        # Initial stock: o[j,0] = mu[j] + z*sigma[j]
        for j in self.J:
            self.model += (
                self.o[j][0] == self.mu[j-1] + z * self.sigma[j-1],
                f"Initial_stock_service_{j}",
            )
        
        
        # Inventory balance with loss curve OL[j][t] from GP (Option A)
        for j in self.J:
            for i, t in enumerate(self.T):
                prev = 0 if i == 0 else self.T[i - 1]
                self.model += (
                    self.o[j][prev] * (d.OL[j][t]) - pl.lpSum(self.x[j][t][k] for k in self.K)
                    == self.o[j][t],
                    f"InvBalance_{j}_{t}",
                )
        
        # Aggregate service-level constraint
        self.model += (
            pl.lpSum(self.x[j][t][k] for j in self.J for t in self.T for k in self.K)
            >= mu_agg + z * sigma_agg,
            "ServiceLevel_Aggregate",
        )

        # --------------------
        # Operational capacity constraints (optional/inactive if set to 0)
        # --------------------

        # Machine area (certainty equivalent mu[j])
        if getattr(d, "HMC", 0.0) > 0:
            self.model += (
                pl.lpSum(
                    self.x[j][t][k]
                    for j in self.J if d.mu[j] != 0
                    for t in self.T
                    for k in self.K1
                ) >= d.HMC,
                "Min_machine_area",
            )

        # Shaker constraints: choose ONE of the two patterns below.
        # A) If you want "hours" variable h and productivity PD: x <= PD * h, plus sum(h) >= HMB
        if getattr(d, "HMB", 0.0) > 0:
            self.model += (
                pl.lpSum(self.x[j][t][k] for j in self.J for t in self.T for k in self.K2) >= d.HMB,
                "Min_Shaker_tons",
            )

        # Minimum manual tonnage
        if getattr(d, "HMM", 0.0) > 0:
            self.model += (
                pl.lpSum(self.x[j][t][k] for j in self.J for t in self.T for k in self.K4) >= d.HMM,
                "Min_manual_tons",
            )

        # Resource capacity: sum_j x[j,t,k] <= C[k]   for all t,k
        # Requires: d.RC[k] defined for k in self.K
        for t in self.T:
            for k in self.K:
                for j in self.J:
                    self.model += (
                        self.x[j][t][k] <= d.PD[j][k],
                        f"ResCap_{k}_{t}_{j}",
                    )

        # Plant capacity
        for t in self.T:
            self.model += (
                pl.lpSum(self.x[j][t][k] for j in self.J for k in self.K)
                <= d.CP[t],
                f"PlantCap_{t}",
            )

        # --------------------
        # KPI: Required budget expression (not constrained)
        # --------------------
        self.req_budget_expr = (
            pl.lpSum(
                (self.x[j][t][k] * d.S[j] * d.UC[k] / d.mu[j])
                for j in self.J for t in self.T for k in self.K1
                if d.mu[j] != 0
            )
            + pl.lpSum(
                self.x[j][t][k] * d.UC[k]
                for j in self.J for t in self.T for k in self.K1_comp
            )
        )

        return self.model

    def solve(self, msg: bool = True, solver: Optional[pl.LpSolver] = None) -> str:
        if self.model is None:
            self.build()
        solver = solver or pl.PULP_CBC_CMD(msg=msg)
        status_code = self.model.solve(solver)
        return pl.LpStatus[status_code]

    def objective_value(self) -> Optional[float]:
        return pl.value(self.model.objective) if self.model is not None else None

    def required_budget_value(self) -> Optional[float]:
        return pl.value(self.req_budget_expr) if self.req_budget_expr is not None else None

    def get_x(self, tol: float = 1e-8) -> Dict[Any, float]:
        sol: Dict[Any, float] = {}
        for j in self.J:
            for t in self.T:
                for k in self.K:
                    v = self.x[j][t][k].varValue
                    if v is not None and v > tol:
                        sol[(j, t, k)] = v
        return sol

    def get_o(self, tol: float = 1e-8) -> Dict[Any, float]:
        sol: Dict[Any, float] = {}
        for j in self.J:
            for t in self.T0:
                v = self.o[j][t].varValue
                if v is not None and v > tol:
                    sol[(j, t)] = v
        return sol

class OliveHarvestModel:
    def __init__(self, data):
        data.validate()
        self.data = data

        self.T = data.T
        self.T0 = [0] + self.T
        self.J = data.J
        self.K = data.K

        self.K1 = data.K1
        self.K2 = data.K2
        self.K3 = data.K3
        self.K4 = data.K4

        # Complement of K1
        self.K1_comp = [k for k in self.K if k not in self.K1]

        # PuLP objects
        self.model = None
        self.x = None
        self.o = None
        self.h = None

    # ------------------------------------------------------
    # Build model
    # ------------------------------------------------------
    def build(self):
        d = self.data
        self.model = pl.LpProblem("Olive_Harvest_Model", pl.LpMaximize)

        # --------------------
        # Decision variables
        # --------------------
        self.x = pl.LpVariable.dicts("x", (self.J, self.T, self.K),
                                     lowBound=0, cat="Continuous")
        self.o = pl.LpVariable.dicts("o", (self.J, self.T0),
                                     lowBound=0, cat="Continuous")
        self.h = pl.LpVariable.dicts("h", (self.J, self.T, self.K2),
                                     lowBound=0, cat="Continuous")

        # --------------------
        # Objective
        # --------------------
        self.model += pl.lpSum(
            self.x[j][t][k] * d.OC[j][t]
            for j in self.J for t in self.T for k in self.K
        ), "Oil_Maximization"

        # --------------------------------------------------
        # Constraints
        # --------------------------------------------------

        # (10) Budget constraint
        self.model += (
            pl.lpSum(
                self.x[j][t][k] * d.S[j] * d.UC[k] / d.OSn[j]
                for j in self.J for t in self.T for k in self.K1
            )
            + pl.lpSum(
                self.x[j][t][k] * d.UC[k]
                for j in self.J for t in self.T for k in self.K1_comp
            )
            <= d.BH,
            "Budget"
        )

        # (19) Initial stock
        for j in self.J:
            self.model += (self.o[j][0] == d.OSn[j], f"Initial_stock_{j}")

        # (20) Inventory balance
        for j in self.J:
            for i, t in enumerate(self.T):
                prev = 0 if i == 0 else self.T[i - 1]
                self.model += (
                    self.o[j][prev] * (1 - d.OL[j][t])
                    - pl.lpSum(self.x[j][t][k] for k in self.K)
                    == self.o[j][t],
                    f"InvBalance_{j}_{t}"
                )

        # (24) Minimum machine-harvested area
        self.model += (
            pl.lpSum(
                self.x[j][t][k] * d.S[j] / d.OSn[j]
                for j in self.J if d.OSn[j] != 0
                for t in self.T
                for k in self.K1
            ) >= d.HMC,
            "Min_machine_area"
        )

        # (25-LIN-1) Productivity for Shaker A
        for j in self.J:
            for t in self.T:
                for k in self.K2:
                    self.model += (
                        self.x[j][t][k] <= d.PD[j][k] * self.h[j][t][k],
                        f"ShakerA_prod_{j}_{t}_{k}"
                    )

        # (25-LIN-2) Minimum Shaker A hours
        self.model += (
            pl.lpSum(self.h[j][t][k] for j in self.J for t in self.T for k in self.K2)
            >= d.HMB,
            "Min_ShakerA_hours"
        )

        # (26) Minimum manual harvest
        self.model += (
            pl.lpSum(
                self.x[j][t][k]
                for j in self.J for t in self.T for k in self.K4
            ) >= d.HMM,
            "Min_manual"
        )

        # (27) Fraction harvested per lot
        for j in self.J:
            self.model += (
                pl.lpSum(
                    self.x[j][t][k] for t in self.T for k in self.K
                ) >= d.F[j] * d.OSn[j],
                f"Fraction_{j}"
            )

        # (28) Plant capacity
        for t in self.T:
            self.model += (
                pl.lpSum(
                    self.x[j][t][k] for j in self.J for k in self.K
                )
                + d.OE[t]
                <= d.CP[t],
                f"PlantCap_{t}"
            )

        return self.model

    # ------------------------------------------------------
    # Solve
    # ------------------------------------------------------
    def solve(self, msg=True, solver=None):
        if self.model is None:
            self.build()

        solver = solver or pl.PULP_CBC_CMD(msg=msg)
        result = self.model.solve(solver)
        return pl.LpStatus[result]

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------
    def objective_value(self):
        return pl.value(self.model.objective)

    def get_x(self, tol=1e-6):
        sol = {}
        for j in self.J:
            for t in self.T:
                for k in self.K:
                    v = self.x[j][t][k].varValue
                    if v and v > tol:
                        sol[(j, t, k)] = v
        return sol

    def get_o(self, tol=1e-6):
        sol = {}
        for j in self.J:
            for t in self.T0:
                v = self.o[j][t].varValue
                if v and v > tol:
                    sol[(j, t)] = v
        return sol
