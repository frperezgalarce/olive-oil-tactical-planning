# olive_model.py
# ----------------------------------------------------------
# Linear model for olive harvest optimization using PuLP
# Converted from the notebook/script demo_lp.py
# ----------------------------------------------------------


import pulp as pl


# ==========================================================
#  MODEL CLASS
# ==========================================================

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
