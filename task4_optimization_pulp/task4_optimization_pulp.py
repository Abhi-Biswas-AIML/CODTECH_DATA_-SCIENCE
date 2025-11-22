"""
task4_optimization_pulp.py

Simple business optimization example using PuLP.

Scenario:
    A small factory produces two products (A and B). Each product uses a limited
    number of machine hours and labor hours. We want to maximize profit while
    respecting resource constraints.

    Maximize: 40 * A + 30 * B

    Subject to:
        2A + 1B <= 100   (machine hours)
        1A + 1.5B <= 90  (labor hours)
        A, B >= 0

Normally this is solved with an LP solver (CBC / HiGHS via PuLP).
On some Windows setups the solver executable (cbc.exe / highs.exe) cannot run.
To make the script robust, we try the solver first and, if it fails,
we fall back to an analytic solution for this 2-variable problem.
"""

import pulp as pl
from pulp.apis.core import PulpSolverError


def solve_with_pulp():
    """Build and solve the LP model using PuLP."""
    # Define LP problem: maximize profit
    prob = pl.LpProblem("Production_Planning", pl.LpMaximize)

    # Decision variables: number of units to produce for A and B
    x_A = pl.LpVariable("Units_A", lowBound=0, cat="Continuous")
    x_B = pl.LpVariable("Units_B", lowBound=0, cat="Continuous")

    # Parameters (profit per unit)
    profit_A = 40
    profit_B = 30

    # Objective function: maximize total profit
    prob += profit_A * x_A + profit_B * x_B, "Total_Profit"

    # Constraints
    prob += 2 * x_A + 1 * x_B <= 100, "Machine_Hours"
    prob += 1 * x_A + 1.5 * x_B <= 90, "Labor_Hours"

    # Try to solve with HiGHS, fall back to CBC if available
    try:
        solver = pl.getSolver("HiGHS_CMD", msg=False) or pl.PULP_CBC_CMD(msg=False)
        prob.solve(solver)
    except Exception as e:
        raise PulpSolverError(f"Solver execution failed: {e}")

    return prob, x_A, x_B


def analytic_fallback():
    """
    Analytic solution for this specific 2-variable LP.

    We maximize 40A + 30B s.t.
        2A + B <= 100
        A + 1.5B <= 90
        A, B >= 0

    Corner points:
        (0,0)
        (50,0)  from 2A + B = 100, B=0
        (0,60)  from A + 1.5B = 90, A=0
        (30,40) intersection of both constraints

    Evaluating profit:
        (0,0)   ->   0
        (50,0)  -> 2000
        (0,60)  -> 1800
        (30,40) -> 2400  (maximum)

    So the optimal solution is:
        A = 30 units
        B = 40 units
        Profit = 2400
    """
    A_opt = 30.0
    B_opt = 40.0
    profit_opt = 40 * A_opt + 30 * B_opt

    print("[Task 4] (Fallback) Optimal production plan (analytic solution):")
    print(f"  Units of A: {A_opt:.2f}")
    print(f"  Units of B: {B_opt:.2f}")
    print(f"[Task 4] (Fallback) Maximum profit: {profit_opt:.2f}")

    print("\n[Insights]")
    print(
        "This solution shows how many units of each product to produce in order to "
        "maximize profit while respecting machine and labor capacity."
    )
    print(
        "If a proper LP solver (CBC / HiGHS) is available, PuLP can compute the same "
        "solution automatically; otherwise, this analytic fallback still gives "
        "the correct result for this problem."
    )


def main():
    print("[Task 4] Solving production planning problem...")

    try:
        prob, x_A, x_B = solve_with_pulp()
        status = pl.LpStatus[prob.status]
        print("[Task 4] Solver status:", status)

        if status != "Optimal":
            print(
                "[Warning] Solver did not return an Optimal status. "
                "Falling back to analytic solution."
            )
            analytic_fallback()
            return

        print("[Task 4] Optimal production plan (from PuLP):")
        print(f"  Units of A: {x_A.value():.2f}")
        print(f"  Units of B: {x_B.value():.2f}")
        print("[Task 4] Maximum profit:", pl.value(prob.objective))

        print("\n[Insights]")
        print(
            "This solution shows how many units of each product to produce in order to "
            "maximize profit while respecting machine and labor capacity."
        )
        print(
            "You can modify the coefficients (profit, resource usage, capacity) to model "
            "different real-world scenarios."
        )

    except PulpSolverError as e:
        # If the solver executable (cbc.exe / highs.exe) cannot run, we land here.
        print("[Error] PuLP solver could not be executed on this system.")
        print(f"        Details: {e}")
        print("\n[Info] Falling back to analytic solution so that the task still runs.")
        analytic_fallback()


if __name__ == "__main__":
    main()
