"""task4_optimization_pulp.py

Simple business optimization example using PuLP.

Scenario:
    A small factory produces two products (A and B). Each product uses a limited
    number of machine hours and labor hours. We want to maximize profit while
    respecting resource constraints.

Deliverable:
    - Problem setup
    - Solution using linear programming
    - Printed insights
"""

import pulp as pl


def main():
    # Define LP problem: maximize profit
    prob = pl.LpProblem("Production_Planning", pl.LpMaximize)

    # Decision variables: number of units to produce for A and B
    x_A = pl.LpVariable("Units_A", lowBound=0, cat="Continuous")
    x_B = pl.LpVariable("Units_B", lowBound=0, cat="Continuous")

    # Parameters
    profit_A = 40  # profit per unit of A
    profit_B = 30  # profit per unit of B

    # Resource constraints:
    # Machine hours
    #   - Product A uses 2 hours per unit
    #   - Product B uses 1 hour per unit
    #   - Total machine hours available = 100
    # Labor hours
    #   - Product A uses 1 hour per unit
    #   - Product B uses 1.5 hours per unit
    #   - Total labor hours available = 90

    # Objective function: maximize total profit
    prob += profit_A * x_A + profit_B * x_B, "Total_Profit"

    # Constraints
    prob += 2 * x_A + 1 * x_B <= 100, "Machine_Hours"
    prob += 1 * x_A + 1.5 * x_B <= 90, "Labor_Hours"

    # Solve
    prob.solve(pl.PULP_CBC_CMD(msg=False))

    print("[Task 4] Status:", pl.LpStatus[prob.status])
    print("[Task 4] Optimal production plan:")
    print(f"  Units of A: {x_A.value():.2f}")
    print(f"  Units of B: {x_B.value():.2f}")
    print("[Task 4] Maximum profit: ", pl.value(prob.objective))

    # Simple insights
    print("\n[Insights]")
    print(
        "This solution shows how many units of each product to produce in order to"
        " maximize profit while respecting machine and labor capacity."
    )
    print(
        "You can modify coefficients (profit, resource usage, capacity) to model"
        " different real-world scenarios."
    )


if __name__ == "__main__":
    main()
