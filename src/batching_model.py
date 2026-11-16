"""Order batching optimisation with Pyomo (HiGHS solver).

Question: how do batching rules (max extra delay for the second order in a
batch) change cost per order and delivery time in a quick-commerce setting?

Model (per scenario):
  x[p] = 1 if candidate pair p = (order i dropped first, order j dropped
         second), from the same dark store and placed within WINDOW_MIN
         minutes of each other, is delivered by one rider on one trip.
  Each order belongs to at most one batch.
  Objective: maximise  (cost of two solo trips - cost of one batched trip)
             - DELAY_PENALTY * extra delay of the second order.
  Feasibility: extra delay of the second order <= max_extra_min.

ALL PARAMETERS BELOW ARE ASSUMPTIONS for a simulation, not measured values.
Run:  python src/batching_model.py
"""
import itertools
import numpy as np
import pandas as pd
import pyomo.environ as pyo

SPEED_KMPH = 20.0           # rider speed
FIXED_COST_PER_TRIP = 18.0  # INR, assumed rider payout per trip
COST_PER_KM = 4.0           # INR per km travelled (incl. return)
PICKUP_MIN = 3.0            # minutes at store per trip
WINDOW_MIN = 1.5            # orders must be placed this close in time to batch
DELAY_PENALTY = 0.8         # INR per minute of extra delay on the 2nd order
SCENARIOS = {
    "no_batching": 0.0,
    "batch_max_extra_3min": 3.0,
    "batch_max_extra_6min": 6.0,
    "batch_max_extra_10min": 10.0,
}


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def solo(r):
    """Distance (km), delivery time (min) and cost (INR) of a solo trip."""
    d = dist((r.store_x, r.store_y), (r.cust_x, r.cust_y))
    return d, PICKUP_MIN + 60 * d / SPEED_KMPH, FIXED_COST_PER_TRIP + COST_PER_KM * 2 * d


def candidate_pairs(df, max_extra_min):
    cands = []
    for _, g in df.groupby("store_id"):
        for a, b in itertools.permutations(list(g.itertuples()), 2):
            if abs(a.order_minute - b.order_minute) > WINDOW_MIN:
                continue
            s, ca, cb = (a.store_x, a.store_y), (a.cust_x, a.cust_y), (b.cust_x, b.cust_y)
            d_a, d_b, d_ab, d_bs = dist(s, ca), dist(s, cb), dist(ca, cb), dist(cb, s)
            delay_b = 60 * (d_a + d_ab - d_b) / SPEED_KMPH  # vs. going straight to b
            if delay_b > max_extra_min:
                continue
            delay_b = max(delay_b, 0.0)
            batch_km = d_a + d_ab + d_bs
            solo_cost = 2 * FIXED_COST_PER_TRIP + COST_PER_KM * (2 * d_a + 2 * d_b)
            batch_cost = FIXED_COST_PER_TRIP + COST_PER_KM * batch_km
            cands.append(dict(i=a.order_id, j=b.order_id, saving=solo_cost - batch_cost,
                              delay_b=delay_b, batch_cost=batch_cost))
    return cands


def solve(cands, order_ids):
    if not cands:
        return []
    m = pyo.ConcreteModel()
    m.P = pyo.RangeSet(0, len(cands) - 1)
    m.x = pyo.Var(m.P, domain=pyo.Binary)
    m.obj = pyo.Objective(
        expr=sum((cands[p]["saving"] - DELAY_PENALTY * cands[p]["delay_b"]) * m.x[p]
                 for p in m.P),
        sense=pyo.maximize)
    by_order = {o: [] for o in order_ids}
    for p, c in enumerate(cands):
        by_order[c["i"]].append(p)
        by_order[c["j"]].append(p)
    m.cover = pyo.ConstraintList()
    for ps in by_order.values():
        if ps:
            m.cover.add(sum(m.x[p] for p in ps) <= 1)
    pyo.SolverFactory("appsi_highs").solve(m)
    return [cands[p] for p in m.P if pyo.value(m.x[p]) > 0.5]


def evaluate(df, chosen):
    role = {}
    for c in chosen:
        role[c["i"]] = ("first", c)
        role[c["j"]] = ("second", c)
    costs, times = [], []
    for r in df.itertuples():
        _, t, c = solo(r)
        if r.order_id in role:
            who, pair = role[r.order_id]
            c = pair["batch_cost"] / 2  # trip cost shared by the two orders
            if who == "second":
                t = t + pair["delay_b"]
        costs.append(c)
        times.append(t)
    return float(np.mean(costs)), float(np.mean(times))


def main():
    df = pd.read_csv("data/orders.csv")
    out = []
    for name, max_extra in SCENARIOS.items():
        chosen = [] if max_extra == 0 else solve(candidate_pairs(df, max_extra), df.order_id.tolist())
        cpo, avg_t = evaluate(df, chosen)
        out.append(dict(scenario=name, max_extra_delay_min=max_extra,
                        batches_formed=len(chosen),
                        orders_batched_pct=round(100 * 2 * len(chosen) / len(df), 1),
                        avg_cost_per_order_inr=round(cpo, 2),
                        avg_delivery_time_min=round(avg_t, 2)))
    res = pd.DataFrame(out)
    base = res.iloc[0]
    res["cost_change_pct"] = ((res.avg_cost_per_order_inr / base.avg_cost_per_order_inr - 1) * 100).round(1)
    res["time_change_pct"] = ((res.avg_delivery_time_min / base.avg_delivery_time_min - 1) * 100).round(1)
    res.to_csv("output/scenario_results.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
