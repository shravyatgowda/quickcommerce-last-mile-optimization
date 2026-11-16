"""Generate a SYNTHETIC quick-commerce order dataset.

No real company data is used. Coordinates are in km on a 12 x 12 km city grid.
Run:  python src/generate_data.py
"""
import numpy as np
import pandas as pd

SEED = 42
N_ORDERS = 400
STORES = {  # dark stores (x_km, y_km)
    "S1": (3.0, 3.0),
    "S2": (8.5, 4.0),
    "S3": (5.5, 9.0),
}


def main() -> None:
    rng = np.random.default_rng(SEED)
    store_ids = rng.choice(list(STORES), size=N_ORDERS, p=[0.4, 0.35, 0.25])
    rows = []
    for i, s in enumerate(store_ids):
        sx, sy = STORES[s]
        # customers cluster within ~3 km of the store (quick-commerce radius)
        cx = np.clip(sx + rng.normal(0, 1.3), 0, 12)
        cy = np.clip(sy + rng.normal(0, 1.3), 0, 12)
        rows.append({
            "order_id": f"O{i:04d}",
            "store_id": s,
            "store_x": sx, "store_y": sy,
            "cust_x": round(float(cx), 3), "cust_y": round(float(cy), 3),
            "order_minute": round(float(rng.uniform(0, 180)), 1),  # 3-hour window
        })
    df = pd.DataFrame(rows).sort_values("order_minute").reset_index(drop=True)
    df["order_id"] = [f"O{i:04d}" for i in range(len(df))]
    df.to_csv("data/orders.csv", index=False)
    print(f"Wrote data/orders.csv with {len(df)} orders")


if __name__ == "__main__":
    main()
