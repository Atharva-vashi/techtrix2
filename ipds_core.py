"""Core logic for the iPDS stock control room: validation, AI demand forecast, allocation, FIFO."""
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.ensemble import HistGradientBoostingRegressor

REQUIRED = {
    "stock": ["batch_id", "godown_id", "commodity", "received_date", "quantity_kg"],   # + expiry_date OR shelf_life_days
    "godowns": ["godown_id", "lat", "lon"],
    "shops": ["shop_id", "beneficiaries", "lat", "lon"],
    "history": ["shop_id", "commodity", "month", "qty_kg"],
}


class DataError(Exception):
    pass


def read_uploads(files):
    """Excel workbook (one sheet per table) and/or CSV files named stock/godowns/shops/history."""
    t = {}
    for f in files:
        name = f.name.lower()
        if name.endswith((".xlsx", ".xls")):
            for sheet, df in pd.read_excel(f, sheet_name=None).items():
                t[sheet.lower().strip()] = df
        else:
            t[name.rsplit(".", 1)[0].strip()] = pd.read_csv(f)
    return t


def clean(t, today):
    """Validate and standardise the four tables. Returns (tables, warnings)."""
    t = {k.lower().strip(): v for k, v in t.items()}
    miss = [k for k in REQUIRED if k not in t]
    if miss:
        raise DataError(f"Missing table(s): {', '.join(miss)}. Use sheets or files named stock, godowns, shops, history.")
    warns, out = [], {}
    for k, cols in REQUIRED.items():
        df = t[k].copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        bad = [c for c in cols if c not in df.columns]
        if bad:
            raise DataError(f"'{k}' is missing column(s): {', '.join(bad)}")
        out[k] = df

    def num(df, cols, name):
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        n0 = len(df)
        df = df.dropna(subset=cols)
        if len(df) < n0:
            warns.append(f"{name}: dropped {n0 - len(df)} row(s) with missing or non-numeric values.")
        return df

    for k, idc in [("godowns", ["godown_id"]), ("shops", ["shop_id"]), ("stock", ["batch_id", "godown_id"]),
                   ("history", ["shop_id"])]:
        for c in idc:
            out[k][c] = out[k][c].astype(str).str.strip()
    for k in ("stock", "history"):
        out[k]["commodity"] = out[k]["commodity"].astype(str).str.strip().str.title()

    out["godowns"] = num(out["godowns"], ["lat", "lon"], "godowns")
    out["shops"] = num(out["shops"], ["beneficiaries", "lat", "lon"], "shops")
    out["history"] = num(out["history"], ["qty_kg"], "history")
    s = num(out["stock"], ["quantity_kg"], "stock")
    s["received_date"] = pd.to_datetime(s.received_date, errors="coerce")
    exp = pd.to_datetime(s["expiry_date"], errors="coerce") if "expiry_date" in s.columns else pd.Series(pd.NaT, index=s.index)
    if "shelf_life_days" in s.columns:
        exp = exp.fillna(s.received_date + pd.to_timedelta(pd.to_numeric(s.shelf_life_days, errors="coerce"), unit="D"))
    elif "expiry_date" not in s.columns:
        raise DataError("'stock' needs a column expiry_date or shelf_life_days.")
    s["expiry_date"] = exp
    n0 = len(s)
    s = s.dropna(subset=["received_date", "expiry_date"])
    if len(s) < n0:
        warns.append(f"stock: dropped {n0 - len(s)} row(s) with an unreadable date or no expiry information.")
    unknown = ~s.godown_id.isin(out["godowns"].godown_id)
    if unknown.any():
        warns.append(f"stock: dropped {int(unknown.sum())} batch(es) whose godown_id is not in the godowns table.")
        s = s[~unknown]
    today = pd.Timestamp(today).normalize()
    s["age_days"] = (today - s.received_date).dt.days
    s["days_to_expiry"] = (s.expiry_date - today).dt.days
    out["stock"] = s.reset_index(drop=True)

    h = out["history"]
    h["month"] = pd.to_datetime(h["month"], errors="coerce").dt.to_period("M")
    n0 = len(h)
    h = h.dropna(subset=["month"])
    if len(h) < n0:
        warns.append(f"history: dropped {n0 - len(h)} row(s) with an unreadable month.")
    out["history"] = h
    if out["shops"].empty or out["godowns"].empty or out["stock"].empty or h.empty:
        raise DataError("One of the tables has no usable rows after cleaning.")
    return out, warns


# --------------------------------------------------------------------------- AI forecaster
def forecast(hist, shops, min_rows=30):
    """Predict next month's demand per shop and commodity.

    Model: gradient boosting on *relative* features (last 1/2/3/12 months divided by the shop's recent average),
    month of year and commodity, trained on all shops' history together. Falls back to a moving average when
    there is too little history, and to a per-beneficiary average for shops with no recent history.
    """
    h = hist.groupby(["shop_id", "commodity", "month"], as_index=False).qty_kg.sum()
    months = pd.period_range(h.month.min(), h.month.max(), freq="M")
    keys = h[["shop_id", "commodity"]].drop_duplicates()
    full = pd.MultiIndex.from_tuples([(s, c, m) for s, c in keys.itertuples(index=False) for m in months],
                                     names=["shop_id", "commodity", "month"])
    h = h.set_index(["shop_id", "commodity", "month"]).qty_kg.reindex(full).reset_index()
    g = h.groupby(["shop_id", "commodity"]).qty_kg
    for k in (1, 2, 3, 12):
        h[f"lag{k}"] = g.shift(k)
    h["scale"] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    comm = {c: i for i, c in enumerate(sorted(h.commodity.unique()))}
    mon = h.month.dt.month
    X = pd.DataFrame({"r1": h.lag1 / h.scale, "r2": h.lag2 / h.scale, "r3": h.lag3 / h.scale, "r12": h.lag12 / h.scale,
                      "m_sin": np.sin(2 * np.pi * mon / 12), "m_cos": np.cos(2 * np.pi * mon / 12),
                      "ccode": h.commodity.map(comm)})
    y = h.qty_kg / h.scale
    ok = h.qty_kg.notna() & h.scale.notna() & (h.scale > 0)

    def new_model():
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                             min_samples_leaf=5, random_state=0)

    model, metrics = None, None
    if ok.sum() >= min_rows and len(months) >= 6:
        if len(months) >= 8:
            cut = months[-3]
            tr, te = ok & (h.month < cut), ok & (h.month >= cut)
            if tr.sum() >= min_rows and te.sum() >= 5:
                p = new_model().fit(X[tr], y[tr]).predict(X[te]) * h.scale[te]
                act, nv = h.qty_kg[te], h.lag1[te]
                m = nv.notna() & (act > 0)
                metrics = {"model_err": float(np.mean(np.abs(p - act)[m] / act[m]) * 100),
                           "naive_err": float(np.mean(np.abs(nv - act)[m] / act[m]) * 100), "n_test": int(m.sum())}
        model = new_model().fit(X[ok], y[ok])

    nxt = months[-1] + 1
    bene = shops.set_index("shop_id").beneficiaries
    hh = h.dropna(subset=["qty_kg"]).assign(b=lambda d: d.shop_id.map(bene))
    hh = hh[hh.b > 0]
    perb = (hh.qty_kg / hh.b).groupby(hh.commodity).median()
    series = {k: gg.qty_kg.values for k, gg in h.groupby(["shop_id", "commodity"])}
    rows, feats, idx = [], [], []
    for s in shops.shop_id:
        for c in comm:
            q = series.get((s, c))
            if q is None or np.isnan(q[-3:]).all():
                rows.append([s, c, bene[s] * perb.get(c, 0.0), "cold start (no recent history)"])
                continue
            sc = float(np.nanmean(q[-3:]))
            lag = lambda k: q[-k] if len(q) >= k else np.nan
            if model is not None and sc > 0:
                feats.append({"r1": lag(1) / sc, "r2": lag(2) / sc, "r3": lag(3) / sc, "r12": lag(12) / sc,
                              "m_sin": np.sin(2 * np.pi * nxt.month / 12), "m_cos": np.cos(2 * np.pi * nxt.month / 12),
                              "ccode": comm[c]})
                idx.append((len(rows), sc))
                rows.append([s, c, np.nan, "AI model"])
            else:
                rows.append([s, c, sc, "moving average (too little history for AI)"])
    if feats:
        pred = model.predict(pd.DataFrame(feats))
        for (i, sc), p in zip(idx, pred):
            rows[i][2] = float(p) * sc
    fc = pd.DataFrame(rows, columns=["shop_id", "commodity", "forecast_kg", "method"])
    fc["forecast_kg"] = fc.forecast_kg.clip(lower=0).round()
    return fc, metrics, nxt, h[["shop_id", "commodity", "month", "qty_kg"]]


# --------------------------------------------------------------------------- planning
def haversine_matrix(g, s):
    p = np.pi / 180
    la1, lo1 = g.lat.values[:, None] * p, g.lon.values[:, None] * p
    la2, lo2 = s.lat.values[None, :] * p, s.lon.values[None, :] * p
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return pd.DataFrame(2 * 6371 * np.arcsin(np.sqrt(a)), index=g.godown_id.values, columns=s.shop_id.values)


def make_plan(stock, godowns, shops, fc, window, buffer, road_factor, cost_km, urg_pen):
    stock = stock.sort_values(["godown_id", "commodity", "received_date"]).reset_index(drop=True)
    expired = stock[stock.days_to_expiry < 0]
    stock = stock[stock.days_to_expiry >= 0].reset_index(drop=True)
    stock["near_expiry"] = stock.days_to_expiry <= window
    dist = haversine_matrix(godowns, shops) * road_factor
    comms = sorted(fc.commodity.unique())
    demand = {c: fc[fc.commodity == c].set_index("shop_id").forecast_kg.mul(1 + buffer).round()
              .reindex(shops.shop_id).fillna(0) for c in comms}

    def allocate(c):
        gs = stock[stock.commodity == c].groupby("godown_id").quantity_kg.sum()
        if gs.empty:
            return pd.DataFrame(columns=["godown_id", "shop_id", "qty_kg", "commodity"])
        gu = stock[(stock.commodity == c) & stock.near_expiry].groupby("godown_id").quantity_kg.sum() \
            .reindex(gs.index, fill_value=0)
        d = demand[c]
        G, S = list(gs.index), list(d.index)
        nG, nS = len(G), len(S)
        nx, n = nG * nS, nG * nS + nS + nG
        cost = np.zeros(n)
        for gi, g in enumerate(G):
            cost[gi * nS:(gi + 1) * nS] = dist.loc[g, S].values * cost_km
        cost[nx:nx + nS], cost[nx + nS:] = 50.0, urg_pen
        Aeq, beq = np.zeros((nS, n)), d.values.astype(float)
        for si in range(nS):
            Aeq[si, [gi * nS + si for gi in range(nG)]] = 1
            Aeq[si, nx + si] = 1
        Aub, bub = np.zeros((2 * nG, n)), np.zeros(2 * nG)
        for gi, g in enumerate(G):
            Aub[gi, gi * nS:(gi + 1) * nS] = 1
            bub[gi] = gs[g]
            Aub[nG + gi, gi * nS:(gi + 1) * nS] = -1
            Aub[nG + gi, nx + nS + gi] = -1
            bub[nG + gi] = -gu[g]
        res = linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
        if not res.success:
            raise DataError(f"Optimizer failed for {c}: {res.message}")
        out = pd.DataFrame([(g, s, res.x[gi * nS + si]) for gi, g in enumerate(G) for si, s in enumerate(S)],
                           columns=["godown_id", "shop_id", "qty_kg"])
        out = out[out.qty_kg > 0.5].copy()
        out["commodity"] = c
        return out

    plan = pd.concat([allocate(c) for c in comms], ignore_index=True)
    plan["qty_kg"] = plan.qty_kg.round()

    rem, ledger = stock.set_index("batch_id").quantity_kg.to_dict(), []
    for (g, c), grp in plan.groupby(["godown_id", "commodity"]):
        batches = stock[(stock.godown_id == g) & (stock.commodity == c)].sort_values("received_date").batch_id
        for r in grp.itertuples():
            need = r.qty_kg
            for b in batches:
                take = min(need, rem[b])
                if take > 0:
                    ledger.append({"godown_id": g, "commodity": c, "shop_id": r.shop_id, "batch_id": b, "qty_kg": take})
                    rem[b] -= take
                    need -= take
                if need <= 0:
                    break
    ledger = pd.DataFrame(ledger, columns=["godown_id", "commodity", "shop_id", "batch_id", "qty_kg"])
    stock["left_kg"] = stock.batch_id.map(rem)
    stock["issued_kg"] = stock.quantity_kg - stock.left_kg

    # baseline for comparison: nearest godown first, batches issued in random order (no FIFO)
    avail, nv = stock.groupby(["godown_id", "commodity"]).quantity_kg.sum().to_dict(), []
    for s in shops.shop_id:
        for c in comms:
            need = demand[c][s]
            for g in dist[s].sort_values().index:
                take = min(need, avail.get((g, c), 0))
                if take > 0:
                    nv.append({"godown_id": g, "shop_id": s, "commodity": c, "qty_kg": take})
                    avail[(g, c)] -= take
                    need -= take
                if need <= 0:
                    break
    nv = pd.DataFrame(nv, columns=["godown_id", "shop_id", "commodity", "qty_kg"])
    nrem = stock.set_index("batch_id").quantity_kg.to_dict()
    for (g, c), grp in nv.groupby(["godown_id", "commodity"]):
        tot = grp.qty_kg.sum()
        for r in stock[(stock.godown_id == g) & (stock.commodity == c)].sample(frac=1, random_state=1).itertuples():
            take = min(tot, nrem[r.batch_id])
            nrem[r.batch_id] -= take
            tot -= take
    tc = lambda p: float(sum(r.qty_kg * dist.loc[r.godown_id, r.shop_id] * cost_km for r in p.itertuples()))
    total_demand = sum(d.sum() for d in demand.values())
    k = {"cost_smart": tc(plan), "cost_naive": tc(nv), "issued": float(plan.qty_kg.sum()),
         "urg_smart": float(stock[stock.near_expiry].left_kg.sum()),
         "urg_naive": float(sum(nrem[b] for b in stock[stock.near_expiry].batch_id)),
         "unmet": float(total_demand - plan.qty_kg.sum()), "expired_kg": float(expired.quantity_kg.sum())}
    issued = plan.groupby(["shop_id", "commodity"]).qty_kg.sum()
    unmet = pd.DataFrame([(s, c, demand[c][s], issued.get((s, c), 0.0)) for c in comms for s in shops.shop_id],
                         columns=["shop_id", "commodity", "planned_kg", "allocated_kg"])
    unmet["unmet_kg"] = unmet.planned_kg - unmet.allocated_kg
    return {"stock": stock, "expired": expired, "plan": plan, "ledger": ledger, "kpi": k, "dist": dist,
            "unmet": unmet[unmet.unmet_kg > 0.5], "demand": demand}


# --------------------------------------------------------------------------- template / demo data
def sample_tables(today):
    """Small example data in the exact input format. Used for the downloadable template and for demo mode."""
    r = np.random.default_rng(1)
    today = pd.Timestamp(today).normalize()
    gd = pd.DataFrame({"godown_id": ["G1", "G2", "G3"], "name": ["Godown A", "Godown B", "Godown C"],
                       "lat": [28.62, 28.45, 28.55], "lon": [77.20, 77.30, 77.45]})
    sh = pd.DataFrame({"shop_id": [f"FPS{i:02d}" for i in range(1, 13)], "name": [f"Shop {i}" for i in range(1, 13)],
                       "beneficiaries": r.integers(300, 1200, 12), "lat": r.uniform(28.35, 28.75, 12).round(4),
                       "lon": r.uniform(77.1, 77.6, 12).round(4)})
    rows, n = [], 1
    for g in gd.godown_id:
        for c in ["Rice", "Wheat"]:
            for _ in range(3):
                rows.append({"batch_id": f"B{n:03d}", "godown_id": g, "commodity": c,
                             "received_date": (today - pd.Timedelta(days=int(r.integers(10, 170)))).date(),
                             "quantity_kg": int(r.integers(2, 9)) * 1000, "shelf_life_days": 180 if c == "Rice" else 150})
                n += 1
    months = pd.period_range(end=today.to_period("M") - 1, periods=24, freq="M")
    hr = []
    for s in sh.itertuples():
        up = r.uniform(0.82, 0.97)
        for c, share, amp in [("Rice", 0.6, 0.08), ("Wheat", 0.4, 0.12)]:
            for m in months:
                seas = 1 + amp * np.sin(2 * np.pi * (m.month - 8) / 12)
                hr.append({"shop_id": s.shop_id, "commodity": c, "month": str(m),
                           "qty_kg": round(s.beneficiaries * 5 * share * up * seas * r.normal(1, 0.04))})
    return {"stock": pd.DataFrame(rows), "godowns": gd, "shops": sh, "history": pd.DataFrame(hr)}
