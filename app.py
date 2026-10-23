import io
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ipds_core import DataError, clean, forecast, make_plan, read_uploads, sample_tables

st.set_page_config(page_title="iPDS stock control room", page_icon="🌾", layout="wide")
st.title("🌾 iPDS stock control room")
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@st.cache_data(ttl=60, show_spinner="Fetching live data...")
def fetch_csv(url):
    return pd.read_csv(url)


@st.cache_data(show_spinner="Training the AI model on your history...")
def run_forecast(hist, shops):
    return forecast(hist, shops)


@st.cache_data(show_spinner="Optimising allocation...")
def run_plan(stock, godowns, shops, fc, window, buffer, road, cost_km, urg):
    return make_plan(stock, godowns, shops, fc, window, buffer, road, cost_km, urg)


def template_xlsx():
    t = sample_tables(date.today())
    readme = pd.DataFrame({"Sheet": ["stock", "godowns", "shops", "history"], "What to put in it": [
        "One row per batch in a godown: batch_id, godown_id, commodity, received_date, quantity_kg, and either shelf_life_days or expiry_date.",
        "One row per godown: godown_id, name (optional), lat, lon.",
        "One row per fair price shop: shop_id, name (optional), beneficiaries, lat, lon.",
        "Monthly distribution actually lifted: shop_id, commodity, month (like 2026-09), qty_kg. More months give a better AI forecast (24 or more is ideal)."]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        readme.to_excel(w, sheet_name="README", index=False)
        for k, n in [("stock", 5), ("godowns", 3), ("shops", 5), ("history", 6)]:
            t[k].head(n).to_excel(w, sheet_name=k, index=False)
    return buf.getvalue()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Data")
    source = st.radio("Data source", ["Upload files", "Live CSV links", "Demo sample (testing only)"])
    today = pd.Timestamp(st.date_input("Today", date.today()))
    st.header("Settings")
    window = st.slider("Near-expiry window (days)", 10, 90, 30)
    buffer = st.slider("Safety buffer on forecast (%)", 0, 20, 5) / 100
    road = st.slider("Road distance factor", 1.0, 2.0, 1.3, 0.05,
                     help="Straight-line distance between coordinates is multiplied by this to estimate road distance.")
    cost_km = st.slider("Transport cost (Rs per kg per km)", 0.005, 0.10, 0.02, 0.005)
    urg = st.slider("Priority for clearing old stock", 1.0, 20.0, 5.0, 0.5,
                    help="Higher means the optimizer accepts more transport cost to clear near-expiry stock.")

tables = None
if source == "Upload files":
    files = st.file_uploader("Upload one Excel workbook (sheets: stock, godowns, shops, history) or four CSV files",
                             type=["xlsx", "xls", "csv"], accept_multiple_files=True)
    if files:
        try:
            tables = read_uploads(files)
        except Exception as e:
            st.error(f"Could not read the files: {e}")
elif source == "Live CSV links":
    st.write("Paste a CSV link for each table, for example a Google Sheet published to the web as CSV. "
             "Data is re-read about once a minute, or press the button.")
    if st.button("Refresh now"):
        st.cache_data.clear()
    urls = {k: st.text_input(f"{k} CSV link") for k in ["stock", "godowns", "shops", "history"]}
    if all(urls.values()):
        try:
            tables = {k: fetch_csv(u) for k, u in urls.items()}
        except Exception as e:
            st.error(f"Could not fetch a link: {e}")
else:
    tables = sample_tables(today)
    st.warning("Demo mode: this is made-up example data for testing the app, not real iPDS data.")

if tables is None:
    st.info("Add your data to begin. The template shows the exact columns needed.")
    st.download_button("Download the Excel template", template_xlsx(), "ipds_input_template.xlsx", XLSX)
    st.stop()

tabs = st.tabs(["Data", "Overview", "FIFO queue", "Allocation", "AI forecast", "Download"])

with tabs[0]:
    st.subheader("Current stock (editable)")
    st.caption("Add a row when a truck arrives, change a quantity, or delete a batch. Everything recalculates.")
    stock_key = next((k for k in tables if k.lower().strip() == "stock"), None)
    if stock_key:
        tables[stock_key] = st.data_editor(tables[stock_key], num_rows="dynamic", key=f"ed_{source}", width="stretch")
    st.download_button("Excel template", template_xlsx(), "ipds_input_template.xlsx", XLSX)
try:
    tab, warns = clean(tables, today)
    fc, metrics, nxt, hist = run_forecast(tab["history"], tab["shops"])
    R = run_plan(tab["stock"], tab["godowns"], tab["shops"], fc, window, buffer, road, cost_km, urg)
except DataError as e:
    with tabs[0]:
        st.error(str(e))
    st.stop()
with tabs[0]:
    for w in warns:
        st.warning(w)
    st.success(f"Loaded {len(tab['stock'])} batches, {len(tab['godowns'])} godowns, {len(tab['shops'])} shops and "
               f"{len(tab['history'])} monthly records. Planning month: {nxt}.")

stock, plan, k = R["stock"], R["plan"], R["kpi"]
godowns, shops = tab["godowns"], tab["shops"]
gname = dict(zip(godowns.godown_id, godowns["name"] if "name" in godowns.columns else godowns.godown_id))
comms = sorted(fc.commodity.unique())

with tabs[1]:
    c = st.columns(5)
    c[0].metric("To issue next month", f"{k['issued']:,.0f} kg")
    ch = (k["cost_smart"] / k["cost_naive"] - 1) * 100 if k["cost_naive"] else 0
    c[1].metric("Transport cost", f"Rs {k['cost_smart']:,.0f}", f"{ch:+.1f}% vs nearest-godown", delta_color="inverse")
    c[2].metric("Near-expiry stock left", f"{k['urg_smart']:,.0f} kg", f"nearest-godown plan: {k['urg_naive']:,.0f} kg", delta_color="off")
    c[3].metric("Unmet demand", f"{k['unmet']:,.0f} kg")
    c[4].metric("Forecast error (test)", f"{metrics['model_err']:.1f}%" if metrics else "n/a",
                f"last-month guess: {metrics['naive_err']:.1f}%" if metrics else None, delta_color="off")
    st.caption("The smart plan may cost a little more in transport when it routes stock to clear batches close to expiry.")
    if k["unmet"] > 0.5:
        st.warning("Stock is not enough for the forecast demand. Shops with a shortfall:")
        st.dataframe(R["unmet"], hide_index=True, width="stretch")
    if len(R["expired"]):
        st.error(f"{len(R['expired'])} expired batch(es), {k['expired_kg']:,.0f} kg. Not used in the plan.")
        st.dataframe(R["expired"][["batch_id", "godown_id", "commodity", "quantity_kg", "expiry_date"]],
                     hide_index=True, width="stretch")
    st.subheader("Issue first: near-expiry batches")
    al = stock[stock.near_expiry][["batch_id", "godown_id", "commodity", "quantity_kg", "expiry_date", "days_to_expiry", "left_kg"]]
    st.dataframe(al, hide_index=True, width="stretch") if len(al) else st.info("No batches within the window.")

with tabs[2]:
    st.caption("Oldest batch on the left. Colour shows shelf life used (green to red). Hatching is what the plan issues.")
    for g in godowns.godown_id:
        fig, axes = plt.subplots(len(comms), 1, figsize=(10, 1.3 * len(comms) + 0.4), squeeze=False)
        for ax, cm in zip(axes[:, 0], comms):
            x = 0
            for r in stock[(stock.godown_id == g) & (stock.commodity == cm)].sort_values("received_date").itertuples():
                life = min(max(r.age_days / max((r.expiry_date - r.received_date).days, 1), 0), 1)
                ax.barh(0, r.quantity_kg, left=x, color=plt.cm.RdYlGn_r(life), edgecolor="white")
                ax.barh(0, r.issued_kg, left=x, color="none", hatch="//", edgecolor="black", lw=0)
                ax.text(x + r.quantity_kg / 2, 0, r.batch_id, ha="center", va="center", fontsize=7)
                x += r.quantity_kg
            ax.set_yticks([]); ax.set_ylabel(cm, rotation=0, ha="right"); ax.margins(x=0)
        fig.suptitle(gname[g], x=0.01, ha="left"); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

with tabs[3]:
    sel = st.selectbox("Godown", godowns.godown_id, format_func=gname.get)
    gp, sp = godowns.set_index("godown_id"), shops.set_index("shop_id")
    rows = plan[plan.godown_id == sel].groupby("shop_id").qty_kg.sum()
    fig, ax = plt.subplots(figsize=(9, 5))
    for s, q in rows.items():
        ax.plot([gp.lon[sel], sp.lon[s]], [gp.lat[sel], sp.lat[s]], color="#1f4e5f", alpha=.5, lw=1 + q / max(rows.max() / 4, 1))
    ax.scatter(sp.lon, sp.lat, c=["#1f4e5f" if s in rows.index else "#cccccc" for s in sp.index], s=40, zorder=3)
    for g in gp.index:
        ax.scatter(gp.lon[g], gp.lat[g], marker="s", s=150, c="#b7791f" if g == sel else "#5d6b74", zorder=4)
        ax.text(gp.lon[g], gp.lat[g], "  " + gname[g], fontsize=9, zorder=5)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); st.pyplot(fig); plt.close(fig)
    piv = plan[plan.godown_id == sel].pivot_table(index="shop_id", columns="commodity", values="qty_kg", aggfunc="sum").fillna(0)
    piv["Total"] = piv.sum(axis=1)
    st.write(f"**{gname[sel]}** sends {piv.Total.sum():,.0f} kg to {len(piv)} shops")
    st.dataframe(piv, width="stretch")

with tabs[4]:
    st.subheader(f"AI demand forecast for {nxt}")
    st.write("A gradient boosting model learns from every shop's monthly history: the last 1, 2, 3 and 12 months relative to "
             "the shop's recent average, plus month of year (seasonality) and commodity. Your safety buffer is added on top.")
    if metrics:
        st.success(f"Tested on the last 3 months it had not seen ({metrics['n_test']} cases): average error "
                   f"{metrics['model_err']:.1f}% against {metrics['naive_err']:.1f}% for a 'same as last month' guess.")
    else:
        st.info("Not enough history to measure accuracy on held-out months (needs about 8 months from many shops).")
    st.write("Methods used:", fc.method.value_counts().to_dict())
    a, b = st.columns(2)
    shop, com = a.selectbox("Shop", shops.shop_id), b.selectbox("Commodity", comms)
    h = hist[(hist.shop_id == shop) & (hist.commodity == com)]
    f = fc[(fc.shop_id == shop) & (fc.commodity == com)]
    if len(h) and len(f):
        chart = pd.DataFrame({"Actual": h.qty_kg.values}, index=h.month.astype(str).values)
        chart.loc[str(nxt), "Forecast"] = float(f.forecast_kg.iloc[0])
        chart.loc[chart.index[-2], "Forecast"] = chart["Actual"].iloc[-1]
        st.line_chart(chart)
        st.write(f"Forecast {f.forecast_kg.iloc[0]:,.0f} kg, planned with buffer {f.forecast_kg.iloc[0] * (1 + buffer):,.0f} kg ({f.method.iloc[0]}).")
    st.dataframe(fc, hide_index=True, width="stretch")

with tabs[5]:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        stock.to_excel(w, sheet_name="Stock_Batches_FIFO", index=False)
        al.to_excel(w, sheet_name="Near_Expiry_Alerts", index=False)
        plan.to_excel(w, sheet_name="Allocation_Plan", index=False)
        R["ledger"].to_excel(w, sheet_name="FIFO_Dispatch_Ledger", index=False)
        fc.to_excel(w, sheet_name="Demand_Forecast", index=False)
        R["unmet"].to_excel(w, sheet_name="Unmet_Demand", index=False)
    st.download_button("Download plan as Excel", buf.getvalue(), "ipds_plan.xlsx", XLSX)
