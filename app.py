import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Intelligence Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL STYLES ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:        #0d0f14;
    --bg2:       #13161e;
    --bg3:       #1a1e2a;
    --border:    #252a38;
    --accent:    #4f8eff;
    --accent2:   #00e5c3;
    --accent3:   #ff6b6b;
    --gold:      #f5c842;
    --text:      #e2e8f0;
    --muted:     #6b7694;
    --card-glow: 0 0 30px rgba(79,142,255,0.08);
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Main content */
.main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.5rem !important;
    box-shadow: var(--card-glow) !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-family: 'DM Mono', monospace !important; font-size: 0.72rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 1.8rem !important; }
[data-testid="stMetricDelta"] { font-family: 'DM Mono', monospace !important; }

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Selectbox / inputs */
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Dividers */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* Dataframes */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px !important; }

/* Section headers */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 0.2rem;
}
.insight-box {
    background: linear-gradient(135deg, rgba(79,142,255,0.08), rgba(0,229,195,0.05));
    border: 1px solid rgba(79,142,255,0.25);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    line-height: 1.6;
}
.warn-box {
    background: rgba(255,107,107,0.07);
    border: 1px solid rgba(255,107,107,0.25);
    border-left: 3px solid var(--accent3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
}
.success-box {
    background: rgba(0,229,195,0.07);
    border: 1px solid rgba(0,229,195,0.25);
    border-left: 3px solid var(--accent2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB DARK THEME ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#13161e",
    "axes.facecolor":    "#1a1e2a",
    "axes.edgecolor":    "#252a38",
    "axes.labelcolor":   "#9aa3b8",
    "axes.titlecolor":   "#e2e8f0",
    "xtick.color":       "#6b7694",
    "ytick.color":       "#6b7694",
    "grid.color":        "#252a38",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "text.color":        "#e2e8f0",
    "legend.facecolor":  "#1a1e2a",
    "legend.edgecolor":  "#252a38",
    "font.family":       "monospace",
})
ACCENT   = "#4f8eff"
ACCENT2  = "#00e5c3"
ACCENT3  = "#ff6b6b"
GOLD     = "#f5c842"
PALETTE  = [ACCENT, ACCENT2, ACCENT3, GOLD, "#c084fc", "#fb923c"]

# ── DATA LOADER ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file, encoding="ISO-8859-1")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df_clean = df[(df['Quantity'] > 0) & (df['Price'] > 0)].copy()
    df_clean['Revenue'] = df_clean['Quantity'] * df_clean['Price']
    return df, df_clean

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem'>
        <div style='font-family:DM Mono,monospace;font-size:0.65rem;letter-spacing:0.15em;color:#6b7694;text-transform:uppercase'>Project</div>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#e2e8f0;line-height:1.2'>Retail Intelligence<br>Engine</div>
        <div style='font-family:DM Mono,monospace;font-size:0.7rem;color:#4f8eff;margin-top:0.3rem'>Neel Shah · neeljshah</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader("Upload online_retail_II.csv", type=["csv"])

    if uploaded:
        st.divider()
        st.markdown("<div class='section-label'>Filters</div>", unsafe_allow_html=True)
        country_placeholder = st.empty()
        churn_days = st.slider("Churn Threshold (days)", 30, 180, 90, 10)
        k_clusters = st.slider("Customer Segments (k)", 2, 6, 3)
        forecast_days = st.slider("Forecast Horizon (days)", 30, 180, 90, 30)

    st.divider()
    st.markdown("""
    <div style='font-family:DM Mono,monospace;font-size:0.65rem;color:#6b7694;line-height:1.8'>
        Phase 1 · Revenue Forecasting<br>
        Phase 2 · Customer Segmentation<br>
        Phase 3 · Marketing Attribution<br>
        Phase 4 · Product KPIs
    </div>
    """, unsafe_allow_html=True)

# ── HERO HEADER ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:2rem'>
    <div style='font-family:DM Mono,monospace;font-size:0.7rem;letter-spacing:0.18em;color:#6b7694;text-transform:uppercase'>Unified Business Intelligence</div>
    <div style='font-family:Syne,sans-serif;font-size:2.8rem;font-weight:800;line-height:1.1;margin:0.3rem 0'>
        Retail Intelligence Engine
    </div>
    <div style='font-family:DM Mono,monospace;font-size:0.82rem;color:#6b7694;max-width:600px;line-height:1.6'>
        4-phase analytics pipeline · Forecasting · Segmentation · Attribution · KPIs
    </div>
</div>
""", unsafe_allow_html=True)

# ── NO FILE STATE ──────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#13161e,#1a1e2a);border:1px dashed #252a38;border-radius:16px;padding:3rem;text-align:center;margin-top:1rem'>
        <div style='font-size:2.5rem;margin-bottom:1rem'>📂</div>
        <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:0.5rem'>Upload your dataset to begin</div>
        <div style='font-family:DM Mono,monospace;font-size:0.78rem;color:#6b7694'>
            Drop <code style='color:#4f8eff'>online_retail_II.csv</code> in the sidebar uploader<br>
            Available at UCI Machine Learning Repository
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset..."):
    df_raw, df = load_data(uploaded)

countries = sorted(df['Country'].unique())
with country_placeholder:
    selected_country = st.selectbox("Country", countries, index=countries.index("United Kingdom") if "United Kingdom" in countries else 0)

df_c = df[df['Country'] == selected_country].copy()

# ── OVERVIEW METRICS ───────────────────────────────────────────────────────────
st.divider()
col1, col2, col3, col4, col5 = st.columns(5)
total_rev    = df_c['Revenue'].sum()
total_orders = df_c['Invoice'].nunique()
total_prods  = df_c['Description'].nunique()
aov          = total_rev / total_orders if total_orders else 0
cust_col     = 'Customer ID' if 'Customer ID' in df_c.columns else 'CustomerID'
total_custs  = df_c[cust_col].nunique()

col1.metric("Total Revenue",   f"£{total_rev:,.0f}")
col2.metric("Total Orders",    f"{total_orders:,}")
col3.metric("Unique Products", f"{total_prods:,}")
col4.metric("Avg Order Value", f"£{aov:.2f}")
col5.metric("Unique Customers",f"{total_custs:,}")

st.divider()

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Phase 1 · Forecasting",
    "🧩  Phase 2 · Segmentation",
    "📣  Phase 3 · Attribution",
    "🏆  Phase 4 · Product KPIs",
])

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — REVENUE FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-label'>Phase 1</div><div class='section-title'>Revenue Forecasting & Pricing Elasticity</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])

    with col_a:
        # Daily revenue chart (matplotlib — no prophet dependency required at runtime)
        daily = df_c.resample('D', on='InvoiceDate')['Revenue'].sum().reset_index()
        daily.columns = ['ds', 'y']

        # Simple 30-day rolling mean as forecast proxy (avoids Prophet install issues)
        daily['rolling'] = daily['y'].rolling(14, min_periods=1).mean()

        # Extend forecast
        last_date = daily['ds'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
        avg_rev = daily['y'].tail(60).mean()
        trend   = (daily['y'].tail(30).mean() - daily['y'].tail(60).mean()) / 30
        future_vals = [max(0, avg_rev + trend * i + np.random.normal(0, avg_rev * 0.08)) for i in range(forecast_days)]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(daily['ds'], daily['y'], alpha=0.15, color=ACCENT)
        ax.plot(daily['ds'], daily['y'], color=ACCENT, lw=1.2, alpha=0.7, label='Actual Revenue')
        ax.plot(daily['ds'], daily['rolling'], color=ACCENT2, lw=1.8, label='14-day MA')
        ax.fill_between(future_dates, [v * 0.82 for v in future_vals], [v * 1.18 for v in future_vals],
                        alpha=0.12, color=GOLD)
        ax.plot(future_dates, future_vals, color=GOLD, lw=1.8, linestyle='--', label=f'{forecast_days}-day Forecast')
        ax.axvline(last_date, color='#252a38', lw=1.5, linestyle=':')
        ax.set_title(f'Daily Revenue — {selected_country}', pad=12, fontsize=11)
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Revenue (£)', fontsize=9)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        forecast_total = sum(future_vals)
        monthly_avg    = daily['y'].resample('ME', on='ds').sum().mean() if hasattr(daily.set_index('ds')['y'].resample('ME'), 'sum') else daily['y'].mean() * 30
        st.metric("Forecast Revenue", f"£{forecast_total:,.0f}", f"Next {forecast_days} days")
        st.metric("Daily Avg (Historical)", f"£{daily['y'].mean():,.0f}")
        peak_day = daily.loc[daily['y'].idxmax(), 'ds'].strftime('%b %d, %Y')
        st.metric("Peak Revenue Day", peak_day)

    st.divider()

    # Pricing Elasticity
    st.markdown("<div class='section-label'>Pricing Analysis</div>", unsafe_allow_html=True)
    top_products = df_c.groupby('Description')['Quantity'].sum().nlargest(20).index.tolist()
    selected_prod = st.selectbox("Select Product for Elasticity Analysis", top_products)

    prod_df = df_c[df_c['Description'] == selected_prod]
    elast_df = prod_df.groupby('Price')['Quantity'].mean().reset_index()
    elast_df = elast_df[elast_df['Price'] > 0]

    col_e1, col_e2 = st.columns([2, 1])
    with col_e1:
        if len(elast_df) >= 3:
            X = sm.add_constant(np.log(elast_df['Price']))
            y_e = np.log(elast_df['Quantity'])
            reg = sm.OLS(y_e, X).fit()
            elast_val = reg.params.iloc[1]

            x_line = np.linspace(elast_df['Price'].min(), elast_df['Price'].max(), 100)
            y_line = np.exp(reg.params.iloc[0] + reg.params.iloc[1] * np.log(x_line))

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.scatter(elast_df['Price'], elast_df['Quantity'], color=ACCENT, s=60, zorder=5, alpha=0.8)
            ax2.plot(x_line, y_line, color=ACCENT3, lw=2, label=f'Elasticity: {elast_val:.2f}')
            ax2.set_title(f'Price vs Demand — {selected_prod[:40]}', fontsize=10, pad=10)
            ax2.set_xlabel('Price (£)', fontsize=9)
            ax2.set_ylabel('Avg Quantity Sold', fontsize=9)
            ax2.legend(fontsize=9)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close()
        else:
            st.info("Not enough price variation for this product. Try another.")
            elast_val = None

    with col_e2:
        if elast_val is not None:
            st.metric("Price Elasticity", f"{elast_val:.2f}")
            if elast_val < -1:
                st.markdown("<div class='warn-box'>⚠️ <strong>Price Sensitive</strong><br>Demand drops sharply with price increases. Do not raise prices on this SKU.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='success-box'>✅ <strong>Price Inelastic</strong><br>Demand is stable under price changes. Safe margin to raise price.</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-label'>Phase 2</div><div class='section-title'>Customer Segmentation & Churn Prediction</div>", unsafe_allow_html=True)

    df_cust = df_c.dropna(subset=[cust_col]).copy()

    if len(df_cust) < 10:
        st.warning("Not enough customer data for this country. Try United Kingdom.")
    else:
        snapshot = df_cust['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = df_cust.groupby(cust_col).agg(
            Recency=('InvoiceDate', lambda x: (snapshot - x.max()).days),
            Frequency=('Invoice', 'nunique'),
            Monetary=('Revenue', 'sum')
        ).reset_index()
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

        # Scale & cluster
        scaler   = StandardScaler()
        rfm_log  = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
        rfm_sc   = scaler.fit_transform(rfm_log)
        km       = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        rfm['Cluster'] = km.fit_predict(rfm_sc)

        # Label clusters by mean monetary
        cluster_means = rfm.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False)
        labels = ['VIP 👑', 'Loyal 💙', 'At-Risk ⚠️', 'Dormant 💤', 'New 🌱', 'Low-Value 📦']
        label_map = {c: labels[i] for i, c in enumerate(cluster_means.index)}
        rfm['Segment'] = rfm['Cluster'].map(label_map)

        # Churn
        rfm['Is_Churned'] = (rfm['Recency'] > churn_days).astype(int)
        X_ch = rfm[['Frequency', 'Monetary']]
        y_ch = rfm['Is_Churned']
        if y_ch.nunique() > 1:
            X_tr, X_te, y_tr, y_te = train_test_split(X_ch, y_ch, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_tr, y_tr)
            churn_acc = clf.score(X_te, y_te)
        else:
            churn_acc = None

        # Metrics row
        vip_count   = (rfm['Segment'] == 'VIP 👑').sum()
        churn_count = rfm['Is_Churned'].sum()
        churn_rate  = rfm['Is_Churned'].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{len(rfm):,}")
        c2.metric("VIP Customers", f"{vip_count:,}")
        c3.metric("Churned Customers", f"{churn_count:,}", f"{churn_rate:.1%} rate")
        c4.metric("Churn Model Accuracy", f"{churn_acc:.1%}" if churn_acc else "N/A")

        st.divider()

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            fig3, ax3 = plt.subplots(figsize=(7, 5))
            seg_colors = [ACCENT, ACCENT2, ACCENT3, GOLD, "#c084fc", "#fb923c"]
            for i, (seg, grp) in enumerate(rfm.groupby('Segment')):
                ax3.scatter(grp['Recency'], grp['Monetary'], s=18, alpha=0.55,
                            color=seg_colors[i % len(seg_colors)], label=seg)
            ax3.set_yscale('log')
            ax3.set_title('Customer Segments — Recency vs Monetary', fontsize=10, pad=10)
            ax3.set_xlabel('Recency (days since last purchase)', fontsize=9)
            ax3.set_ylabel('Total Spend (£, log scale)', fontsize=9)
            ax3.legend(fontsize=7, ncol=2)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close()

        with col_s2:
            seg_counts = rfm['Segment'].value_counts()
            fig4, ax4 = plt.subplots(figsize=(7, 5))
            bars = ax4.barh(seg_counts.index, seg_counts.values,
                            color=[seg_colors[i % len(seg_colors)] for i in range(len(seg_counts))],
                            height=0.6)
            for bar, val in zip(bars, seg_counts.values):
                ax4.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                         f'{val:,}', va='center', fontsize=8, color='#e2e8f0')
            ax4.set_title('Customers per Segment', fontsize=10, pad=10)
            ax4.set_xlabel('Customer Count', fontsize=9)
            ax4.invert_yaxis()
            fig4.tight_layout()
            st.pyplot(fig4)
            plt.close()

        # RFM summary table
        st.divider()
        st.markdown("<div class='section-label'>Segment Summary</div>", unsafe_allow_html=True)
        summary = rfm.groupby('Segment').agg(
            Customers=('CustomerID', 'count'),
            Avg_Recency=('Recency', 'mean'),
            Avg_Frequency=('Frequency', 'mean'),
            Avg_Monetary=('Monetary', 'mean'),
            Total_Revenue=('Monetary', 'sum')
        ).round(1).sort_values('Total_Revenue', ascending=False)
        summary['Avg_Monetary'] = summary['Avg_Monetary'].map('£{:,.0f}'.format)
        summary['Total_Revenue'] = summary['Total_Revenue'].map('£{:,.0f}'.format)
        st.dataframe(summary, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — MARKETING ATTRIBUTION & A/B TEST
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-label'>Phase 3</div><div class='section-title'>Marketing Attribution & A/B Testing</div>", unsafe_allow_html=True)

    df_cust3 = df_c.dropna(subset=[cust_col]).copy()

    if len(df_cust3) < 10:
        st.warning("Not enough data for this country.")
    else:
        snapshot3 = df_cust3['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm3 = df_cust3.groupby(cust_col).agg(
            Recency=('InvoiceDate', lambda x: (snapshot3 - x.max()).days),
            Frequency=('Invoice', 'nunique'),
            Monetary=('Revenue', 'sum')
        ).reset_index()
        rfm3.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

        def assign_channel(row):
            if row['Frequency'] > 5: return 'Email'
            elif row['Monetary'] > 1000: return 'Google Ads'
            else: return 'Social Media'

        rfm3['Channel'] = rfm3.apply(assign_channel, axis=1)
        channel_costs = {'Google Ads': 5.0, 'Social Media': 2.0, 'Email': 0.5}
        rfm3['Cost'] = rfm3['Channel'].map(channel_costs)

        perf = rfm3.groupby('Channel').agg(
            Customers=('CustomerID', 'count'),
            Revenue=('Monetary', 'sum'),
            Cost=('Cost', 'sum')
        )
        perf['ROI'] = ((perf['Revenue'] - perf['Cost']) / perf['Cost']).round(1)
        perf['AOV'] = (perf['Revenue'] / perf['Customers']).round(2)

        col_r1, col_r2, col_r3 = st.columns(3)
        for col, (ch, row) in zip([col_r1, col_r2, col_r3], perf.iterrows()):
            col.metric(f"{ch} ROI", f"{row['ROI']:.0f}x", f"£{row['Revenue']:,.0f} revenue")

        st.divider()

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            ch_colors = [ACCENT, ACCENT2, GOLD]
            fig5, ax5 = plt.subplots(figsize=(7, 4.5))
            bars5 = ax5.bar(perf.index, perf['ROI'], color=ch_colors[:len(perf)], width=0.5)
            for bar, val in zip(bars5, perf['ROI']):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{val:.0f}x', ha='center', fontsize=10, fontweight='bold', color='#e2e8f0')
            ax5.axhline(0, color='#252a38', lw=1.5)
            ax5.set_title('Marketing Channel ROI', fontsize=10, pad=10)
            ax5.set_ylabel('Return on Investment (×)', fontsize=9)
            fig5.tight_layout()
            st.pyplot(fig5)
            plt.close()

        with col_c2:
            fig6, ax6 = plt.subplots(figsize=(7, 4.5))
            ax6.bar(perf.index, perf['Customers'], color=ch_colors[:len(perf)], width=0.5, alpha=0.7)
            ax6.set_title('Customers per Channel', fontsize=10, pad=10)
            ax6.set_ylabel('Customer Count', fontsize=9)
            fig6.tight_layout()
            st.pyplot(fig6)
            plt.close()

        st.divider()
        st.markdown("<div class='section-label'>A/B Test Simulator</div>", unsafe_allow_html=True)

        col_ab1, col_ab2 = st.columns(2)
        with col_ab1:
            loc_a = st.slider("Group A — Free Shipping (avg £)", 100, 250, 150)
            loc_b = st.slider("Group B — Discount Coupon (avg £)", 100, 250, 135)
            sample_size = st.slider("Sample Size per Group", 100, 2000, 500)

        np.random.seed(42)
        group_a = np.random.normal(loc=loc_a, scale=40, size=sample_size)
        group_b = np.random.normal(loc=loc_b, scale=35, size=sample_size)
        t_stat, p_val = stats.ttest_ind(group_a, group_b)
        lift = ((group_a.mean() - group_b.mean()) / group_b.mean()) * 100

        with col_ab2:
            st.metric("Group A (Free Shipping)", f"£{group_a.mean():.2f}")
            st.metric("Group B (Discount)", f"£{group_b.mean():.2f}")
            st.metric("Revenue Lift", f"{lift:+.1f}%")
            st.metric("P-Value", f"{p_val:.4f}", "✅ Significant" if p_val < 0.05 else "❌ Not Significant")

        fig7, ax7 = plt.subplots(figsize=(10, 4))
        ax7.fill_between(*sns.kdeplot(group_a, ax=ax7, color=ACCENT, fill=False).get_lines()[-1].get_data(),
                         alpha=0.15, color=ACCENT)
        ax7.fill_between(*sns.kdeplot(group_b, ax=ax7, color=ACCENT3, fill=False).get_lines()[-1].get_data(),
                         alpha=0.15, color=ACCENT3)
        ax7.set_title('A/B Test — Revenue Distribution', fontsize=10, pad=10)
        ax7.set_xlabel('Revenue per Customer (£)', fontsize=9)
        ax7.set_ylabel('Density', fontsize=9)
        patch_a = mpatches.Patch(color=ACCENT, label=f'Free Shipping (£{group_a.mean():.0f})')
        patch_b = mpatches.Patch(color=ACCENT3, label=f'Discount (£{group_b.mean():.0f})')
        ax7.legend(handles=[patch_a, patch_b], fontsize=9)

        if p_val < 0.05:
            st.markdown(f"<div class='success-box'>✅ <strong>Statistically Significant</strong> (p = {p_val:.4f}) — Free Shipping generated <strong>{lift:+.1f}%</strong> higher revenue. Recommend deploying Free Shipping campaign.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='warn-box'>⚠️ <strong>Not Significant</strong> (p = {p_val:.4f}) — No reliable difference detected. Use the more cost-effective option.</div>", unsafe_allow_html=True)

        fig7.tight_layout()
        st.pyplot(fig7)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — PRODUCT KPIs
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-label'>Phase 4</div><div class='section-title'>Product KPIs & Profitability Matrix</div>", unsafe_allow_html=True)

    df_raw_c = df_raw[df_raw['Country'] == selected_country].copy()
    df_raw_c['Revenue'] = df_raw_c['Quantity'] * df_raw_c['Price']
    df_raw_c['Is_Return'] = df_raw_c['Quantity'] < 0

    prod = df_raw_c.groupby('Description').agg(
        Qty_Sold=('Quantity', lambda x: x[x > 0].sum()),
        Returns=('Is_Return', 'sum'),
        Avg_Price=('Price', 'mean'),
        Revenue=('Revenue', 'sum')
    ).reset_index()
    prod = prod[prod['Qty_Sold'] > 0]
    prod['Return_Rate'] = (prod['Returns'] / prod['Qty_Sold']).clip(0, 1)
    total_vol = prod['Qty_Sold'].sum()
    prod['Turnover_Share'] = prod['Qty_Sold'] / total_vol

    med_qty = prod['Qty_Sold'].median()
    med_ret = prod['Return_Rate'].median()

    def categorize(row):
        if row['Qty_Sold'] > med_qty:
            return 'Hero ⭐' if row['Return_Rate'] <= med_ret else 'Problem ⚠️'
        return 'Low Volume 📦'

    prod['Category'] = prod.apply(categorize, axis=1)

    aov4 = df_c.groupby('Invoice')['Revenue'].sum().mean()
    hero_count   = (prod['Category'] == 'Hero ⭐').sum()
    problem_count = (prod['Category'] == 'Problem ⚠️').sum()
    ret_rate_overall = df_raw_c['Is_Return'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Order Value", f"£{aov4:.2f}")
    c2.metric("Hero Products", f"{hero_count:,}")
    c3.metric("Problem Products", f"{problem_count:,}")
    c4.metric("Overall Return Rate", f"{ret_rate_overall:.1%}")

    st.divider()

    col_p1, col_p2 = st.columns([3, 2])

    with col_p1:
        cat_colors = {'Hero ⭐': ACCENT2, 'Problem ⚠️': ACCENT3, 'Low Volume 📦': '#6b7694'}
        fig8, ax8 = plt.subplots(figsize=(8, 5.5))
        for cat, grp in prod.groupby('Category'):
            ax8.scatter(grp['Qty_Sold'], grp['Return_Rate'],
                        color=cat_colors.get(cat, ACCENT), s=15, alpha=0.5, label=cat)
        ax8.axhline(med_ret, color='#ff6b6b', lw=1, linestyle='--', alpha=0.7, label='Median Return Rate')
        ax8.axvline(med_qty, color='#4f8eff', lw=1, linestyle='--', alpha=0.7, label='Median Volume')
        ax8.set_xscale('log')
        ax8.set_title('Product Profitability Matrix', fontsize=10, pad=10)
        ax8.set_xlabel('Units Sold (log scale)', fontsize=9)
        ax8.set_ylabel('Return Rate', fontsize=9)
        ax8.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax8.legend(fontsize=8)
        fig8.tight_layout()
        st.pyplot(fig8)
        plt.close()

    with col_p2:
        cat_counts = prod['Category'].value_counts()
        fig9, ax9 = plt.subplots(figsize=(5, 5.5))
        wedge_colors = [cat_colors.get(c, ACCENT) for c in cat_counts.index]
        wedges, texts, autotexts = ax9.pie(
            cat_counts.values, labels=cat_counts.index,
            colors=wedge_colors, autopct='%1.0f%%',
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor='#13161e', linewidth=2)
        )
        for t in texts: t.set_fontsize(8)
        for at in autotexts: at.set_fontsize(8); at.set_color('#e2e8f0')
        ax9.set_title('Product Mix', fontsize=10, pad=10)
        fig9.tight_layout()
        st.pyplot(fig9)
        plt.close()

    st.divider()

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("<div class='section-label'>Top 10 Hero Products</div>", unsafe_allow_html=True)
        heroes = prod[prod['Category'] == 'Hero ⭐'].sort_values('Qty_Sold', ascending=False).head(10)
        heroes_display = heroes[['Description', 'Qty_Sold', 'Avg_Price', 'Revenue', 'Return_Rate']].copy()
        heroes_display['Revenue'] = heroes_display['Revenue'].map('£{:,.0f}'.format)
        heroes_display['Avg_Price'] = heroes_display['Avg_Price'].map('£{:.2f}'.format)
        heroes_display['Return_Rate'] = heroes_display['Return_Rate'].map('{:.1%}'.format)
        heroes_display.columns = ['Product', 'Units Sold', 'Avg Price', 'Revenue', 'Return Rate']
        st.dataframe(heroes_display.reset_index(drop=True), use_container_width=True, hide_index=True)

    with col_h2:
        st.markdown("<div class='section-label'>Top 10 Problem Products</div>", unsafe_allow_html=True)
        problems = prod[prod['Category'] == 'Problem ⚠️'].sort_values('Return_Rate', ascending=False).head(10)
        if len(problems):
            prob_display = problems[['Description', 'Qty_Sold', 'Revenue', 'Return_Rate']].copy()
            prob_display['Revenue'] = prob_display['Revenue'].map('£{:,.0f}'.format)
            prob_display['Return_Rate'] = prob_display['Return_Rate'].map('{:.1%}'.format)
            prob_display.columns = ['Product', 'Units Sold', 'Revenue', 'Return Rate']
            st.dataframe(prob_display.reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            st.info("No problem products detected for this country.")

    # Revenue by category bar
    st.divider()
    rev_by_cat = prod.groupby('Category')['Revenue'].sum().sort_values(ascending=True)
    fig10, ax10 = plt.subplots(figsize=(10, 3))
    colors10 = [cat_colors.get(c, ACCENT) for c in rev_by_cat.index]
    bars10 = ax10.barh(rev_by_cat.index, rev_by_cat.values, color=colors10, height=0.4)
    for bar, val in zip(bars10, rev_by_cat.values):
        ax10.text(bar.get_width() + rev_by_cat.max() * 0.01, bar.get_y() + bar.get_height()/2,
                  f'£{val:,.0f}', va='center', fontsize=9, color='#e2e8f0')
    ax10.set_title('Revenue by Product Category', fontsize=10, pad=10)
    ax10.set_xlabel('Total Revenue (£)', fontsize=9)
    ax10.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'£{x/1e3:.0f}K'))
    fig10.tight_layout()
    st.pyplot(fig10)
    plt.close()

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;font-family:DM Mono,monospace;font-size:0.7rem;color:#6b7694;padding:1rem 0'>
    Retail Intelligence Engine · Built by <a href='https://github.com/neeljshah' style='color:#4f8eff;text-decoration:none'>Neel Shah</a>
    · <a href='https://linkedin.com/in/neeljshah22' style='color:#4f8eff;text-decoration:none'>LinkedIn</a>
</div>
""", unsafe_allow_html=True)
