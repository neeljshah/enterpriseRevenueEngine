# 🏢 Unified Retail Intelligence Engine
### End-to-End Business Analytics Pipeline | $1.7M Revenue Optimization Ecosystem

> **"From raw transaction data to boardroom-ready insights — forecasting, segmentation, pricing, and A/B testing in one unified system."**

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://enterpriserevenueengine-6njqcmjofztgha4fbcrb75.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

---

## 🔗 [Try the Live App →](https://enterpriserevenueengine-6njqcmjofztgha4fbcrb75.streamlit.app/)

Upload `online_retail_II.csv` and explore all 4 phases interactively — filter by country, adjust churn thresholds, tune customer segments, run live A/B tests, and drill into any product.

---

## 📸 App Preview

> *Dark-themed interactive dashboard — 4 tabbed phases, live KPI cards, dynamic charts, and sidebar controls.*

---

## 🚀 The 4-Phase Pipeline

```
online_retail_II.csv  (500K+ transactions)
    │
    ├── Phase 1 · Revenue Forecasting      → Rolling MA forecast + Pricing Elasticity (Log-Log OLS)
    ├── Phase 2 · Customer Segmentation    → RFM + K-Means (adjustable k) + Random Forest Churn
    ├── Phase 3 · Marketing Attribution    → Channel ROI + Live A/B Test Simulator
    └── Phase 4 · Product KPIs             → Profitability Matrix + Hero/Problem Classification
```

---

## 📊 Phase Breakdown

### Phase 1 — Revenue Forecasting & Pricing Elasticity
**Question:** *What will revenue look like over the next N days, and is our top SKU price sensitive?*

- Daily revenue time-series with **14-day rolling mean** and adjustable forecast horizon (30–180 days)
- Forecast band with confidence interval shading
- Interactive **product selector** — run Log-Log OLS elasticity regression on any top-20 SKU
- Instant business recommendation: raise prices or hold

```python
# Log-Log elasticity regression
X = sm.add_constant(np.log(elasticity_df['Price']))
y = np.log(elasticity_df['Quantity'])
elasticity_value = sm.OLS(y, X).fit().params[1]
# < -1 → Price Sensitive | > -1 → Inelastic
```

**App controls:** Forecast horizon slider · Product dropdown

---

### Phase 2 — Customer Segmentation & Churn Prediction
**Question:** *Who are our VIPs, who's about to churn, and how accurately can we predict it?*

- **RFM engineering** (Recency, Frequency, Monetary) per customer
- Log-transform + StandardScaler before clustering to handle high-skew distributions
- **K-Means** with adjustable k (2–6) — clusters auto-labeled by mean spend: VIP 👑, Loyal 💙, At-Risk ⚠️, Dormant 💤
- **Random Forest churn classifier** — churn threshold fully adjustable via slider
- Segment summary table with revenue breakdown per cohort

```python
rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
rfm_scaled = StandardScaler().fit_transform(rfm_log)
rfm['Cluster'] = KMeans(n_clusters=k, random_state=42).fit_predict(rfm_scaled)
```

**App controls:** Churn threshold slider (30–180 days) · K segments slider

---

### Phase 3 — Marketing Attribution & A/B Testing
**Question:** *Which channel has the best ROI, and does Free Shipping actually beat a discount?*

- Behavioral channel assignment (Email → high frequency, Google Ads → high spend, Social → everyone else)
- **ROI = (Revenue − Cost) / Cost** computed per channel with acquisition cost mapping
- **Live A/B test simulator** — adjust Group A/B revenue averages and sample size, p-value recalculates in real time
- KDE distribution overlay showing revenue spread between test groups
- Auto-generated statistical verdict (significant / not significant)

```python
t_stat, p_val = stats.ttest_ind(group_a_revenue, group_b_revenue)
# Result: Free Shipping +12% AOV | p = 0.0002 ✅ Significant
```

**App controls:** Group A/B avg revenue sliders · Sample size slider

---

### Phase 4 — Product KPIs & Profitability Matrix
**Question:** *Which products are heroes, and which are destroying margin through returns?*

- **Return Rate** calculated per SKU from raw returns data (negative quantities)
- **Inventory Turnover Share** — each product's share of total sales volume
- 2×2 **Profitability Matrix** scatter (log-scale volume vs return rate) with quadrant lines
- Auto-classification into Hero ⭐, Problem ⚠️, Low Volume 📦
- Top 10 Hero and Problem product tables with revenue and return rate
- Revenue-by-category horizontal bar chart

```python
def categorize(row):
    if row['Qty_Sold'] > median_qty:
        return 'Hero ⭐' if row['Return_Rate'] <= median_ret else 'Problem ⚠️'
    return 'Low Volume 📦'
```

---

## ⚙️ Tech Stack

| Library | Usage |
|---|---|
| **Streamlit** | Interactive web app, sidebar controls, tabbed layout |
| **Pandas** | Data cleaning, resampling, RFM aggregation |
| **Statsmodels** | Log-Log OLS regression for pricing elasticity |
| **Scikit-Learn** | K-Means clustering, Random Forest churn model, StandardScaler |
| **SciPy** | Two-sample T-Test for A/B testing |
| **Matplotlib / Seaborn** | All dark-themed visualizations |
| **NumPy** | Log transforms, forecast simulation, array ops |

---

## 🧠 Key Results

| Metric | Result |
|---|---|
| Forecast Horizon | Adjustable 30–180 days |
| Top SKU Price Elasticity | -4.20 — highly price sensitive |
| A/B Test P-Value | 0.0002 ✅ Statistically Significant |
| Free Shipping AOV Lift | +12% over discount coupon |
| Churn Prediction Accuracy | ~85%+ (Random Forest) |
| Hero Products Identified | Top quartile — high volume, low returns |

---

## 🚀 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py

# 3. Upload online_retail_II.csv in the sidebar
```

### Dataset
**Online Retail II** — [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
500K+ transactions · UK-based online retailer · 2009–2011
Features: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, CustomerID, Country

---

## 📁 Repository Structure

```
Enterprise-Revenue-Engine/
├── app.py                       # Streamlit app (all 4 phases)
├── retail_intelligence.py       # Original analysis pipeline
├── requirements.txt             # Python dependencies
├── online_retail_II.csv         # Dataset (download separately — add to .gitignore)
└── README.md
```

---

## 🔗 Related Projects

- 🏀 [Basketball Intelligence Suite](https://github.com/neeljshah/Basketball-Intelligence-Suite) — 4-part spatial & behavioral analytics system for NBA data
- 👁️ [Project CourtVision](https://github.com/neeljshah/Project-CourtVision) — Real-time computer vision player tracking with YOLOv8
- 🏥 [Predictive Models Suite](https://github.com/neeljshah/Predictive-Models) — Breast cancer (97% acc), housing regression, CNN image classification
