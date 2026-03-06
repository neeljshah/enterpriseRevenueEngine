# 🏢 Unified Retail Intelligence Engine
### End-to-End Business Analytics Pipeline | $1.7M Revenue Optimization Ecosystem

> **"From raw transaction data to boardroom-ready insights — forecasting, segmentation, pricing, and A/B testing in one unified system."**

---

## 📸 Project Overview

This project is a **4-phase business intelligence pipeline** built on a real-world UK retail dataset (500K+ transactions). Each phase answers a critical business question using production-grade data science techniques — from predicting next quarter's revenue to identifying which marketing channel deserves more budget.

---

## 🚀 The 4-Phase Pipeline

```
Raw Data (online_retail_II.csv)
    │
    ├── Phase 1: Revenue Forecasting       → Prophet 90-day demand model
    ├── Phase 2: Customer Segmentation     → RFM + K-Means + Churn Prediction
    ├── Phase 3: Marketing Attribution     → Channel ROI + A/B Testing
    └── Phase 4: Product KPIs              → Profitability Matrix + Return Rate
```

---

## 📊 Phase Breakdown

### Phase 1 — Revenue Forecasting
**Question:** *What will revenue look like over the next 90 days?*

- Resampled daily UK revenue and fed it into **Facebook Prophet** with yearly seasonality
- Forecasted 90 days ahead with confidence intervals
- Extracted **pricing elasticity** on the top-selling SKU using **Log-Log OLS Regression**
- Elasticity coefficient interpretation: < -1 = price sensitive (don't raise prices), > -1 = inelastic (safe to raise)

```python
model = Prophet(yearly_seasonality=True)
model.fit(daily_revenue)
forecast = model.predict(future)  # 90-day horizon

# Log-Log elasticity regression
X = sm.add_constant(np.log(elasticity_df['Price']))
y = np.log(elasticity_df['Quantity'])
elasticity_value = sm.OLS(y, X).fit().params[1]
```

**Output:**
- 📈 90-day revenue forecast chart with uncertainty bands
- 💰 Elasticity score + business recommendation per SKU

---

### Phase 2 — Customer Segmentation & Churn Prediction
**Question:** *Who are our best customers, who's about to leave, and how accurately can we predict it?*

- Built **RFM features** (Recency, Frequency, Monetary) per customer
- Applied **log-transform + StandardScaler** to normalize high-skew distributions before clustering
- Ran **K-Means (k=3)** to segment customers into behavioral clusters (VIP, At-Risk, Low-Engagement)
- Built a **Random Forest churn classifier** using Frequency + Monetary as features
- Defined churn as Recency > 90 days

```python
rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
rfm_scaled = scaler.fit_transform(rfm_log)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
```

**Output:**
- 🧩 Customer segment scatter plot (Recency vs Monetary, log-scale)
- 🎯 Churn prediction accuracy score
- 📊 VIP customer count isolated from smallest cluster

---

### Phase 3 — Marketing Attribution & A/B Testing
**Question:** *Which marketing channel delivers the highest ROI, and does Free Shipping beat a 15% Discount?*

- Assigned simulated marketing channels (Email, Google Ads, Social Media) based on behavioral proxies from RFM
- Mapped **channel acquisition costs** and computed **ROI = (Revenue - Cost) / Cost** per channel
- Ran a **Two-Sample T-Test** (A/B test) comparing Free Shipping vs 15% Discount coupon on 1,000 at-risk customers
- Statistical significance threshold: p < 0.05

```python
t_stat, p_val = stats.ttest_ind(free_shipping_revenue, discount_coupon_revenue)
# Free Shipping: ~$150 avg revenue
# Discount Coupon: ~$135 avg revenue
# P-Value: 0.0002 → Statistically Significant
```

**Output:**
- 📊 Channel ROI bar chart (coolwarm palette)
- ✅ A/B test winner: **Free Shipping** generated ~12% higher AOV (p = 0.0002)

---

### Phase 4 — Operational KPIs & Product Profitability Matrix
**Question:** *Which products are heroes, and which are quietly destroying margin through returns?*

- Calculated **Return Rate** per product by comparing negative vs positive quantity rows
- Computed **Inventory Turnover Share** as each product's share of total sales volume
- Built a **Product Profitability Matrix** categorizing every SKU into:
  - ⭐ **Star (Hero):** High volume + low return rate
  - ⚠️ **Problem:** High volume + high return rate
  - 📦 **Low Volume:** Below-median sales
- Calculated **Average Order Value (AOV)** across all UK invoices

```python
def categorize_product(row):
    if row['Quantity'] > product_stats['Quantity'].median():
        if row['Return_Rate'] < product_stats['Return_Rate'].median():
            return 'Star (Hero)'
        else:
            return 'Problem (High Returns)'
    else:
        return 'Low Volume'
```

**Output:**
- 🔵 Profitability Matrix scatter (log-scale volume vs return rate)
- 🏆 Top 5 Hero Products ranked by volume
- 💳 AOV summary + overall return rate

---

## ⚙️ Tech Stack

| Library | Usage |
|---|---|
| **Pandas** | Data cleaning, resampling, RFM aggregation |
| **Prophet** | 90-day time-series revenue forecasting |
| **Statsmodels** | Log-Log OLS regression for pricing elasticity |
| **Scikit-Learn** | K-Means clustering, Random Forest churn model, StandardScaler |
| **SciPy** | Two-sample T-Test for A/B testing |
| **Matplotlib / Seaborn** | All visualizations |
| **NumPy** | Log transforms, array operations |

---

## 🧠 Key Results

| Metric | Result |
|---|---|
| Forecast Horizon | 90 days |
| Top SKU Price Elasticity | -4.20 (highly price sensitive) |
| A/B Test P-Value | 0.0002 ✅ Significant |
| Free Shipping AOV Lift | ~12% over discount coupon |
| Churn Prediction Accuracy | ~85%+ (Random Forest) |
| VIP Customers Identified | Smallest K-Means cluster |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas prophet statsmodels scikit-learn scipy matplotlib seaborn numpy
```

### Run the Pipeline
```bash
# 1. Place your dataset in the project root
#    Dataset: online_retail_II.csv (UCI Machine Learning Repository)

# 2. Run the full pipeline
python retail_intelligence.py
```

### Dataset
This project uses the **Online Retail II** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II).
- 500K+ transactions from a UK-based online retailer (2009–2011)
- Features: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, Price, CustomerID, Country

---

## 📁 Repository Structure

```
Enterprise-Revenue-Engine/
├── retail_intelligence.py       # Full 4-phase pipeline
├── online_retail_II.csv         # Dataset (download separately)
├── outputs/
│   ├── forecast_chart.png
│   ├── customer_segments.png
│   ├── channel_roi.png
│   └── profitability_matrix.png
└── README.md
```

---

## 🔗 Related Projects

- 🏀 [Basketball Intelligence Suite](https://github.com/neeljshah/Basketball-Intelligence-Suite) — 4-part spatial & behavioral analytics system for NBA data
- 👁️ [Project CourtVision](https://github.com/neeljshah/Project-CourtVision) — Real-time computer vision player tracking with YOLOv8
