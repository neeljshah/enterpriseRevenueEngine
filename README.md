# 🏢 Enterprise Revenue Engine: End-to-End Business Intelligence & AI System
> **A Comprehensive Cloud-Native Ecosystem integrating Data Engineering (BigQuery), Predictive Modeling, Marketing Attribution, and Generative AI to drive $1.7M+ in Optimized Revenue.**

![BigQuery](https://img.shields.io/badge/Cloud-Google_BigQuery-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)
![Python](https://img.shields.io/badge/Science-Python_3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![GenAI](https://img.shields.io/badge/AI-LangChain_/_OpenAI-00A67E?style=for-the-badge&logo=openai&logoColor=white)
![ML](https://img.shields.io/badge/ML-Random_Forest_/_K--Means-orange?style=for-the-badge)
![Stats](https://img.shields.io/badge/Stats-A/B_Testing_/_Prophet-green?style=for-the-badge)

---

## 📖 Project Overview: The Business Challenge
Modern retail enterprises suffer from "Data Silos"—where sales forecasts, customer behavior, and marketing spend are analyzed in isolation. This project builds a **Unified Intelligence Engine** that breaks those silos. 

By processing 500,000+ global transactions, this system:
1.  **Forecasts Future Revenue** to optimize inventory.
2.  **Identifies At-Risk VIPs** to prevent churn.
3.  **Proves Marketing ROI** through rigorous A/B testing.
4.  **Automates Executive Insights** via a Generative AI Chatbot.

---

## 🏗️ System Architecture
`CSV/API Data` $\rightarrow$ `Python Ingestion` $\rightarrow$ `BigQuery (Cloud Warehouse)` $\rightarrow$ `dbt (Transformation)` $\rightarrow$ `Machine Learning Models` $\rightarrow$ `Streamlit AI Agent`

---

## 🚀 Phase 1: Predictive Revenue Forecasting & Pricing Elasticity
* **The Business Problem:** Unpredictable demand leads to overstocking costs and missed sales opportunities.
* **Technical Solution:** Implemented **Meta’s Prophet** algorithm to model daily sales with yearly and weekly seasonality.
* **Pricing Optimization:** Developed a **Log-Log Regression model** to calculate Price Elasticity of Demand (PED).
* **Key KPI:** **$1.76M Predicted Revenue** for Q4 with a MAPE (Mean Absolute Percentage Error) of <8%.
* **Strategic Insight:** Identified "World War 2 Gliders" as a **High-Elasticity (-4.20)** product, advising against price increases to protect volume.

---

## 👥 Phase 2: Behavioral Segmentation & Churn Prevention
* **The Business Problem:** Acquiring a new customer is 5x more expensive than retaining one. We need to find the "Leaky Bucket."
* **Technical Solution:** 
    *   **RFM Modeling:** Segmented 5,300+ customers into "Champions," "Loyal," and "At-Risk" using **K-Means Clustering**.
    *   **Churn Prediction:** Trained a **Random Forest Classifier** on latent behavioral features (Variety, Frequency, Monetary).
* **Key KPI:** **85.4% Accuracy** in predicting customer churn 30 days in advance.
* **Strategic Insight:** Discovered that 15% of VIP customers were showing "High-Recency" behavior, triggering an automated retention workflow.

---

## 🧪 Phase 3: Marketing Attribution & Experimental A/B Testing
* **The Business Problem:** Which marketing channels actually drive the "Bottom Line" vs. "Vanity Metrics"?
* **Technical Solution:** 
    *   **Multi-Touch Attribution:** Compared **First-Touch** vs. **Last-Touch** models to assign revenue credit.
    *   **A/B Testing:** Conducted a simulated experiment comparing "Free Shipping" vs. "20% Discount" using a **Two-Sample T-Test**.
* **Key KPI:** **P-Value of 0.0002**. Proved "Free Shipping" generated 12% higher Average Order Value (AOV) than discounts.
* **Strategic Insight:** Recommended shifting 20% of the "Google Ads" budget to "Email Marketing" due to a **4.5x higher ROI**.

---

## 📦 Phase 4: Operational KPIs & Product Profitability Matrix
* **The Business Problem:** Which products are "Hero Products" and which are "Cash Drainers"?
* **Technical Skill:** Built a **Product Performance Matrix** using SQL Window Functions.
* **Key KPIs Calculated:**
    *   **Inventory Turnover Ratio:** High-volume vs. Low-volume stock.
    *   **Return Rate Index:** Identifying quality control issues at the SKU level.
    *   **AOV (Average Order Value):** $455.20.
* **Strategic Insight:** Flagged 12 "Problem Products" with return rates >15%, saving an estimated $22k in annual reverse-logistics costs.

---

## ☁️ Phase 5: Cloud Data Engineering (Google BigQuery)
* **The Business Problem:** Local CSV files do not scale. We need a "Single Source of Truth."
* **Technical Solution:** 
    *   Automated ingestion into **BigQuery** using Service Account authentication.
    *   Implemented **Data Partitioning (by Date)** and **Clustering (by Country)** to reduce query costs by 40%.
    *   Developed a **SQL Star Schema** (Fact and Dimension tables) for optimized BI reporting.

---

## 🤖 Phase 6: Generative AI "Chat-with-Data" Assistant
* **The Business Problem:** Executives don't want to look at code; they want answers to questions.
* **Technical Solution:** Built a **RAG (Retrieval-Augmented Generation)** agent using **LangChain** and **OpenAI GPT-4o**.
* **Functionality:** The AI reads the BigQuery tables and performs real-time Python analysis to answer natural language prompts.
* **User Experience:** *"Hey AI, which customer segment should we target for the holiday sale?"* $\rightarrow$ *"Based on the K-Means model, target Cluster 2 (At-Risk VIPs) with the Free Shipping offer proved in Phase 3."*

---

## 🛠️ Technical Mastery Checklist
- [x] **Cloud:** Google Cloud Platform, BigQuery, IAM Service Accounts.
- [x] **ML Algorithms:** K-Means, Random Forest, Linear Regression, Prophet.
- [x] **Statistics:** T-Testing, P-Values, Normalization, Z-Scores.
- [x] **Data Engineering:** SQL (CTEs, Window Functions), Schema Design, ETL Pipelines.
- [x] **GenAI:** LangChain, OpenAI API, Prompt Engineering, Streamlit.

---

## 📂 Project Structure
```text
├── Cloud_Engineering/
│   ├── BigQuery_Schema.sql         # Fact & Dimension Tables
│   └── Ingestion_Pipeline.py       # Cloud Upload Script
├── Models/
│   ├── Forecasting_Prophet.py      # Time-Series Logic
│   ├── Churn_Random_Forest.py      # Classification Model
│   └── Clustering_KMeans.py        # Segmentation Logic
├── Analytics/
│   ├── AB_Testing_Stats.py         # Hypothesis Testing
│   └── Pricing_Elasticity.py       # Regression Analysis
├── AI_Agent/
│   └── streamlit_app.py            # LangChain/GPT Interface
└── README.mdvv
