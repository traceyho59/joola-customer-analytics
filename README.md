# JOOLA Customer Analytics & Churn Prediction

<p align="center">
  <img src="docs/images/joola-logo.png" alt="JOOLA Logo" width="200"/>
</p>

## 📋 Project Overview

A comprehensive business analytics project for **JOOLA**, a premium pickleball equipment manufacturer. This project analyzes customer retention, profitability patterns, and marketing opportunities using transaction data spanning July 2024 to September 2025.

**Team:** Next Gen Consulting - Columbia University Applied Analytics Program

### 🎯 Objectives

1. **Identify customers most likely to churn** based on historical sales data
2. **Uncover profitability drivers** and revenue concentration patterns
3. **Discover market opportunities** through Google Analytics user behavior analysis
4. **Build predictive models** for customer conversion and churn
5. **Deliver actionable recommendations** for retention campaigns and customer lifetime value optimization

---

## 🏗️ Repository Structure

```
joola-customer-analytics/
│
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── 01_GA_EDA.ipynb          # Google Analytics exploratory analysis
│   ├── 02_Sales_EDA.ipynb       # Sales data cleaning and profitability analysis
│   ├── 03_Google_Search.ipynb   # Search keyword opportunity analysis
│   ├── 04_Churn_XGBoost.ipynb   # XGBoost churn prediction model
│   └── 05_Churn_LogReg.ipynb    # Logistic regression churn model
│
├── src/                          # Source code
│   └── data_processing.py       # Data cleaning and feature engineering utilities
│
├── dashboard/                    # Streamlit dashboard application
│   ├── app.py                   # Main Streamlit application
│   └── requirements.txt         # Dashboard dependencies
│
├── models/                       # Trained model artifacts
│   ├── churn_pipe.pkl           # Serialized churn prediction pipeline
│   └── churn_features.csv       # Feature data for model
│
├── data/                         # Data directory (not tracked in git)
│   └── .gitkeep
│
├── docs/                         # Documentation and reports
│   ├── images/                  # Images for documentation
│   └── methodology.md           # Detailed methodology documentation
│
├── presentations/                # Presentation materials
│   └── JOOLA_Final_Presentation.pdf
│
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## 📊 Key Findings

### 1. Revenue Concentration Risk
- **3 premium paddle SKUs generate ~50% of gross profit**
- Over-dependence on Perseus, Ben Johns, and Vision CGS lines
- Recommendation: Diversify with mid-tier product lines

### 2. Customer Lifetime Value Gap
- **66% of customers are one-time buyers**
- Only 18 VIP customers drive disproportionate revenue
- Converting just 5% of one-time buyers to repeat = **$6M+ additional LTV**

### 3. Untapped Marketing Opportunity
- Court-related keywords: **110K monthly searches, low competition**
- CPC range: $0.86-$3.69 (vs. $0.01-$0.10 for high-competition terms)
- First-mover advantage window: 6-12 months

### 4. Churn Prediction Insights
| Factor | Impact on Churn |
|--------|-----------------|
| Number of discounts used | ↓ Reduces churn |
| Total spending | ↓ Reduces churn |
| Average discount amount | ↑ Increases churn |
| Average spend per order | ↑ Increases churn |

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- pip or conda

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/joola-customer-analytics.git
cd joola-customer-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
cd dashboard
streamlit run app.py
```

---

## 📈 Models

### XGBoost Classifier
- **AUC: 0.76**
- Best for capturing non-linear feature interactions
- Top features: `n_discounts`, `total_spend`, `avg_discount`

### Logistic Regression
- **AUC: 0.57**
- Provides interpretable coefficients
- Key insight: Purchase frequency is strongest retention signal

### Feature Set
| Feature | Description |
|---------|-------------|
| `avg_spend` | Average spending per order |
| `total_spend` | Total lifetime spend |
| `avg_items` | Average items per order |
| `marketing_optin` | Email marketing subscription status |
| `n_discounts` | Number of orders with discount codes |
| `avg_discount` | Average discount amount received |
| `frequency` | Purchase frequency (order count) |
| `avg_gap_days` | Average days between purchases |

---

## 🖥️ Interactive Dashboard

The Streamlit dashboard provides a **"What-If" churn prediction tool** allowing JOOLA to:

1. Adjust customer behavior parameters via sliders
2. See real-time churn probability predictions
3. Understand which factors most influence retention

[![Dashboard Demo](https://img.youtube.com/vi/QlOoYwnm3-w/maxresdefault.jpg)](https://www.youtube.com/watch?v=QlOoYwnm3-w)

---

## 📁 Data Sources

| Dataset | Description | Records |
|---------|-------------|---------|
| Sales Orders | 5 CSV exports merged | 1.9M line items |
| General Ledger SKU | Product cost/margin data | - |
| Google Analytics | User demographics, channels, interests | ~1M users |
| Search Keywords | 5,149 keywords with competition metrics | 5,149 rows |

**Time Period:** July 2024 – September 2025

---

## 👥 Team

| Name | Role |
|------|------|
| Billy Chen | Tech Lead |
| Tracey Ho | Front-End Engineer |
| Ellen Li | Marketing & Communications |
| Sunny Wang | Market Researcher |
| Samantha Yung | Product Manager |
| Victor Zhan | Back-End Engineer |

**Program:** Columbia University, M.S. Applied Analytics

---

## 📝 License

This project is for academic purposes as part of Columbia University's Capstone program.

---

## 🙏 Acknowledgments

- **JOOLA** for providing data and business context
- **Columbia University** Applied Analytics faculty and advisors
- The growing pickleball community 🏓
