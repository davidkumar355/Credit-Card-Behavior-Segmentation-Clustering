<div align="center">

# 💳 Credit Card Behavioral Segmentation
*Data-driven customer archetypes for targeted banking strategies.*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Tech Stack](https://img.shields.io/badge/Tech_Stack-Streamlit%20%7C%20Scikit--Learn%20%7C%20Plotly-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Deployed App](https://img.shields.io/badge/Deployed-Streamlit-red.svg)](https://streamlit.io/)

</div>

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Customer Archetypes](#-customer-archetypes)
- [Getting Started / Quickstart](#-getting-started--quickstart)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)

---

## 📖 Overview

This project segments a portfolio of active credit card customers into distinct behavioral archetypes using unsupervised machine learning. By analyzing financial habits—such as spending intensity, payment discipline, and cash dependency—the model provides actionable, targeted marketing and operational strategies for banking institutions. 

The project transitions from an exploratory Jupyter Notebook pipeline into a fully interactive Streamlit web application, allowing stakeholders to visually audit clusters and simulate CRM tagging in real-time.

<div align="center">

---

## ✨ Features

* 🛠️ **Robust Feature Engineering:** Generates domain-specific financial ratios (Spend Intensity, Payment Discipline, Revolving Behavior, Cash Dependency) rather than relying solely on raw data.
* 📈 **Advanced Preprocessing:** Mitigates skewness via 1st/99th percentile Winsorization and log1p scaling, followed by Standard Scaling and Principal Component Analysis (PCA) for dimensionality reduction.
* 🧠 **K-Means Clustering:** Deploys an optimized K-Means algorithm (K=4) validated by Silhouette Scores and inertia metrics.
* 🌐 **Interactive Streamlit Dashboard:** A minimalist dark-mode UI featuring:
  * **Persona Dashboard:** Plotly-powered spider/radar charts and cohort summaries.
  * **Segment Explorer:** 3D scatter visualizations of clustered populations.
  * **Real-Time Tagging Engine:** A CRM simulation tool that categorizes new customers instantly through the deployed mathematical pipeline.
* 🔒 **Production-Ready Architecture:** Decouples model training from web hosting using an `export_models.py` workflow (`.pkl` via `joblib`) to guarantee zero data-drift.

---

## 👥 Customer Archetypes

The algorithm identified four distinct customer segments, each driving a specific business strategy:

1. **Silent Revolvers (Cluster 0):** Moderate spenders carrying month-over-month balances with minimal full-payoffs. *(Strategy: Payoff incentives, balance transfers).*
2. **Cash Dependent (Cluster 1):** High cash-advance usage with poor payment discipline; indicating financial survival mode. *(Strategy: Debt restructuring, cash-out blocks).*
3. **Power Spenders (Cluster 2):** Highly profitable, low-risk users who maximize limits but clear balances regularly. *(Strategy: Retain and upsell premium reward tiers).*
4. **Disciplined Payers (Cluster 3):** Prudent convenience users with minimal balances and perfect payment behavior. *(Strategy: Nudge behavior via limit increases).*

<img width="990" height="705" alt="Screenshot 2026-04-15 201620" src="https://github.com/user-attachments/assets/7da4b383-aed9-4682-8593-fce6fefbd994" />

---

## 🚀 Getting Started / Quickstart

### Prerequisites
Ensure you have the following installed:
* **Python** (v3.8 or higher)
* **Git**

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/credit-card-segmentation.git](https://github.com/yourusername/credit-card-segmentation.git)
   cd credit-card-segmentation
