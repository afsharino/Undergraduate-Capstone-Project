# Undergraduate Capstone Project: On-Chain Indicator Fusion for Bitcoin Trading

![Bitcoin Logo](https://img.shields.io/badge/Bitcoin-Trading%20Strategy-orange) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyGAD](https://img.shields.io/badge/PyGAD-Genetic%20Algorithm-green) ![CryptoQuant](https://img.shields.io/badge/Data-CryptoQuant%20%26%20Fear%26Greed-red)

## 📌 Project Overview

This repository contains the **Undergraduate Capstone Project** focused on developing a **data-driven Bitcoin trading framework** that outperforms traditional strategies such as **Buy-and-Hold** and **Fear & Greed Index-based allocation**.

The core idea:  
Instead of relying on a single sentiment metric (like the Fear & Greed Index) or passive holding, we **combine 170 on-chain and market indicators from CryptoQuant** to create a **composite indicator** using **linear models and neural networks**, optimized via **genetic algorithms**. This new indicator dynamically adjusts portfolio exposure (Bitcoin vs. cash) to maximize returns and minimize drawdowns — especially during sharp market downturns.

---

## 🎯 Motivation

- **Buy-and-Hold** suffers from **high drawdowns** in bear markets.
- **Fear & Greed Index** uses sentiment but ignores on-chain fundamentals.
- CryptoQuant provides **170+ institutional-grade on-chain metrics** across 16 categories.
- **Can we combine these signals intelligently to create a better trading signal?**

**Yes — and this project proves it.**

We designed a **rebalancing strategy** where:
> **Exposure to Bitcoin = f(New Indicator Value)**  
> *(e.g., 60 → invest 60% of portfolio in BTC, 40% in cash)*

The goal: **Outperform Buy-and-Hold and Fear & Greed**, especially in downtrends.

---

## 🛠 Data Sources

| Source | Description | # Indicators |
|-------|-------------|---------------|
| **[CryptoQuant](https://cryptoquant.com/asset/btc/summary)** | Institutional on-chain & market data | **170** |
| **[Alternative.me Fear & Greed](https://alternative.me/crypto/fear-and-greed-index/)** | Daily sentiment index (0–100) | 1 |

### CryptoQuant Indicator Categories (16 Total)
1. Exchange Flows  
2. Flow Indicator  
3. Market Indicator  
4. Network Indicator  
5. Miner Flows  
6. Derivatives  
7. Fund Data  
8. Market Data  
9. Addresses  
10. Fees And Revenue  
11. Network Stats  
12. Supply  
13. Transactions  
14. Inter Entity Flows  
15. Bank Flows  
16. Research  

> **All data is daily resolution.** Final aligned period: **`2021-03-12` to `2024-03-08`**

---

## 🔄 Data Preprocessing Pipeline

Due to missing values and misaligned date ranges, extensive preprocessing was required:

### 1. **Date Integrity Check**
- Verified all indicators within a category share identical dates.

### 2. **Shape Integrity Check**
- Ensured equal number of rows per indicator in each category.

### 3. **Period Alignment**
- Found common overlapping timeframe: `2021-03-12` → `2024-03-08`
- Removed indicators with insufficient overlap.

### 4. **Consolidation per Category**
- Merged indicators within each category into one DataFrame.

### 5. **Fear & Greed Integration**
- Aligned and merged with main dataset.

### 6. **Final Dataset**
- **171 features**: `170 indicators + bias column`
- **Price & Fear-and-Greed used only for trading logic**, **not model inputs**

---

## 🧠 Model Design: Creating the New Indicator

We explored **two approaches** to fuse the 170 indicators:

| Approach | Description |
|--------|-----------|
| **Linear Model** | `α₁·x₁ + α₂·x₂ + ... + α₁₇₀·x₁₇₀` |
| **MLP (Forward Pass Only)** | Matrix multiplication: `W₂·σ(W₁·X)` (no backprop) |

> **Optimization**: Genetic Algorithm [](https://pygad.readthedocs.io/)  
> **Objective**: Maximize **portfolio profit** under rebalancing strategy

### Training Strategy: Sliding Window
- **Window Size**: 260 days (~9 months)
- **Step Size**: Configurable (e.g., 1 day)
- Model retrained every step → adapts to changing market regimes

---

## ⚖ Trading Strategies Compared

| Strategy | Logic |
|--------|-------|
| **Buy-and-Hold** | 100% BTC at start → hold |
| **Fear & Greed** | Exposure = F&G Index value (0–100) |
| **New Indicator (Linear / MLP)** | Exposure = `clamp(new_indicator, 0, 100)` |

### Rebalancing Example
Day 1: Balance = $10,000 | BTC Price = $50,000 | Indicator = 80
→ Buy $8,000 BTC (0.16 BTC) | Keep $2,000 cash

Day 2: BTC Price = $51,000 | Indicator = 82
→ Total value = $10,160
→ Target: 82% in BTC → $8,331 BTC
→ Buy more BTC to reach the target

---

## 📊 Evaluation Metrics

The performance of each trading approach (**Buy-and-Hold**, **Fear & Greed**, and **New Indicator**) was evaluated on the **test dataset** using the following metrics:

### 1. **Profitability (Profit %)**
Total percentage profit at the end of each **market phase**:

\[
\text{Profit (\%)} = \frac{\text{Total Balance after trading} - \text{Initial Balance}}{\text{Initial Balance}} \times 100
\]

- Evaluated **separately** for:
  - **Uptrend**
  - **Sideways**
  - **Downtrend**

### 2. **Overall Profit (%)**
Cumulative profit across **all phases** in the test period.

### 3. **Absolute Drawdown (%)**
Largest drop from **initial balance** during the **entire test period**:

\[
\text{Absolute Drawdown (\%)} = \frac{\text{Initial Balance} - \text{Lowest Value During the Period}}{\text{Initial Balance}} \times 100
\]

> Unlike Maximum Drawdown (peak-to-trough), this measures **worst-case loss from starting capital** — critical for long-term risk assessment.

---

### 📋 Evaluation Process

| Split | Date Range | Purpose |
|------|------------|--------|
| **Training** | `2021-01-01` → `2022-06-30` | Optimize weights for New Indicator |
| **Test** | `2022-07-01` → `2024-03-08` | Evaluate all strategies |

- **Profitability**: Measured **per market phase**
- **Absolute Drawdown**: Measured over **entire test period**

---

### 📈 Results on Test Set

> **Note for Dr. Fazl**: Neural network results vary due to genetic algorithm randomness. **Linear model consistently outperforms MLP.**

#### Fixed Training (260 days)

| Strategy | Uptrend Profit (%) | Sideways Profit (%) | Downtrend Profit (%) | **Overall Profit (%)** | **Absolute Drawdown (%)** |
|--------|--------------------|---------------------|----------------------|-------------------------|----------------------------|
| **Buy-and-Hold** | 149.01 | 21.44 | -62.48 | **22.00%** | **72.69%** |
| **Fear & Greed** | 84.59 | 3.66 | -22.15 | **56.00%** | **30.57%** |
| **Linear Model** | 133.59 | 20.27 | -17.22 | **154.00%** | **31.10%** |
| **Neural Net (MLP)** | 100.01 | 22.45 | -26.19 | — | — |

---

#### Sliding Window (Linear Model)

| Window Size | Step Size | Uptrend Profit (%) | Sideways Profit (%) | Downtrend Profit (%) | **Overall Profit (%)** | **Absolute Drawdown (%)** |
|-----------|-----------|--------------------|---------------------|----------------------|-------------------------|----------------------------|
| 260 | 20 | 52.80 | 3.09 | -26.66 | **19.00%** | **42.04%** |
| 260 | 90 | 58.81 | 14.97 | -37.23 | **20.00%** | **41.05%** |
| 260 | 120 | 90.00 | 0.97 | -27.36 | **41.00%** | **43.61%** |
| 260 | 150 | 101.00 | 13.61 | -31.38 | **19.00%** | **40.87%** |

---

### 🏆 Key Insights

- **Linear Model (Fixed)** achieves **highest overall profit (154%)**
- **Sliding Window** reduces drawdown vs Buy-and-Hold but **underperforms fixed training** in bull runs
- **Fear & Greed** is robust but **misses on-chain signals**
- **MLP unstable** due to GA — **not recommended for production**
- **Best for risk-averse investors**: **Linear Model (Fixed Training)**

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/afsharino/Undergraduate-Capstone-Project.git
cd Undergraduate-Capstone-Project

# 2. Create and activate conda environment
conda env create -f environment.yml
conda activate capstone-btc

# 3. Launch the main notebook
jupyter notebook main.ipynb
```

---

## 🏆 Conclusion

This project **successfully demonstrates** that:

> **A genetically optimized fusion of 170 on-chain indicators can create a superior trading signal** — **outperforming both Buy-and-Hold and Fear & Greed Index strategies**.

### Key Achievements:
- **Highest overall profit**: `154%` (Linear Model, Fixed Training)  
- **Significant drawdown reduction** vs. Buy-and-Hold  
- **Robust performance** across uptrend, sideways, and downtrend phases  
- **Adaptive framework** via sliding window optimization  

It provides a **practical, data-driven, and adaptive trading framework** for **passive crypto investors** seeking **higher returns with lower risk** — leveraging **institutional-grade on-chain data** and **evolutionary optimization**.

---

**Future Work**:
- Test on live data
- Extend to multi-asset portfolios
- Explore ensemble of linear + neural models

