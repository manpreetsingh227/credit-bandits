# Bandit-based Approaches for Credit Risk Modeling under Drift and Selective Labels

- **Author:** Manpreet Singh  
- **Date:** 29/08/2025  

---

## 📝 Overview
This repository contains the full implementation accompanying my MSc thesis:  
**Bandit-Based Approaches to Credit Risk Modelling under Drift and Selective Labels**.

The project investigates how credit scoring models can remain robust when:  
- The relationship between features and repayment outcomes **changes over time** (*concept drift*), and  
- Outcomes are **selectively observed only for approved loans** (*selective labels*).  

It evaluates three adaptive learning strategies — **Online Gradient Descent (OGD)**, **Cumulative Refit (CF)**, and **Rolling Refit (RF)** — under both **full feedback (oracle)** and **partial feedback (bandit)** regimes, with and without **ε-greedy exploration**.
---

## 🌟 Key Contributions
- Demonstrates empirically how standard online learning deteriorates under selective feedback.  
- Implements a one-armed contextual bandit framework for credit approval.  
- Evaluates exploration strategies to mitigate selection bias under drift.  
- Provides a reproducible experimental pipeline and analysis using real-world credit risk data.    

---
## 📂 Repository Structure

```text
msc-project/
├── src/
│   ├── merged.py             
│   ├── features.py         
│   ├── CF.py           
│   ├── OGD.py            
│   ├── RF.py             
│
├── jobs/                     
│   ├── merged.pbs
│   ├── features.pbs
│   ├── cf.pbs
│   ├── ogd.pbs
│   ├── rf.pbs
│
├── notebooks/                
│   └── analysis.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```
---


## Workflow

1. **Preprocessing**:  
   Run `src/merged.py` to generate `notebooks/oldmerged.parquet`.  

2. **Feature selection**:  
   Run `src/features.py` to train CatBoost and compute feature importances.  
   - Output: `results/importance.csv`.  
   - Note: Algorithms use a hardcoded top-100 feature list; this file is for verification.  

3. **Algorithms**:  
   Run experiments using:  
   - `src/CF.py` (Cumulative refit)  
   - `src/OGD.py` (Online Gradient Descent)  
   - `src/RF.py` (Rolling refit)  
   Each supports modes (`oracle`, `bandit`, `epsilon_greedy`) and outputs CSV result files.  

4. **Analysis**:  
   Use `notebooks/analysis.ipynb` to plot metrics, compare algorithms, and generate figures for the thesis.  

---

## Dependencies

- pandas  
- numpy  
- scikit-learn  
- catboost  
- pyarrow  
- joblib  
- jupyter *(optional, for notebooks)*  

Dependencies are listed in `requirements.txt`.  

---

## HPC Usage

Job submission scripts are provided in the `jobs/` directory.  
Each script corresponds to one Python script in `src/`.  

Examples:  
- `jobs/merged.pbs` → runs `src/merged.py`  
- `jobs/features.pbs` → runs `src/features.py`  
- `jobs/cf.pbs` → runs `src/CF.py`  
- `jobs/ogd.pbs` → runs `src/OGD.py`  
- `jobs/rf.pbs` → runs `src/RF.py`  

Submit jobs with:  
```bash
qsub jobs/<script>.pbs     



