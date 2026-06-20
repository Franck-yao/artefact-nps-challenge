# 📡 NPS Prediction — Artefact CI Challenge
**Senior Data Scientist | Franck Yao | June 2026**

---

## 👤 Author

| | |
|---|---|
| **Name** | Franck Yao |
| **Email** | yaoamemou996@gmail.com |
| **Position applied** | Senior Data Scientist |
| **Company** | Artefact CI |
| **Date** | June 2026 |

---

## 🎯 Business Objective

Build a machine learning system that predicts the NPS category
(**Detractor / Passive / Promoter**) of telecom customers from their
account and behavioral data, so the retention team can :

- Prioritize outreach towards predicted Detractors before they churn
- Identify the main drivers of detraction at individual and segment level
- Simulate the expected NPS of the **silent 85%** who never answer surveys

---

## 📁 Project Structure

```
artefact-nps-challenge/
├── data/
│   ├── raw/                              ← IBM Telco Excel files
│   └── processed/                        ← Generated CSV splits
├── models/
│   ├── lr_pipeline.pkl                   ← Trained model
│   ├── features_final.pkl                ← Feature list
│   └── model_metadata.json               ← Metrics and decisions
├── notebooks/
│   └── NPS_Prediction.ipynb              ← Full pipeline notebook
├── outputs/
│   ├── screenshot_app_prediction.png     ← Streamlit interface screenshot
│   ├── screenshot_app_shap.png           ← Streamlit SHAP screenshot
│   ├── shap_bar_detractor.png            ← Global SHAP importance
│   ├── shap_beeswarm_detractor.png       ← SHAP beeswarm plot
│   ├── shap_waterfall_detractor.png      ← SHAP waterfall individual
│   ├── calibration_lift_detractor.png    ← Calibration + Lift curve
│   └── correlation_heatmap.png           ← Feature correlation heatmap
├── app.py                                ← Streamlit interface
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Franck-yao/artefact-nps-challenge.git
cd artefact-nps-challenge
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download IBM Telco Customer Churn (11.1.3+) from Kaggle :
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the 5 Excel files in `data/raw/`.

### 5. Run the notebook end-to-end
```
notebooks/NPS_Prediction.ipynb
```
This regenerates all processed data, trained model, and SHAP plots.
All results are fully reproducible with `RANDOM_STATE = 42`.

> **Note :** If running locally, change `DATA_PATH` in Cell 04
> from `/content/` to `data/raw/`

### 6. Launch the Streamlit interface
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## 📊 Model Results

| Model | QWK | Macro F1 | Recall Detractor |
|---|---|---|---|
| DummyClassifier | 0.000 | 0.196 | 0.00 |
| mord LogisticAT | 0.293 | 0.347 | 0.11 |
| XGBoost | 0.378 | 0.511 | 0.59 |
| **LogisticRegression** | **0.449** | **0.534** | **0.84** |

**Selected model : LogisticRegression (multinomial, balanced)**

Why LogReg beats XGBoost :
- Small train set (844 samples) favors simpler models
- Best Recall Detractor (84%) — most critical business metric
- Naturally interpretable for fairness auditing

**Lift curve (Detractor class) :**

| % customers contacted | % Detractors captured | Lift vs random |
|---|---|---|
| Top 10% | 27% | 2.7x |
| Top 20% | 57% | 2.8x |
| Top 30% | 77% | 2.6x |
| Top 50% | 95% | 1.9x |

> Recommendation : contact top 30% of customers ranked by P(Detractor)
> to capture 77% of all Detractors at 2.6x efficiency vs random outreach.

---

## 🗂️ NPS Target Construction

| Satisfaction Score | NPS Category | Rationale |
|---|---|---|
| 1 - 2 | Detractor | Dissatisfied customers |
| 3 | Passive | Neutral — aligned with NPS 7-8 definition |
| 4 - 5 | Promoter | Satisfied customers |

Diverges from Artefact baseline (Score ≤3 = Detractor) because
Score 3 represents 37.8% of the base — labeling it as Detractor
produces an unrealistic 58% Detractor majority.

Empirically validated : Churn Score of Score 3 customers = 55.0,
close to satisfied customers (50.2), far from dissatisfied 1-2 (81.9).

---

## 📦 Derived NPS Dataset

The derived NPS dataset is available at :
- `data/processed/silent_base_scored.csv` — 5,987 silent customers
  scored with predicted NPS category and probabilities
- Regenerate deterministically by running the notebook end-to-end
  with `RANDOM_STATE = 42`

---

## 🔍 Key Business Insights

1. **Contract is the #1 predictor** — Month-to-Month : 35.8% Detractor rate vs Two Year : 1.6%
2. **New customers detract most** — Detractors avg 18 months tenure vs 36 for others
3. **Fiber customers are dissatisfied** — Fiber Optic : 31.2% vs DSL : 14.9%
4. **Family = loyalty** — With dependents : 5.2% Detractor rate vs without : 25.0%
5. **Offer E is a red flag** — 39.1% Detractor rate vs Offer A : 4.8%
6. **Senior citizens at risk** — 31.8% Detractor rate vs 18.3% for non-seniors
7. **High charge = poor value perception** — Detractors pay $74.97/month vs $58.42
8. **Referrals signal loyalty** — Detractors avg 0.56 referrals vs 2.37 for Promoters

---

## 🔎 Segment-Level Drivers (Section 4.6)

| Segment | Top Driver | Actionable ? |
|---|---|---|
| Month-to-Month customers | `is_month_to_month` | ✅ Propose annual contract |
| Senior citizens | `is_senior` + `Monthly Charge` | ✅ Dedicated support + simplified offer |
| Fiber Optic customers | `has_fiber` + `Monthly Charge` | ✅ Review pricing vs quality |
| Customers without family | `has_family` = 0 | ⚠️ Non-actionable — demographic |
| New customers (< 12 months) | `Tenure in Months` | ✅ Improve onboarding experience |

---

## ⚖️ Fairness Audit (Section 4.7)

| Group | Recall Detractor | Status |
|---|---|---|
| Overall | 0.841 | ✅ |
| Senior Citizen | 1.000 | ✅ Excellent |
| Non-Senior | 0.806 | ✅ Good |
| Has Family | 0.000 | ⚠️ n=2 — statistically unreliable |
| No Family | 0.881 | ✅ Good |
| Male | 0.875 | ✅ Good |
| Female | 0.800 | ✅ Good |

**Findings to escalate before production :**
- Gender excluded from features (delta = 0.3 pts, no predictive signal)
- Has Family audit inconclusive — needs more labeled data
- ZIP code excluded — residual geographic proxy risk is minimal

---

## ✅ What is implemented vs 🔮 Future work

| Section | Status | Notes |
|---|---|---|
| NPS target construction | ✅ Implemented | Custom mapping justified empirically |
| Data leakage management | ✅ Implemented | 7 columns excluded with proof |
| Feature engineering | ✅ Implemented | 15 features, all justified by EDA |
| Baseline models | ✅ Implemented | DummyClassifier + LogisticRegression |
| Advanced models | ✅ Implemented | XGBoost + mord LogisticAT |
| SHAP interpretability | ✅ Implemented | Global + individual waterfall |
| Segment-level drivers | ✅ Implemented | Section 4.6 — 5 segments analyzed |
| Fairness audit | ✅ Implemented | 3 demographic groups audited |
| Silent base scoring | ✅ Implemented | 5,987 customers scored |
| Streamlit interface | ✅ Implemented | Prediction + SHAP + action |
| Synthetic verbatims (LLM) | 🔮 Future work | Bonus — not implemented within time box |
| Monitoring & retraining | 🔮 Future work | Section 4.9 — out of scope |
| TabPFN foundation model | 🔮 Future work | Bonus — not implemented within time box |

---

## 🤖 AI Tools Used

This project used **Claude (Anthropic)** for :
- Code scaffolding and debugging assistance
- Documentation and markdown drafting
- Explanation of technical concepts during development

All modeling decisions, justifications, results, and final code
are the author's own work. The author remains fully responsible
for the quality and correctness of all outputs.

---

## 📬 Contact & Submission

| | |
|---|---|
| **Author** | Franck Yao |
| **Email** | yaoamemou996@gmail.com |
