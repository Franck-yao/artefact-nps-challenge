# ============================================================
# app.py — NPS Prediction Interface
# Artefact CI Challenge — Senior Data Scientist
# Author : Franck Yao | June 2026
# Run : streamlit run app.py
# ============================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import json


# ── Constants ──────────────────────────────────────────────
NPS_ORDER  = ['Detractor', 'Passive', 'Promoter']
NPS_EMOJI  = {'Detractor': '🔴', 'Passive': '🟡', 'Promoter': '🟢'}
NPS_COLORS = {'Detractor': '#E24B4A', 'Passive': '#EF9F27',
              'Promoter': '#639922'}


# ══════════════════════════════════════════════════════════
# CLASS 1 — ModelLoader
# Responsible for loading and caching model artifacts
# ══════════════════════════════════════════════════════════
class ModelLoader:
    def __init__(self,
                 pipeline_path  : str = 'models/lr_pipeline.pkl',
                 features_path  : str = 'models/features_final.pkl',
                 metadata_path  : str = 'models/model_metadata.json'):
        self.pipeline_path = pipeline_path
        self.features_path = features_path
        self.metadata_path = metadata_path

    @st.cache_resource
    def load(_self):
        pipeline = joblib.load(_self.pipeline_path)
        features = joblib.load(_self.features_path)
        with open(_self.metadata_path, 'r') as f:
            metadata = json.load(f)
        return pipeline, features, metadata


# ══════════════════════════════════════════════════════════
# CLASS 2 — CustomerInput
# Responsible for building the feature vector from sidebar
# ══════════════════════════════════════════════════════════
class CustomerInput:
    def __init__(self, features: list):
        self.features = features
        self._collect()

    def _collect(self):
        """Collect all inputs from the Streamlit sidebar."""
        st.sidebar.header("🧑 Customer Profile")
        st.sidebar.markdown("Enter customer attributes below.")

        # Numerical inputs
        self.tenure         = st.sidebar.slider(
            "Tenure (months)", 1, 72, 12)
        self.monthly_charge = st.sidebar.slider(
            "Monthly Charge ($)", 20, 120, 65)
        self.age            = st.sidebar.slider(
            "Age", 18, 90, 45)
        self.avg_gb         = st.sidebar.slider(
            "Avg Monthly GB Download", 0, 60, 15)
        self.ld_charges     = st.sidebar.slider(
            "Total Long Distance Charges ($)", 0, 4000, 500)
        self.num_referrals  = st.sidebar.number_input(
            "Number of Referrals", 0, 15, 0)

        st.sidebar.divider()

        # Categorical inputs
        self.contract = st.sidebar.selectbox(
            "Contract Type",
            ["Month-to-Month", "One Year", "Two Year"])
        self.internet = st.sidebar.selectbox(
            "Internet Type",
            ["Fiber Optic", "DSL", "Cable", "No Internet"])
        self.support  = st.sidebar.selectbox(
            "Premium Tech Support", ["No", "Yes"])
        self.referred = st.sidebar.selectbox(
            "Has referred at least once ?", ["No", "Yes"])
        self.family   = st.sidebar.selectbox(
            "Has dependents ?", ["No", "Yes"])
        self.senior   = st.sidebar.selectbox(
            "Senior Citizen ?", ["No", "Yes"])
        self.offer    = st.sidebar.selectbox(
            "Received Offer E ?", ["No", "Yes"])

    def build(self) -> pd.DataFrame:
        """Build and return the feature vector as a DataFrame."""
        is_new = 1 if self.tenure <= 12 else 0
        nb_svc = sum([
            1 if self.support  == "Yes" else 0,
            1 if self.referred == "Yes" else 0,
        ])
        cps = self.monthly_charge / (nb_svc + 1)

        return pd.DataFrame([{
            'Tenure in Months'            : self.tenure,
            'is_month_to_month'           : 1 if self.contract == "Month-to-Month" else 0,
            'is_new_customer'             : is_new,
            'has_fiber'                   : 1 if self.internet == "Fiber Optic" else 0,
            'has_premium_support'         : 1 if self.support  == "Yes" else 0,
            'Monthly Charge'              : self.monthly_charge,
            'Total Long Distance Charges' : self.ld_charges,
            'charge_per_service'          : cps,
            'Number of Referrals'         : self.num_referrals,
            'has_referred'                : 1 if self.referred == "Yes" else 0,
            'is_offer_E'                  : 1 if self.offer    == "Yes" else 0,
            'Age'                         : self.age,
            'has_family'                  : 1 if self.family   == "Yes" else 0,
            'is_senior'                   : 1 if self.senior   == "Yes" else 0,
            'Avg Monthly GB Download'     : self.avg_gb,
        }])


# ══════════════════════════════════════════════════════════
# CLASS 3 — NPSPredictor
# Responsible for prediction and SHAP explanation
# ══════════════════════════════════════════════════════════
class NPSPredictor:
    def __init__(self, pipeline, features: list):
        self.pipeline  = pipeline
        self.features  = features
        self.scaler    = pipeline.named_steps['scaler']
        self.lr_model  = pipeline.named_steps['clf']

    def predict(self, X: pd.DataFrame) -> dict:
        """Return prediction label, index and probabilities."""
        proba     = self.pipeline.predict_proba(X)[0]
        pred_idx  = int(np.argmax(proba))
        pred_label = NPS_ORDER[pred_idx]
        return {
            'label'       : pred_label,
            'idx'         : pred_idx,
            'proba'       : proba,
            'emoji'       : NPS_EMOJI[pred_label],
            'color'       : NPS_COLORS[pred_label],
        }

    def explain(self, X: pd.DataFrame, pred_idx: int,
                top_n: int = 5) -> pd.DataFrame:
        """Compute SHAP values and return top N drivers."""
        X_scaled = self.scaler.transform(X)
        X_bg     = self.scaler.transform(
            pd.DataFrame([dict(zip(self.features,
                                   [0] * len(self.features)))])
        )
        explainer = shap.LinearExplainer(
            self.lr_model, X_bg,
            feature_perturbation="interventional"
        )
        shap_vals = explainer.shap_values(X_scaled)

        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            sv = shap_vals[0, :, pred_idx]
        else:
            sv = shap_vals[pred_idx][0]

        return (pd.DataFrame({
            'Feature'    : self.features,
            'SHAP Value' : sv
        }).sort_values('SHAP Value', key=abs, ascending=False)
          .head(top_n))


# ══════════════════════════════════════════════════════════
# CLASS 4 — NPSDashboard
# Responsible for rendering the Streamlit UI
# ══════════════════════════════════════════════════════════
class NPSDashboard:
    def __init__(self, predictor: NPSPredictor, metadata: dict):
        self.predictor = predictor
        self.metadata  = metadata

    def render_header(self):
        st.title("📡 NPS Predictor — Telecom Operator")
        st.markdown(
            "**Retention team interface** — predict customer NPS "
            "and identify the key drivers of detraction."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model",            "LogisticRegression")
        c2.metric("QWK (test)",       str(self.metadata['metrics_test']['QWK']))
        c3.metric("Recall Detractor", f"{self.metadata['metrics_test']['Recall_Detractor']:.0%}")
        c4.metric("Train size",       f"{self.metadata['train_size']} customers")
        st.divider()

    def render_prediction(self, result: dict):
        st.subheader("📊 Prediction Result")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted NPS",
                  f"{result['emoji']} {result['label']}")
        c2.metric("P(Detractor)", f"{result['proba'][0]:.1%}")
        c3.metric("P(Passive)",   f"{result['proba'][1]:.1%}")
        c4.metric("P(Promoter)",  f"{result['proba'][2]:.1%}")

        # Probability bar chart
        prob_df = pd.DataFrame({
            'NPS Class'   : NPS_ORDER,
            'Probability' : result['proba']
        })
        st.bar_chart(
            prob_df.set_index('NPS Class'),
            color=result['color']
        )
        st.divider()

    def render_shap(self, shap_df: pd.DataFrame, pred_label: str):
        st.subheader("🔍 Prediction Drivers (SHAP)")
        st.markdown(
            f"**Top 5 features driving this "
            f"{pred_label} prediction :**"
        )
        for _, row in shap_df.iterrows():
            direction = ("↑ increases detraction risk"
                         if row['SHAP Value'] > 0
                         else "↓ decreases detraction risk")
            icon = "🔴" if row['SHAP Value'] > 0 else "🟢"
            st.markdown(
                f"{icon} **{row['Feature']}** "
                f"({row['SHAP Value']:+.3f}) — {direction}"
            )
        st.divider()

    def render_action(self, pred_label: str):
        st.subheader("💡 Recommended Action")
        if pred_label == 'Detractor':
            st.error(
                "⚠️ **High detraction risk.** "
                "Recommend immediate retention outreach. "
                "Priority action : propose migration to annual "
                "contract with a loyalty discount."
            )
        elif pred_label == 'Passive':
            st.warning(
                "ℹ️ **Passive customer — monitor closely.** "
                "Consider a proactive engagement offer "
                "before satisfaction declines."
            )
        else:
            st.success(
                "✅ **Promoter — customer is satisfied.** "
                "Consider activating referral program "
                "to leverage this customer's loyalty."
            )
        st.divider()

    def render_footer(self):
        st.caption(
            "Model : LogisticRegression (multinomial, balanced) | "
            f"QWK = {self.metadata['metrics_test']['QWK']} | "
            "Recall Detractor = 84% | "
            f"Train : {self.metadata['train_size']} customers | "
            "Artefact CI Challenge — Franck Yao | June 2026"
        )

    def run(self, X_input: pd.DataFrame):
        self.render_header()
        result  = self.predictor.predict(X_input)
        shap_df = self.predictor.explain(X_input, result['idx'])
        self.render_prediction(result)
        self.render_shap(shap_df, result['label'])
        self.render_action(result['label'])
        self.render_footer()


# ══════════════════════════════════════════════════════════
# MAIN — Entry point
# ══════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title = "NPS Predictor — Telecom",
        page_icon  = "📡",
        layout     = "wide"
    )

    # Load artifacts
    loader                      = ModelLoader()
    pipeline, features, metadata = loader.load()

    # Collect customer input
    customer  = CustomerInput(features)
    X_input   = customer.build()

    # Run dashboard
    predictor = NPSPredictor(pipeline, features)
    dashboard = NPSDashboard(predictor, metadata)
    dashboard.run(X_input)


if __name__ == "__main__":
    main()