import streamlit as st
import requests
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Learning Recommender", page_icon="🎓", layout="centered")

st.title("🎓 Personalized Learning Recommender")
st.markdown("**SVD-based Collaborative Filtering | PRO008**")
st.divider()

# Model load for SHAP
@st.cache_resource
def load_model():
    with open('svd_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
rec_df = model['rec_df']
item_avg = model['item_avg']
global_avg = model['global_avg']
user_item = model['user_item']

# ── Section 1: Recommendations ──
st.subheader("📚 Get Course Recommendations")
user_id = st.number_input("Enter Student ID (1-943)", min_value=1, max_value=943, value=1, step=1)
top_n = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

if st.button("🔍 Get Recommendations", use_container_width=True):
    with st.spinner("Finding best courses..."):
        try:
            res = requests.post(
                "http://127.0.0.1:8000/recommend",
                json={"user_id": int(user_id), "top_n": int(top_n)}
            )
            data = res.json()
            if "error" in data:
                st.error(data["error"])
            else:
                st.success(f"Top {top_n} recommendations for Student {user_id}:")
                for i, item in enumerate(data["recommendations"], 1):
                    rating = min(item['predicted_rating'], 5.0)
                    st.metric(
                        label=f"#{i} — Course ID: {item['item_id']}",
                        value=f"⭐ {rating:.2f} / 5.0"
                    )
        except Exception as e:
            st.error(f"API Error: {e}")

st.divider()

# ── Section 2: Single Rating Predict ──
st.subheader("🎯 Predict Single Rating")
col1, col2 = st.columns(2)
with col1:
    uid = st.number_input("Student ID", min_value=1, max_value=943, value=1, key="uid")
with col2:
    iid = st.number_input("Course ID", min_value=1, max_value=1682, value=50, key="iid")

if st.button("⚡ Predict Rating", use_container_width=True):
    with st.spinner("Predicting..."):
        try:
            res = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"user_id": int(uid), "item_id": int(iid)}
            )
            data = res.json()
            rating = min(data['predicted_rating'], 5.0)
            st.success(f"Predicted Rating: ⭐ {rating:.2f} / 5.0")
        except Exception as e:
            st.error(f"API Error: {e}")

st.divider()

# ── Section 3: Explainability ──
st.subheader("🔍 Model Explainability")
exp_user = st.number_input("Student ID for Explanation", min_value=1, max_value=943, value=1, key="exp")

if st.button("📊 Show Explanation", use_container_width=True):
    with st.spinner("Generating explanation..."):
        if exp_user in rec_df.index:
            user_ratings = rec_df.loc[exp_user]
            already_rated = user_item.loc[exp_user].dropna().index.tolist()
            unrated = user_ratings.drop(labels=already_rated, errors='ignore')
            top10 = unrated.nlargest(10).clip(upper=5.0)

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#2ecc71' if v >= 4 else '#f39c12' if v >= 3 else '#e74c3c' for v in top10.values]
            bars = ax.barh([f"Course {i}" for i in top10.index], top10.values, color=colors)
            ax.set_xlabel("Predicted Rating")
            ax.set_title(f"Top 10 Course Predictions — Student {exp_user}")
            ax.set_xlim(0, 5.5)
            ax.axvline(x=4.0, color='green', linestyle='--', alpha=0.5, label='Good (4.0)')
            ax.legend()
            for bar, val in zip(bars, top10.values):
                ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                        f'{val:.2f}', va='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.savefig('shap_plot.png', dpi=150, bbox_inches='tight')
            st.caption("Green = Highly Recommended (≥4.0) | Orange = Average | Red = Low")
        else:
            st.error("User not found!")

st.divider()
st.caption("PRO008 — Krish Kumar Rathi | B.Tech CSE (AI & ML)")