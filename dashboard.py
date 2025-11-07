import os
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# 🔄 Auto-refresh every 5 seconds
st_autorefresh(interval=5000, limit=None, key="dashboard_autorefresh")

# Paths
BASE_DIR = os.path.dirname(__file__)
log_path = os.path.join(BASE_DIR, "simulation_logs.csv")

st.title("🚦 Traffic Control Reinforcement Learning Dashboard")

# Debug info
st.write("🔍 Current working directory:", os.getcwd())
st.write("🔍 Looking for file at:", log_path)

# File check
if not os.path.exists(log_path):
    st.error("❌ simulation_logs.csv not found. Please run train.py first.")
else:
    try:
        df = pd.read_csv(log_path)

        if df.empty:
            st.warning("⚠ simulation_logs.csv is empty. Run training to generate logs.")
        else:
            st.success("✅ Loaded simulation logs successfully!")

            # --- KPI Metrics ---
            latest_gen = int(df["generation"].iloc[-1])
            latest_reward = float(df["reward"].iloc[-1])
            latest_accuracy = float(df["accuracy"].iloc[-1])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latest Generation", latest_gen)
            with col2:
                prev_reward = df["reward"].iloc[-2] if len(df) > 1 else 0
                st.metric("Reward", latest_reward, delta=latest_reward - prev_reward)
            with col3:
                prev_acc = df["accuracy"].iloc[-2] if len(df) > 1 else 0
                st.metric("Accuracy (%)", f"{latest_accuracy:.2f}", delta=latest_accuracy - prev_acc)

            # Chart + logs
            st.line_chart(df.set_index("generation")[["reward", "accuracy"]])
            st.write("### Raw Logs")
            st.dataframe(df)

    except Exception as e:
        st.error(f"🚨 Error reading simulation_logs.csv: {e}")