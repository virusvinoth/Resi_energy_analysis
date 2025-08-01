import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Custom modules (assumed)
from forecast import load_data, train_forecast_model, predict_usage
from analyzer import detect_anomalies
from recommender import generate_recommendations

# Set page config
st.set_page_config(page_title="HomeWatt", page_icon="💡", layout="wide")

# Load NLP models once
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
recommendations = [
    "Use energy efficient appliances",
    "Turn off devices when not in use",
    "Avoid using heavy devices during peak hours",
    "Switch to LED lighting",
    "Use smart plugs and schedulers",
    "Monitor real-time usage through HomeWatt dashboard"
]
corpus_embeddings = similarity_model.encode(recommendations, convert_to_tensor=True)

# Tabs for Navigation
tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Assistant"])

# ----------------------
# Tab 1: Dashboard
# ----------------------
with tab1:
    st.title("📊 HomeWatt - Energy Analytics Dashboard")

    uploaded_file = st.file_uploader("Upload your energy usage CSV", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)

        if not {'timestamp', 'usage_kwh'}.issubset(df.columns):
            st.error("CSV must contain 'timestamp' and 'usage_kwh'")
            st.stop()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day_name()

        # Chart 1: Line Chart
        st.subheader("📈 Usage Over Time")
        st.line_chart(df.set_index('timestamp')['usage_kwh'])

        # Chart 2: Hourly Usage Bar Chart
        st.subheader("🕒 Hourly Energy Usage")
        hourly_usage = df.groupby('hour')['usage_kwh'].sum().reset_index()
        bar_chart = alt.Chart(hourly_usage).mark_bar().encode(
            x='hour:O', y='usage_kwh:Q', tooltip=['hour', 'usage_kwh']
        ).properties(height=300)
        st.altair_chart(bar_chart, use_container_width=True)

        # Chart 3: Heatmap Day vs Hour
        st.subheader("🔥 Hour vs Day Usage Heatmap")
        heatmap_data = df.pivot_table(index='day', columns='hour', values='usage_kwh', aggfunc='sum').fillna(0)
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt=".1f", linewidths=0.3, ax=ax)
        plt.title("Average Usage by Day and Hour")
        st.pyplot(fig)

        # Chart 4: Rolling Avg
        st.subheader("📊 24-Hour Rolling Average")
        df_sorted = df.sort_values('timestamp')
        df_sorted['rolling_avg'] = df_sorted['usage_kwh'].rolling(window=24).mean()
        st.line_chart(df_sorted.set_index('timestamp')[['usage_kwh', 'rolling_avg']])

        # Forecasting
        st.subheader("🔮 Forecasting (Next 24 Hours)")
        model = train_forecast_model(df)
        forecast = predict_usage(model, list(range(24)))
        st.line_chart(forecast)

        # Anomaly Detection
        st.subheader("⚠️ Anomaly Detection")
        anomalies = detect_anomalies(df)
        st.dataframe(anomalies[anomalies['anomaly'] == -1])

        # Recommendations
        st.subheader("💡 Optimization Suggestions")
        for rec in generate_recommendations(df):
            st.info(rec)
    else:
        st.warning("👈 Please upload a CSV file to begin.")

# ----------------------
# Tab 2: Chat Assistant
# ----------------------
with tab2:
    st.title("💬 HomeWatt - AI Energy Assistant")
    user_input = st.text_input("Ask HomeWatt anything about your energy usage:")

    if user_input:
        # Sample static context, or you can make it dynamic using uploaded `df`
        energy_context = """
        The air conditioner consumes around 2000W, washing machine 500W, and kitchen appliances 800W.
        Major consumption is observed between 6PM and 10PM. Usage can be optimized by turning off idle devices,
        shifting to LED lighting, and scheduling heavy appliance usage during non-peak hours.
        """

        answer = qa_pipeline(question=user_input, context=energy_context)
        query_emb = similarity_model.encode(user_input, convert_to_tensor=True)
        sim_scores = util.cos_sim(query_emb, corpus_embeddings)
        best_idx = sim_scores.argmax().item()
        best_tip = recommendations[best_idx]

        st.subheader("🌟 Answer from HomeWatt:")
        st.write(answer['answer'])

        st.subheader("🔎 Best Energy Tip:")
        st.success(best_tip)

        st.caption("(Powered by open-source NLP models)")
