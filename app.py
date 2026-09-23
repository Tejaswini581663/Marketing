import streamlit as st
import pandas as pd
import sqlite3
import pickle
import os

st.set_page_config(page_title="Marketing Analytics", layout="wide")

st.title("📊 Marketing Campaign Analytics & Local RAG Assistant")

# --- 1. CACHED DATABASE QUERY ---
@st.cache_data(ttl=600)
def load_data():
    db_path = 'Data/marketing.db'
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM marketing_campaign LIMIT 100", conn)
        conn.close()
        return df
    return pd.DataFrame()

# --- 2. CACHED RAG SETUP ---
@st.cache_resource
def load_rag_chain():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory="Data/chroma_db", embedding_function=embeddings)
        return db
    except Exception as e:
        return None

# --- SIDEBAR WIDGETS ---
st.sidebar.header("Conversion Predictor")
test_group = st.sidebar.selectbox("Test Group", ["ad", "psa"])
total_ads = st.sidebar.number_input("Total Ads Seen", min_value=1, value=10)
most_ads_hour = st.sidebar.slider("Peak Hour", 0, 23, 12)

if st.sidebar.button("Predict Conversion", key="predict_conversion_btn"):
    model_path = 'models/conversion_random_forest.pkl'
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            group_val = 1 if test_group == "ad" else 0
            pred = model.predict([[group_val, total_ads, most_ads_hour]])
            res_text = 'Converted' if pred[0] == 1 else 'No Conversion'
            st.sidebar.success(f"Prediction: {res_text}")
        except Exception:
            st.sidebar.error("Model file corrupted. Re-run train_model.py.")
    else:
        st.sidebar.warning("Model file not found.")

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["Database Analytics", "Visual Reports", "Local AI Assistant"])

with tab1:
    st.subheader("SQLite Campaign Records")
    df_records = load_data()
    if not df_records.empty:
        st.dataframe(df_records, use_container_width=True)
    else:
        st.info("No database records found. Ensure Data/marketing.db is uploaded.")

with tab2:
    st.subheader("Visual Analysis")
    if os.path.exists("reports"):
        for img in os.listdir("reports"):
            if img.endswith((".png", ".jpg")):
                st.image(os.path.join("reports", img))

with tab3:
    st.subheader("Local AI Assistant")
    st.info("RAG Query engine ready for marketing campaign documentation.")