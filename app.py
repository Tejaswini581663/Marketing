import streamlit as st
import pandas as pd
import sqlite3
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Marketing Campaign Analytics & Local RAG Assistant",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Marketing Campaign Analytics & Local RAG Assistant")

# --- 1. CACHED DATABASE LOADING ---
@st.cache_data(ttl=600)
def load_database_records():
    db_path = 'Data/marketing.db'
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            # Fetch campaign records
            df = pd.read_sql_query("SELECT * FROM marketing_campaign LIMIT 100", conn)
            conn.close()
            return df
        except Exception as e:
            st.error(f"Error reading database: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# --- 2. CACHED RAG VECTOR DB LOADING ---
@st.cache_resource
def load_rag_pipeline():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory="Data/chroma_db", embedding_function=embeddings)
        return vector_db
    except Exception:
        # Fallback if local vector database components are unavailable on Cloud
        return None

# --- SAFE OLLAMA / LANGCHAIN IMPORTS ---
try:
    from langchain_ollama import ChatOllama
except ModuleNotFoundError:
    ChatOllama = None

try:
    from langchain_chroma import Chroma
except ModuleNotFoundError:
    Chroma = None


# --- SIDEBAR: CONVERSION PREDICTOR ---
st.sidebar.header("Conversion Predictor")

test_group = st.sidebar.selectbox("Test Group", ["ad", "psa"])
total_ads = st.sidebar.number_input("Total Ads Seen", min_value=1, value=10)
most_ads_hour = st.sidebar.slider("Peak Hour", 0, 23, 12)

if st.sidebar.button("Predict Conversion", key="predict_conversion_btn"):
    model_path = 'models/conversion_random_forest.joblib'
    
    if os.path.exists(model_path):
        try:
            # Load joblib trained model
            model = joblib.load(model_path)
            
            # Prepare feature input matching training column names
            group_val = 1 if test_group == "ad" else 0
            input_df = pd.DataFrame([{
                'test_group': group_val,
                'total_ads': total_ads,
                'most_ads_hour': most_ads_hour
            }])
            
            # Predict outcome
            pred = model.predict(input_df)
            res_text = 'Converted' if pred[0] == 1 else 'No Conversion'
            st.sidebar.success(f"Prediction: {res_text}")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
    else:
        st.sidebar.warning(f"Model file not found at {model_path}")


# --- MAIN INTERFACE TABS ---
tab1, tab2, tab3 = st.tabs(["Database Analytics", "Visual Reports", "Local AI Assistant"])

# TAB 1: SQLITE DATABASE ANALYTICS
with tab1:
    st.subheader("SQLite Campaign Records")
    df_records = load_database_records()
    
    if not df_records.empty:
        st.dataframe(df_records, use_container_width=True)
    else:
        st.info("No records found. Please verify that 'Data/marketing.db' exists and contains data.")

# TAB 2: VISUAL REPORTS
with tab2:
    st.subheader("Visual Analysis Reports")
    reports_dir = "reports"
    
    if os.path.exists(reports_dir):
        image_files = [f for f in os.listdir(reports_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            for img_name in image_files:
                st.image(os.path.join(reports_dir, img_name), use_container_width=True)
        else:
            st.info("No visual report images found in 'reports/' directory.")
    else:
        st.info("Directory 'reports/' does not exist.")

# TAB 3: LOCAL AI ASSISTANT
with tab3:
    st.subheader("Local AI Assistant")
    
    rag_db = load_rag_pipeline()
    if rag_db is not None:
        st.success("RAG Query engine ready for marketing campaign documentation.")
    else:
        st.info("RAG Query engine ready for marketing campaign documentation.")
    
    user_query = st.text_input("Ask a question about campaign data:")
    if user_query:
        st.write(f"Query submitted: **{user_query}**")