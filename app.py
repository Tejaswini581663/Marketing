import streamlit as st
import pandas as pd
import sqlite3
import pickle
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Marketing A/B Testing & AI Engine", layout="wide")
st.title("📊 Marketing Campaign Analytics & Local RAG Assistant")

# Sidebar - Machine Learning Inference
st.sidebar.header("Conversion Predictor")
total_ads = st.sidebar.slider("Total Ads Seen", 1, 100, 10)
most_ads_hour = st.sidebar.slider("Peak Hour", 0, 23, 15)
test_group = st.sidebar.selectbox("Test Group", ["ad", "psa"])

if st.sidebar.button("Predict Conversion"):
    model_path = 'models/conversion_random_forest.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        group_val = 1 if test_group == "ad" else 0
        pred = model.predict([[group_val, total_ads, most_ads_hour]])
        st.sidebar.success(f"Prediction: {'Converted' if pred[0] == 1 else 'Not Converted'}")
    else:
        st.sidebar.error("Model file not found. Run 'python scripts/train_model.py' first.")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["Database Analytics", "Visual Reports", "Local AI Assistant"])

with tab1:
    st.subheader("SQLite Campaign Records")
    db_path = 'Data/marketing.db'
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        try:
            df_db = pd.read_sql_query("SELECT * FROM marketing_campaign LIMIT 100", conn)
            st.dataframe(df_db)
        except Exception as e:
            st.error(f"Error querying table 'marketing_campaign': {e}")
        finally:
            conn.close()
    else:
        st.error("Database file missing. Run 'python scripts/push_to_db.py' first.")

with tab2:
    st.subheader("Visual Reports")
    col1, col2 = st.columns(2)
    with col1:
        img1 = "reports/conversion_by_variant.png"
        if os.path.exists(img1):
            st.image(img1, caption="Conversion by Variant")
        else:
            st.warning("Missing conversion_by_variant.png chart.")
    with col2:
        img2 = "reports/hourly_engagement_heatmap.png"
        if os.path.exists(img2):
            st.image(img2, caption="Hourly Engagement")
        else:
            st.warning("Missing hourly_engagement_heatmap.png chart.")

with tab3:
    st.subheader("Offline Marketing AI (Ollama + ChromaDB)")
    user_query = st.text_input("Ask a question about the campaign:", "What is the key insight from the test?")
    if st.button("Query Local RAG"):
        persist_dir = 'data/chroma_db'
        if not os.path.exists(persist_dir):
            st.error("Vector DB not found. Run 'python scripts/rag_campaign_qa.py' first.")
        else:
            with st.spinner("Analyzing context via local Llama 3.2..."):
                try:
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vectorstore = Chroma(
                        collection_name="local_marketing_insights",
                        embedding_function=embeddings,
                        persist_directory=persist_dir
                    )
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
                    llm = ChatOllama(model="llama3.2", temperature=0)

                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    system_prompt = "You are a marketing assistant. Answer using only context:\n\n{context}"
                    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
                    
                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt | llm | StrOutputParser()
                    )
                    
                    answer = rag_chain.invoke(user_query)
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error generating AI response: {e}")