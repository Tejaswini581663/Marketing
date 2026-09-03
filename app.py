import streamlit as st
import pandas as pd
import sqlite3
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Marketing A/B Testing & AI Engine", layout="wide")
st.title("📊 Marketing Campaign Analytics & Local RAG Assistant")

# Sidebar - Tier 3 ML Inference
st.sidebar.header("Conversion Predictor")
total_ads = st.sidebar.slider("Total Ads Seen", 1, 100, 10)
most_ads_hour = st.sidebar.slider("Peak Hour", 0, 23, 15)
test_group = st.sidebar.selectbox("Test Group", ["ad", "psa"])

if st.sidebar.button("Predict Conversion"):
    with open('models/conversion_random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    # Simple encoding matching your training format
    group_val = 1 if test_group == "ad" else 0
    pred = model.predict([[group_val, total_ads, most_ads_hour]])
    st.sidebar.success(f"Prediction: {'Converted' if pred[0] == 1 else 'Not Converted'}")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["Database Analytics", "Visual Reports", "Local AI Assistant"])

with tab1:
    st.subheader("Tier 1 & 2: SQLite Campaign Records")
    conn = sqlite3.connect('Data/marketing.db')
    df_db = pd.read_sql_query("SELECT * FROM marketing_campaign LIMIT 100", conn)
    st.dataframe(df_db)
    conn.close()

with tab2:
    st.subheader("Visual Reports")
    col1, col2 = st.columns(2)
    with col1:
        st.image("reports/conversion_by_variant.png", caption="Conversion by Variant")
    with col2:
        st.image("reports/hourly_engagement_heatmap.png", caption="Hourly Engagement")

with tab3:
    st.subheader("Tier 4: Offline Marketing AI (Ollama + ChromaDB)")
    user_query = st.text_input("Ask a question about the campaign:", "What is the key insight from the test?")
    if st.button("Query Local RAG"):
        with st.spinner("Analyzing context via local Llama 3.2..."):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma(collection_name="local_marketing_insights", embedding_function=embeddings)
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