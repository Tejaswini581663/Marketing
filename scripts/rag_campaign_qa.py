import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

summary_file = 'data/campaign_summary.txt'

# 1. Safety Check: Verify summary file exists and is not empty
if not os.path.exists(summary_file) or os.path.getsize(summary_file) == 0:
    print(f"Error: '{summary_file}' is missing or empty. Please run 'python scripts/llm_insight_generator.py' first!")
    sys.exit(1)

# 2. Load Campaign Data Summary
loader = TextLoader(summary_file, encoding='utf-8')
documents = loader.load()

# 3. Chunk Documents safely
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Prevent empty embedding error
if not docs:
    print("Error: No text chunks were generated from the summary file.")
    sys.exit(1)

print(f"Successfully loaded and split document into {len(docs)} text chunk(s).")

# 4. Create Vector Store with Local Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="local_marketing_insights"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatOllama(model="llama3.2", temperature=0)

# 5. Define Context Formatter & Prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

system_prompt = (
    "You are a helpful CMO marketing assistant.\n"
    "Answer the user question using only the following retrieved context:\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

# 6. LCEL RAG Pipeline
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Execute Query
user_query = "Should we continue running the PSA control group or switch 100% to Ads?"
print(f"\nUser Query: {user_query}")

response = rag_chain.invoke(user_query)
print("\n--- Local RAG Engine Response ---")
print(response)