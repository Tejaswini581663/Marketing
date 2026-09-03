import os
import sys
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

summary_file = 'data/campaign_summary.txt'
persist_dir = 'data/chroma_db'

# 1. Safety Check: Verify summary file exists and is not empty
if not os.path.exists(summary_file) or os.path.getsize(summary_file) == 0:
    print(f"Error: '{summary_file}' is missing or empty. Run 'python scripts/llm_insight_generator.py' first!")
    sys.exit(1)

# 2. Read Summary File Directly
with open(summary_file, 'r', encoding='utf-8') as f:
    content = f.read()

documents = [Document(page_content=content)]

# 3. Chunk Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 4. Create Persistent Vector Store using langchain-chroma
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="local_marketing_insights",
    persist_directory=persist_dir
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

# 6. Construct LCEL RAG Pipeline
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Test Query Execution
user_query = "Should we continue running the PSA control group or switch 100% to Ads?"
print(f"\nUser Query: {user_query}")

response = rag_chain.invoke(user_query)
print("\n--- Local RAG Engine Response ---")
print(response)