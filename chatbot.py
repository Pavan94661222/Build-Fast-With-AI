import os
import sys
import pandas as pd
from datetime import datetime
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# Function to load and split data
def load_data(file_path="emails.csv"):
    try:
        data = pd.read_csv(file_path)
        if "text" not in data.columns:
            raise ValueError("CSV file must contain a 'text' column")

        texts = data["text"].dropna().tolist()
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )

        split_docs = []
        for text in texts:
            chunks = text_splitter.split_text(text)
            split_docs.extend([Document(page_content=chunk) for chunk in chunks])

        return split_docs

    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        sys.exit(1)

# Function to setup RAG pipeline
def setup_rag(docs):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        llm = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            huggingfacehub_api_token="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",
            model_kwargs={"temperature": 0.5, "max_length": 512}
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
    except Exception as e:
        print(f"❌ RAG setup error: {e}")
        sys.exit(1)

# Save to .txt and .xlsx
def save_response_to_files(query, response):
    # Save to .txt
    with open("chat_history.txt", "a", encoding="utf-8") as f:
        f.write(f"Q: {query}\nA: {response}\n\n")

    # Save to .xlsx
    filename = "chat_history.xlsx"
    if os.path.exists(filename):
        df = pd.read_excel(filename)
    else:
        df = pd.DataFrame(columns=["Timestamp", "Query", "Response"])

    df.loc[len(df)] = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), query, response]
    df.to_excel(filename, index=False)

# Load data and initialize pipeline
try:
    documents = load_data()
    qa_chain = setup_rag(documents)
except Exception as e:
    print(f"❌ Initialization error: {str(e)}")
    sys.exit(1)

# Streamlit or CLI
if 'streamlit' in sys.modules:
    import streamlit as st

    st.title("📬 Email RAG Chatbot")
    st.write("Ask questions about your `emails.csv` dataset")

    query = st.text_input("💬 Ask your question:")

    if query:
        with st.spinner("🔍 Searching for answers..."):
            try:
                result = qa_chain({"query": query})
                st.subheader("🧠 Answer:")
                st.success(result["result"])

                st.subheader("📄 Source Documents")
                for i, doc in enumerate(result["source_documents"]):
                    with st.expander(f"Source {i+1}"):
                        st.code(doc.page_content)

                save_response_to_files(query, result["result"])
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

else:
    print("🤖 Email Chatbot Ready! (type 'exit' to quit)")
    while True:
        try:
            query = input("\nYou: ").strip()
            if query.lower() in ["exit", "quit"]:
                print("👋 Exiting. Chat saved to 'chat_history.txt' and 'chat_history.xlsx'.")
                break

            if not query:
                continue

            result = qa_chain({"query": query})
            print(f"Bot: {result['result']}")

            save_response_to_files(query, result['result'])
            print("(✅ Response saved to chat_history.txt and chat_history.xlsx)")

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
