import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
import datetime

# Load environment variables
load_dotenv()

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Determine model based on date
current_date = datetime.datetime.now().date()
target_date = datetime.date(2024, 6, 12)
llm_model = "gpt-3.5-turbo-0301"

# UI layout
st.title("🧢 Product Catalog Q&A (LangChain + Memory)")
st.write("Upload a CSV catalog and ask questions about it!")

# File uploader
uploaded_file = st.file_uploader("Upload your product catalog CSV", type="csv")

# Query input
user_query = st.text_input("Ask a question about your catalog",
                           placeholder="e.g., List all shirts with sun protection")

# Memory setup (Session-based)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file:
    with open("uploaded_catalog.csv", "wb") as f:
        f.write(uploaded_file.read())

    loader = CSVLoader(file_path="uploaded_catalog.csv")
    embeddings = OpenAIEmbeddings()

    # Create vector index in-memory vectorstore using DocArray, which is great for small/temporary datasets. But for more scalable, persistent, or powerful setups, there are many alternatives.
    # we can use pinecone, elasticsearch or redis
    vector_index = DocArrayInMemorySearch.from_documents(loader.load(), embeddings)  

    # LLM and Summarized Memory
    llm = ChatOpenAI(temperature=0.0, model=llm_model)
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # use this if u want the entire chat
    # there is also conversation buffer window memory which stores the most recent one depending on k - memory = ConversationBufferWindowMemory(k=1)
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=100,
    )

    # Conversational Retrieval Chain with Memory - this is the best one since we need to hold a conversation over the Q&A
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_index.as_retriever(),
        memory=memory,
        verbose=False
    )

    # If user entered a query
    if user_query:
        with st.spinner("Searching the catalog..."):
            result = qa_chain.run(user_query)
            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("AI", result))

        # Display answer
        st.markdown("### 📄 Response:")
        st.markdown(result)

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation History")
    for role, text in st.session_state.chat_history:
        st.markdown(f"**{role}:** {text}")
