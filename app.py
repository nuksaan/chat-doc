import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyPDFLoader

def load_as_dataframe(uploaded_file):
    ext = uploaded_file.name.lower().split(".")[-1]

    try:
        if ext == "csv":
            df = pd.read_csv(uploaded_file)

        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)

        elif ext == "json":
            # Must be a list of {question:"", answer:""}
            data = json.load(uploaded_file)
            df = pd.DataFrame(data)

        elif ext == "txt":
            # Expect lines like: question|answer
            raw = uploaded_file.read().decode("utf-8").splitlines()
            qa_list = [line.split("|") for line in raw if "|" in line]
            df = pd.DataFrame(qa_list, columns=["question", "answer"])

        else:
            raise ValueError("Unsupported file type")

        # Validate required columns
        if not {"question", "answer"}.issubset(df.columns):
            raise ValueError("File MUST contain 'question' and 'answer' columns/fields.")

        return df

    except Exception as e:
        raise ValueError(f"Error reading file: {e}")


def build_vector_index(uploaded_file):
    ext = uploaded_file.name.lower().split(".")[-1]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 🧩 Case 1: structured Q/A files → build Q/A docs
    if ext in ["csv", "xlsx", "xls", "json", "txt"]:
        df = load_as_dataframe(uploaded_file)

        docs = [
            Document(page_content=f"Q: {row['question']}\nA: {row['answer']}")
            for _, row in df.iterrows()
        ]

    # 📄 Case 2: PDF → treat each page as a document
    elif ext == "pdf":
        # Need a real temp file path for PyPDFLoader
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # (optional) clean up temp file
        os.remove(temp_path)

    else:
        raise ValueError(f"Unsupported file type: .{ext}")

    # Build vector store for either case
    vector_index = DocArrayInMemorySearch.from_documents(docs, embeddings)
    return vector_index

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Model
llm_model = "gpt-4o"

# Prompt for final answer
custom_prompt = PromptTemplate(
    template="""
You are a helpful assistant. Use the following context from documents and previous chat history to answer the user's question in a concise, friendly, and cute way.
Summarize and rephrase instead of copying directly. If you can't find a direct answer in the context, do your best to help the user using partial information. If you're still unsure, say "I don't know."

Chat History:
{chat_history}

Context from documents:
{context}

Question: {question}

Answer:
""",
    input_variables=["context", "question", "chat_history"]
)

# Prompt for rewriting follow-up questions
rewrite_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
Given the following conversation and a follow-up question, rewrite the follow-up question to be a fully standalone question.
Explicitly replace any pronouns like "it", "they", "them", etc., with what they refer to, using context from the chat history.

Chat History:
{chat_history}

Follow-Up Question: {question}

Rewritten Standalone Question:
"""
)

st.title("🧢 Q&A (LangChain + Memory + Forced Rewrite)")
st.write("Upload a CSV Q&A and ask follow-up questions — even using 'it' or 'them'!")

uploaded_file = st.file_uploader(
    "Upload a file",
    type=["csv", "xlsx", "xls", "json", "txt", "pdf"]
)
user_query = st.text_input("Ask a question", placeholder="e.g., How do I reset my password?")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file:
    try:
        vector_index = build_vector_index(uploaded_file)
        st.success("File processed successfully! You can now ask questions.")
    except Exception as e:
        st.error(e)

    llm = ChatOpenAI(temperature=0.0, model=llm_model)

    # Conversation memory (stores entire history)
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",
    #     return_messages=True
    # )

    memory = ConversationBufferWindowMemory(
        k=3,  # keep last 3 interactions
        memory_key="chat_history",
        return_messages=True
    )

#     memory = ConversationSummaryMemory(
    #     llm=llm,
    #     memory_key="chat_history",
    #     return_messages=True
    # )

    # memory = ConversationSummaryBufferMemory(
    #     llm=llm,
    #     max_token_limit=1000,  # total summary + recent messages tokens
    #     memory_key="chat_history",
    #     return_messages=True
    # )

    # Rewriting chain to force standalone questions
    # rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)

    # QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_index.as_retriever(),
        memory=memory,
        condense_question_llm=llm,  # The condense LLM is a secondary LLM used to rewrite or rephrase follow-up questions into standalone, self-contained questions before doing retrieval.
        condense_question_prompt=rewrite_prompt,
          # still used for internal fallback
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        verbose=False
    )

if user_query:
    with st.spinner("Rewriting question and thinking..."):
        # Check if there is history
        # if len(st.session_state.chat_history) == 0:
        #     rewritten_question = user_query
        # else:
        #     chat_history_str = "\n".join([f"{role}: {text}" for role, text in st.session_state.chat_history])
        #     rewritten_question = rewrite_chain.run({
        #         "chat_history": chat_history_str,
        #         "question": user_query,
        #     })

        # st.write("🔎 Rewritten Question:", rewritten_question)

        # retrieved_docs = vector_index.similarity_search(rewritten_question, k=3)
        # st.write("🗂️ Retrieved Context:", [doc.page_content for doc in retrieved_docs])

        # Now pass rewritten question to main chain
        result = qa_chain({"question": user_query})

        st.markdown("### 📄 Response:")
        st.markdown(result["answer"] if isinstance(result, dict) else result)

        retrieved_docs = vector_index.similarity_search(user_query, k=3)
        st.write("🗂️ Retrieved Context:", [doc.page_content for doc in retrieved_docs])

        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("AI", result["answer"] if isinstance(result, dict) else result))

    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation History")
        for role, text in st.session_state.chat_history:
            st.markdown(f"**{role}:** {text}")

        st.markdown("### 🧾 Memory (internal messages):")
        st.markdown(
            "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in memory.chat_memory.messages])
        )

        if st.button("🚀 End Chat and Show Final Summary"):
            all_messages = [f"{role}: {text}" for role, text in st.session_state.chat_history]
            if all_messages:
                docs_to_summarize = [Document(page_content=msg) for msg in all_messages]
                summarize_chain = load_summarize_chain(llm, chain_type="stuff")

                # load_summarize_chain() is a helper that builds a pre-defined summarization pipeline.

                # chain_type="stuff" is summarization strategy

                # "stuff" concatenates all documents into one big string and summarizes in one go

                # Other options include "map_reduce" and "refine" for large document sets. 
                final_summary = summarize_chain.run(docs_to_summarize)

                # Feeds your list of Document objects into the summarization chain.
    
                st.markdown("### 🌸 Final Detailed Summary of Our Whole Conversation:")
                st.markdown(final_summary)
            else:
                st.markdown("No messages to summarize yet! 💬")

else:
    st.info("👆 Please upload a CSV file to get started!")
