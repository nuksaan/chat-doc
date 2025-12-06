# 🧢 Chat-Doc 

An interactive **Q&A app** built with **Streamlit** and **LangChain** that lets you:

- Upload a **FILE** or **Q&A dataset** (CSV, Excel, JSON, TXT, PDF)
- Embed the content using **OpenAI embeddings**
- Ask natural-language questions and get answers powered by **OpenAI chat models**
- Maintain **conversational context** with memory

---

## ✨ Features

- 🔼 Upload multiple file types:
  - `csv`, `xlsx`, `xls`, `json`, `txt`, `pdf`
- 🧠 Uses OpenAI embeddings (e.g. `text-embedding-3-small`) + vector store (`DocArrayInMemorySearch`)
- 💬 Conversational Q&A using `ChatOpenAI` + `ConversationalRetrievalChain`
- 🧾 For structured files (CSV/Excel/JSON/TXT) with `question` + `answer` fields:
  - Automatically combines into:  
    `Q: <question>\nA: <answer>`
- 📄 For PDFs:
  - Loads the text of each page into documents and makes them searchable

---

## 🧱 Tech Stack

- **Python 3.10+** (recommended)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- `DocArrayInMemorySearch` vector store

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/chat-doc.git
cd chat-doc
