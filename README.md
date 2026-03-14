# telegram-rag-summarizer
# Sentinel RAG-Summarizer 🤖📄

A high-performance **Retrieval-Augmented Generation (RAG)** agent that connects **Telegram** messages to a private, dynamic knowledge base. This project uses the **Model Context Protocol (MCP)** to separate the knowledge retrieval layer from the communication logic, ensuring a modular and scalable AI architecture.

---

## 🌟 How It Works (The "Sentinel" Loop)

1.  **Ingestion:** Admins push raw data via the `/admin/feed` endpoint.
2.  **Indexing:** The agent updates the `knowledge_base.docx` and triggers the remote **MCP Server** to re-index content immediately using **SentenceTransformers**.
3.  **Telegram Query:** When a user prompts the bot, the request is sent to an LLM along with a tool definition for the MCP server.
4.  **Retrieval & Summarization:** The LLM autonomously decides to "call" the search tool. The agent fetches the most relevant context from the MCP server, and the LLM synthesizes a final summary for the user.

---

## 🛠️ Technical Stack

| Component | Technology |
| :--- | :--- |
| **Orchestration** | Python / FastAPI |
| **Agent Logic** | Custom ChatAgent (LLM-powered) |
| **Communication** | Telegram Bot API (Webhooks) |
| **Knowledge Engine** | [mcp-tool-provider](https://github.com/yourusername/mcp-tool-provider) (FastMCP) |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Similarity** | NumPy-based Cosine Similarity |

---

## ⚙️ Installation & Setup

### 1. Prerequisites
* **Python 3.10+**
* A **Telegram Bot Token** from [@BotFather](https://t.me/botfather).
* The **MCP Server** deployed (see the [mcp-tool-provider](https://github.com/yourusername/mcp-tool-provider) repo).

### 2. Local Installation
```bash
git clone [https://github.com/yourusername/rag-summarizer.git](https://github.com/yourusername/rag-summarizer.git)
cd telegram-rag-summarizer
pip install -r requirements.txt
