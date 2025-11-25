from flask import Flask, render_template, request
from memory_chain import chat_with_memory, memory as chat_memory
from rag import save_uploaded_pdf, ingest_pdf, rag_answer, delete_pdf_chunks
from reliable_agents import reliable_agent_task
from langchain.memory import ConversationBufferMemory
import os

app = Flask(__name__)

# Separate memories for each module
rag_memory = ConversationBufferMemory(return_messages=True)
agent_memory = ConversationBufferMemory(return_messages=True)

def get_history(mem):
    """Convert LangChain memory to (role, message) list for rendering."""
    history = []
    for msg in mem.chat_memory.messages:
        role = "human" if msg.type == "human" else "ai"
        history.append((role, msg.content))
    return history

@app.route("/", methods=["GET"])
def index():
    # Simple landing page that links to the three separate pages
    return render_template("index.html")

# --- Chat with Memory ---
@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        if user_input:
            chat_with_memory(user_input)
    return render_template(
        "chat.html",
        chat_history=get_history(chat_memory)
    )

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    chat_memory.clear()
    return render_template(
        "chat.html",
        chat_history=get_history(chat_memory)
    )

# --- RAG PDF Q&A ---
@app.route("/rag", methods=["GET", "POST"])
def rag_route():
    selected_pdf = request.values.get("selected_pdf", "").strip()
    if request.method == "POST":
        question = request.form.get("question", "")
        pdf_file = request.files.get("pdf_file")

        if pdf_file and pdf_file.filename != "":
            pdf_path = save_uploaded_pdf(pdf_file)
            ingest_pdf(pdf_path)

        response = rag_answer(question, selected_pdf=selected_pdf)

        # Save Q&A into RAG memory
        if question.strip():
            rag_memory.chat_memory.add_user_message(question)
            rag_memory.chat_memory.add_ai_message(response)
    # Build list of uploaded PDFs from data/pdfs
    pdf_dir = os.path.join("data", "pdfs")
    try:
        pdf_list = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    except FileNotFoundError:
        pdf_list = []
    return render_template(
        "rag.html",
        rag_history=get_history(rag_memory),
        pdf_list=pdf_list,
        selected_pdf=selected_pdf
    )

@app.route("/rag/select", methods=["POST"])
def rag_select_pdf():
    selected_pdf = request.form.get("selected_pdf", "")
    # Re-render page with selected pdf; history remains
    pdf_dir = os.path.join("data", "pdfs")
    try:
        pdf_list = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    except FileNotFoundError:
        pdf_list = []
    return render_template(
        "rag.html",
        rag_history=get_history(rag_memory),
        pdf_list=pdf_list,
        selected_pdf=selected_pdf
    )

@app.route("/rag/delete", methods=["POST"])
def rag_delete_pdf():
    pdf_name = request.form.get("pdf_name", "")
    # Delete chunks from vector DB
    deleted = delete_pdf_chunks(pdf_name)  # noqa: F841 not used in template
    # Optionally delete file from disk as well
    pdf_path = os.path.join("data", "pdfs", pdf_name)
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except OSError:
            pass
    # Refresh list and clear selection if it was this file
    pdf_dir = os.path.join("data", "pdfs")
    try:
        pdf_list = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    except FileNotFoundError:
        pdf_list = []
    return render_template(
        "rag.html",
        rag_history=get_history(rag_memory),
        pdf_list=pdf_list,
        selected_pdf=""
    )

@app.route("/rag/clear_chat", methods=["POST"])
def rag_clear_chat():
    selected_pdf = request.form.get("selected_pdf", "")
    rag_memory.clear()
    pdf_dir = os.path.join("data", "pdfs")
    try:
        pdf_list = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    except FileNotFoundError:
        pdf_list = []
    return render_template(
        "rag.html",
        rag_history=get_history(rag_memory),
        pdf_list=pdf_list,
        selected_pdf=selected_pdf
    )

# --- Reliable Agent ---
@app.route("/agent", methods=["GET", "POST"])
def agent():
    if request.method == "POST":
        task = request.form.get("task", "").strip()
        if task:
            response = reliable_agent_task(task)

            # Save into agent memory
            agent_memory.chat_memory.add_user_message(task)
            agent_memory.chat_memory.add_ai_message(response)
    return render_template(
        "agent.html",
        agent_history=get_history(agent_memory)
    )

@app.route("/agent/clear_chat", methods=["POST"])
def agent_clear_chat():
    agent_memory.clear()
    return render_template(
        "agent.html",
        agent_history=get_history(agent_memory)
    )

if __name__ == "__main__":
    # Change host to 0.0.0.0 to make it accessible from other devices on your network
    # Changed port to 5005 as requested
    app.run(debug=True, host='0.0.0.0', port=5005)
