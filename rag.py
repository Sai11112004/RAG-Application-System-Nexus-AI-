import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# Add rate limiting
last_llm_request_time = 0
MIN_LLM_REQUEST_INTERVAL = 2  # Minimum 2 seconds between LLM requests

# Persistent directories
DATA_DIR = os.path.join(os.getcwd(), "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
VECTOR_DIR = os.path.join(DATA_DIR, "chroma")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key or api_key == "your_api_key_here":
		raise ValueError("GEMINI_API_KEY not set in environment")
	return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)


def _get_vectorstore() -> Chroma:
	embeddings = _get_embeddings()
	return Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)


def ingest_pdf(file_path: str) -> int:
	"""Ingest a PDF: load, split into chunks, and add to persistent Chroma.

	Returns number of chunks added.
	"""
	loader = PyPDFLoader(file_path)
	documents: List[Document] = loader.load()

	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=1200,
		chunk_overlap=200,
		length_function=len,
		add_start_index=True,
	)
	chunks = text_splitter.split_documents(documents)

	vectorstore = _get_vectorstore()
	vectorstore.add_documents(chunks)
	vectorstore.persist()
	return len(chunks)


def save_uploaded_pdf(file_storage) -> str:
	"""Save an uploaded Werkzeug FileStorage to disk and return its path."""
	filename = file_storage.filename
	if not filename:
		raise ValueError("Invalid filename")
	# Basic sanitization: strip directory components
	filename = os.path.basename(filename)
	# Ensure unique name
	base, ext = os.path.splitext(filename)
	counter = 1
	candidate = filename
	while os.path.exists(os.path.join(PDF_DIR, candidate)):
		candidate = f"{base}_{counter}{ext}"
		counter += 1
	path = os.path.join(PDF_DIR, candidate)
	file_storage.save(path)
	return path


def rag_answer(query: str, selected_pdf: Optional[str] = None, k: int = 5) -> str:
	"""Answer using RAG over the local Chroma store.

	If selected_pdf is provided, retrieval is filtered to chunks whose
	metadata 'source' contains that file name.
	"""
	global last_llm_request_time
	
	if not query.strip():
		return "Please provide a question."

	# Retrieve
	vectorstore = _get_vectorstore()
	search_kwargs = {"k": k}
	if selected_pdf:
		# Filter by exact source path for the selected file
		full_path = os.path.join(PDF_DIR, selected_pdf)
		search_kwargs["filter"] = {"source": {"$eq": full_path}}
	retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
	relevant_docs = retriever.get_relevant_documents(query)

	if not relevant_docs:
		return "No relevant context found. Try ingesting PDFs first or rephrasing your question."

	# Build context
	context_parts = []
	for i, doc in enumerate(relevant_docs, start=1):
		source = doc.metadata.get("source", "unknown")
		page = doc.metadata.get("page", "?")
		context_parts.append(f"[Chunk {i}] (source: {source}, page: {page})\n{doc.page_content}")
	context_text = "\n\n".join(context_parts)

	# Rate limiting
	current_time = time.time()
	time_since_last = current_time - last_llm_request_time
	if time_since_last < MIN_LLM_REQUEST_INTERVAL:
		time.sleep(MIN_LLM_REQUEST_INTERVAL - time_since_last)

	# Ask LLM with grounded context
	llm = ChatGoogleGenerativeAI(
		model="gemini-2.5-pro",
		api_key=os.getenv("GEMINI_API_KEY"),
		temperature=0.2,
	)

	prompt = (
		"You are a helpful assistant. Use ONLY the provided context to answer.\n"
		"If the answer is not in the context, say you don't know.\n\n"
		f"Context:\n{context_text}\n\n"
		f"Question: {query}\n\n"
		"Answer :"
		)

	try:
		last_llm_request_time = time.time()
		response = llm.invoke(prompt)
		return response.content.strip() if hasattr(response, "content") else str(response)
	except Exception as e:
		if "quota" in str(e).lower() or "429" in str(e):
			return "⚠️ API quota exceeded. Please wait a moment and try again. This is due to free tier limits."
		else:
			return f"⚠️ Error processing your question: {str(e)}"


def delete_pdf_chunks(pdf_name: str) -> int:
	"""Delete all vector chunks in Chroma whose source contains pdf_name.

	Returns number of records deleted (best-effort based on IDs returned).
	"""
	vectorstore = _get_vectorstore()
	# Access underlying collection to use where-filter deletion
	try:
		full_path = os.path.join(PDF_DIR, pdf_name)
		result = vectorstore._collection.delete(where={"source": {"$eq": full_path}})
		if isinstance(result, list):
			return len(result)
		# chroma >=0.5 may return dict
		if isinstance(result, dict) and "ids" in result:
			return len(result.get("ids", []))
	except Exception:
		pass
	return 0