from haystack.dataclasses import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.builders import PromptBuilder

import os
import gradio as gr
import requests
import pytesseract
import pdfplumber
from pdf2image import convert_from_path

# Load PDF and extract text with OCR fallback
def load_pdf_text(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text() or '' for page in pdf.pages)
        if full_text.strip():
            return full_text.strip()
    except Exception as e:
        print(f"pdfplumber failed: {e}")

    print(f"OCR fallback for: {pdf_path}")
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image) + "\n"
    return text.strip()

# Load and parse all PDFs
pdf_dir = "./pdfs"
documents = []
for filename in os.listdir(pdf_dir):
    if filename.lower().endswith(".pdf"):
        content = load_pdf_text(os.path.join(pdf_dir, filename))
        documents.append(Document(content=content, meta={"source": filename}))

# Setup Qdrant document store
document_store = QdrantDocumentStore(
    host="qdrant",
    port=6333,
    embedding_dim=384,
    index="pdf_docs",
    similarity="cosine"
)

# Create document embedder and embed documents
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()
result = doc_embedder.run(documents)
documents = result["documents"]

# Write documents to Qdrant
document_store.write_documents(documents, policy="overwrite")

# Setup retriever (query embedding will be done manually)
retriever = QdrantEmbeddingRetriever(document_store=document_store)

# Query embedder (we use this per question)
query_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
query_embedder.warm_up()

# Prompt template for question answering
template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
{{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
prompt_builder = PromptBuilder(template=template, required_variables=["documents", "question"])


# Function to call Ollama's Mistral model
def call_ollama_mistral(prompt: str) -> str:
    response = requests.post(
        "http://ollama:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "temperature": 0.9,
            "stream": False
        }
    )
    return response.json().get("response", "No response received from Ollama.")

# Full pipeline: embed query -> retrieve -> prompt -> LLM
def ask_question(question: str) -> str:
    # Embed the question directly as a raw string (no dict)
    embedding_result = query_embedder.run(question)
    query_embedding = embedding_result["embedding"]

    # Retrieve documents from Qdrant
    retrieved_docs = retriever.run(query_embedding=query_embedding)["documents"]

    # Build prompt and call the model
    prompt = prompt_builder.run(documents=retrieved_docs, question=question)["prompt"]
    return call_ollama_mistral(prompt)


# Gradio Web UI
gr.Markdown("# Gra Test v0.1 - Readin PDFs and Ask Questions")
gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask something about your PDFs...",
        label="Question"
    ),
    outputs="text",
    title="📄 Gra Test v0.1 Read in PDFs and Ask Questions – Haystack + Qdrant + Ollama"
).launch(server_name="0.0.0.0", server_port=7860)
