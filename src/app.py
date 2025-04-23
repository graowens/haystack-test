from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder

import pdfplumber
import os
import gradio as gr
import requests

# Load PDF and extract text
def load_pdf_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Load all PDFs
pdf_dir = "./pdfs"
documents = []
for filename in os.listdir(pdf_dir):
    if filename.lower().endswith(".pdf"):
        content = load_pdf_text(os.path.join(pdf_dir, filename))
        documents.append(Document(content=content, meta={"source": filename}))

# Set up Haystack components
document_store = InMemoryDocumentStore()
document_store.write_documents(documents)

retriever = InMemoryBM25Retriever(document_store=document_store)

template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
{{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
prompt_builder = PromptBuilder(template=template)

# Ollama call
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
    return response.json()["response"]

# Ask question pipeline
def ask_question(question: str) -> str:
    retrieved_docs = retriever.run(query=question)["documents"]
    prompt = prompt_builder.run(documents=retrieved_docs, question=question)["prompt"]
    return call_ollama_mistral(prompt)

# Gradio UI
gr.Markdown("# Law Document Concept Prototype v0.1")
gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask something about your PDF lad...",
        label="Question lad?"
    ),
    outputs="text",
    title="PDF QA (Haystack + Ollama)"
).launch(server_name="0.0.0.0", server_port=7860)
