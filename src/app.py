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
import re
from PIL import Image
from pdf2image import convert_from_path
from more_itertools import chunked
from jinja2 import Template

# Load PDF and extract text with OCR fallback
def extract_text_from_file(file_path):
    try:
        if file_path.lower().endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                full_text = "\n".join(page.extract_text() or '' for page in pdf.pages)
            if full_text.strip():
                return full_text.strip()
        else:  # assume it's an image
            image = Image.open(file_path)
            return pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Text extraction failed for {file_path}: {e}")
        return ""

# Load and parse all files
pdf_dir = "./pdfs"
documents = []
for filename in os.listdir(pdf_dir):
    if filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        content = extract_text_from_file(os.path.join(pdf_dir, filename))
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
query_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
query_embedder.warm_up()

# Prompt for regular questions
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

# LLM call
def call_ollama_mistral(prompt: str) -> str:
    response = requests.post(
        "http://ollama:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "temperature": 0.9,
            "stream": False
        }
    )
    return response.json().get("response", "No response received from Ollama.")

# Q&A function
def ask_question(question: str) -> str:
    embedding_result = query_embedder.run(question)
    query_embedding = embedding_result["embedding"]
    retrieved_docs = retriever.run(query_embedding=query_embedding)["documents"]
    prompt = prompt_builder.run(documents=retrieved_docs, question=question)["prompt"]
    return call_ollama_mistral(prompt)

# Hybrid extraction function
def extract_names_and_emails() -> str:
    # Pull all documents
    all_docs = document_store.filter_documents({})

    # Extract candidate lines using regex
    candidates = []
    for doc in all_docs:
        for line in doc.content.splitlines():
            if re.search(r"[\w.-]+\s*\[at\]|@\s*[\w.-]+", line, re.IGNORECASE):
                candidates.append(line.strip())

    if not candidates:
        return "No likely name/email lines found."

    # Batch and format prompt
    results = []
    j2_template = Template("""
Please normalize the following name/email/job title candidates. Replace [at] with @. If a job title is mentioned near the name or email, include it.

Return the results in this format:
- Full Name — Job Title <email@example.com>

Candidates:
{% for line in lines %}- {{ line }}
{% endfor %}
""")

    for batch in chunked(candidates, 5):
        prompt = j2_template.render(lines=batch)
        print(f"\n--- Prompt ---\n{prompt}\n")
        result = call_ollama_mistral(prompt)
        results.append(result)

    return "\n".join(results)

# Image description from OCR and LLM
def describe_image(image_path):
    try:
        # Ensure it's a supported format and in RGB mode
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            text = pytesseract.image_to_string(img)

        if not text.strip():
            text = "No visible text found. Describe the image based on its visual features."

        prompt = f"Describe this image based on the following extracted text and any visual clues you can infer:\n\n{text}"
        return call_ollama_mistral(prompt)

    except Exception as e:
        return f"Failed to process image: {str(e)}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Gra Test v0.2 – Ask or Extract Emails (Hybrid)")

    with gr.Row():
        question_box = gr.Textbox(lines=2, placeholder="Ask something about your PDFs...", label="Ask a Question")
        answer_box = gr.Textbox(label="Answer")

    ask_btn = gr.Button("Ask")
    extract_btn = gr.Button("Extract Names & Emails")

    ask_btn.click(fn=ask_question, inputs=question_box, outputs=answer_box)
    extract_btn.click(fn=extract_names_and_emails, inputs=[], outputs=answer_box)

    gr.Markdown("## Upload an Image to Get a Description")

    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload an Image")
        image_description = gr.Textbox(label="Image Description")

    image_btn = gr.Button("Describe Image")
    image_btn.click(fn=describe_image, inputs=image_input, outputs=image_description)

    demo.launch(server_name="0.0.0.0", server_port=7860)

