import os
import uuid
import json
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from services.pdf_utils import extract_text_from_pdf
from services.retriever import DocStore
from services.nlp import LegalNLP

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret-key"
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize NLP and document store
NLP = LegalNLP(
    summarizer_model="facebook/bart-large-cnn",
    qa_model="deepset/roberta-base-squad2",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
)
DOCS = DocStore()

# Frontend route
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Upload PDF and process
@app.post("/api/upload")
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(f.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}_{filename}")
    f.save(save_path)

    try:
        full_text = extract_text_from_pdf(save_path)
        doc_id = str(uuid.uuid4())
        chunks = DOCS.add_document(doc_id, full_text, NLP)

        summary = NLP.summarize(full_text)
        highlights = NLP.highlight_clauses(full_text)
        risks = NLP.flag_risks(full_text)

        return jsonify({
            "document_id": doc_id,
            "summary": summary,
            "highlights": highlights,
            "risks": risks,
            "meta": {"filename": filename, "num_chunks": len(chunks)},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ask question about uploaded document
@app.post("/api/ask")
def ask():
    data: Dict[str, Any] = request.get_json(force=True)
    doc_id = data.get("document_id")
    question = data.get("question", "").strip()

    if not doc_id or not question:
        return jsonify({"error": "document_id and question are required."}), 400
    if doc_id not in DOCS.docs:
        return jsonify({"error": "Unknown document_id"}), 404

    try:
        top_chunks = DOCS.retrieve(doc_id, question, top_k=5)
        context_text = "\n\n".join([c['text'] for c in top_chunks])
        answer, confidence = NLP.answer_question(question, context_text)

        return jsonify({
            "answer": answer,
            "confidence": confidence,
            "context_preview": context_text[:1200]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
