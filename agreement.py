from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import json
import os
from dotenv import load_dotenv
import google.genai as genai
from pdf2image import convert_from_bytes
import pdfplumber
import easyocr
import numpy as np

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise Exception("GOOGLE_API_KEY missing in .env")

# -------------------------------------------------
# Flask app
# -------------------------------------------------
app = Flask(__name__)
CORS(app)  # ✅ CORRECT place (after app creation)

# -------------------------------------------------
# Gemini client
# -------------------------------------------------
client = genai.Client(api_key=GOOGLE_API_KEY)

# -------------------------------------------------
# EasyOCR (CPU ONLY)
# -------------------------------------------------
reader = easyocr.Reader(
    ["en"],       # languages
    gpu=False     # CPU only (no NVIDIA GPU)
)

# -------------------------------------------------
# OCR helpers
# -------------------------------------------------
def extract_text_from_image(file):
    """Extract text from uploaded image using EasyOCR (CPU)."""
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Resize very large images for speed
    image.thumbnail((2000, 2000))

    img_np = np.array(image)

    results = reader.readtext(
        img_np,
        detail=0,
        paragraph=True,
        batch_size=4
    )

    return "\n".join(results).strip()


def extract_text_from_pdf(file):
    """Extract text from PDF (direct text or OCR fallback)."""
    text = ""

    # Read file once
    file_bytes = file.read()

    # 1️⃣ Direct text extraction
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception:
        pass

    # 2️⃣ OCR fallback for scanned PDFs
    if not text.strip():
        images = convert_from_bytes(
            file_bytes,
            dpi=200
        )

        for img in images:
            img = img.convert("RGB")
            img.thumbnail((2000, 2000))
            img_np = np.array(img)

            results = reader.readtext(
                img_np,
                detail=0,
                paragraph=True,
                batch_size=4
            )

            text += "\n".join(results) + "\n"

    return text.strip()

# -------------------------------------------------
# API routes
# -------------------------------------------------
@app.route("/details", methods=["POST"])
def details():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        filename = file.filename.lower()

        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        else:
            text = extract_text_from_image(file)

        if not text:
            return jsonify({"error": "No text extracted"}), 400

        prompt = f"""
You are an information extraction system.

Task:
- Read the text below.
- Identify all distinct data fields (e.g., reference numbers, dates,
  company names, addresses, signatures, amounts, etc.).
- Return ONLY valid JSON.
- Do not invent fields.
- Preserve spelling and formatting exactly.
- If nothing is found, return {{}}.

Text:
{text}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-09-2025",
            contents=[{"text": prompt}],
            config={"response_mime_type": "application/json"}
        )

        try:
            data = json.loads(response.text.strip())
        except json.JSONDecodeError:
            return jsonify({"error": "Model returned invalid JSON"}), 500

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Extraction API is running 🚀",
        "ocr": "EasyOCR",
        "gpu": False,
        "endpoints": {
            "POST /details": "Upload an image or PDF file"
        }
    })

# -------------------------------------------------
# Run app
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
