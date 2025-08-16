import os
import random
import shutil
import string
from typing import List

import fitz  # PyMuPDF
import dspy
import pytesseract
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DSPy / Perplexity setup ----------
PPLX_API_KEY = os.getenv("PPLX_API_KEY")  # DO NOT hardcode tokens
dspy_lm_ready = False

try:
    if PPLX_API_KEY:
        lm = dspy.LM("perplexity/sonar-pro", api_key=PPLX_API_KEY)
        dspy.configure(lm=lm)
        dspy_lm_ready = True
    else:
        print("⚠️  PPLX_API_KEY not set; analysis endpoints will return 503.")
except Exception as e:
    print(f"⚠️  Failed to initialize DSPy LM: {e}")
    dspy_lm_ready = False


class ContributingValue(BaseModel):
    name: str = Field(description="Name of the contributing financial component")
    value: float = Field(description="Numerical value of the component")


class Calculation(BaseModel):
    calculation_of: str = Field(description="Use exact text")
    contributing_values: List[ContributingValue] = Field(description="Use exact text")
    total_amount: float = Field(description="Sum of all contributing values")
    reasoning: str = Field(description="Short reasoning")
    is_calculation_correct: bool = Field(description="True if math is correct")


class CheckCalculationFaithfulness(dspy.Signature):
    """Extract structured financial calculations from document text."""
    calculations: str = dspy.InputField(desc="Raw text of a document/page")
    are_all_calculations_correct: bool = dspy.OutputField()
    calculations_in_json: List[Calculation] = dspy.OutputField(
        desc="List of structured calculations with reasoning and total"
    )


# Instantiate once if LM is ready
faithfulness = dspy.ChainOfThought(CheckCalculationFaithfulness) if dspy_lm_ready else None


def read_pdf(path: str):
    if not dspy_lm_ready or faithfulness is None:
        raise HTTPException(status_code=503, detail="LLM not configured (PPLX_API_KEY missing or invalid).")

    output = []
    doc = fitz.open(path)
    page_texts = []
    start_page_index = 1

    for i, page in enumerate(doc, start=1):
        page_texts.append(page.get_text())

        if (i % 5 == 0) or (i == len(doc)):
            combined_text = "\n".join(page_texts)
            result = faithfulness(calculations=combined_text)
            output.append({
                "pages": f"{start_page_index}-{i}",
                "text": combined_text,
                "faithfulness_result": result
            })
            page_texts = []
            start_page_index = i + 1

    return output


def read_image(path: str):
    if not dspy_lm_ready or faithfulness is None:
        raise HTTPException(status_code=503, detail="LLM not configured (PPLX_API_KEY missing or invalid).")

    img = Image.open(path)
    text = pytesseract.image_to_string(img)
    result = faithfulness(calculations=text)
    return [{
        "text": text,
        "faithfulness_result": result
    }]


@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    # Ensure tesseract is available at runtime for image OCR
    if shutil.which("tesseract") is None:
        return JSONResponse(
            status_code=500,
            content={"error": "tesseract-ocr not installed in container"},
        )

    upload_dir = "./public"
    os.makedirs(upload_dir, exist_ok=True)
    random_prefix = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    file_path = os.path.join(upload_dir, f"{random_prefix}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.lower().endswith(".pdf"):
        results = read_pdf(file_path)
        return {
            "filename": f"{random_prefix}_{file.filename}",
            "type": "pdf",
            "total_pages": len(results),
            "results": results
        }
    else:
        results = read_image(file_path)
        return {
            "filename": f"{random_prefix}_{file.filename}",
            "type": "image",
            "total_pages": 1,
            "results": results
        }


@app.get("/health/")
async def health_check():
    return {"status": "Healthy", "llm_configured": dspy_lm_ready}


@app.get("/")
async def root():
    return {"ok": True, "message": "python-service3 is running"}


if __name__ == "__main__":
    # Run uvicorn directly from Python (fine for local dev)
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
