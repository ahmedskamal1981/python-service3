import fitz
import dspy
import pytesseract
from PIL import Image
from typing import List
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
import random
import string
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure DSPy with Perplexity
lm = dspy.LM('perplexity/sonar-pro', api_key='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')
dspy.configure(lm=lm)

# Test Perplexity API connection
def test_perplexity_connection():
    try:
        # Create a simple test signature
        class TestSignature(dspy.Signature):
            """Test the API connection."""
            prompt: str = dspy.InputField()
            response: str = dspy.OutputField()

        # Create a test module
        test_module = dspy.ChainOfThought(TestSignature)
        
        # Run the test
        result = test_module(prompt="Say 'Hello, this is a test' in one word.")
        print("✅ Perplexity API Test Successful!")
        print(f"Response: {result.response}")
        return True
    except Exception as e:
        print("❌ Perplexity API Test Failed!")
        print(f"Error: {str(e)}")
        return False

# Run the test on startup
test_perplexity_connection()

class ContributingValue(BaseModel):
    name: str = Field(description="Name of the contributing financial component")
    value: float = Field(description="Numerical value of the component")

class Calculation(BaseModel):
    calculation_of: str = Field(description="The name of the total being calculated. Use exact same value as it is in the text.")
    contributing_values: List[ContributingValue] = Field(description="List of items contributing to this calculation. Use exact same value as it is in the text.")
    total_amount: float = Field(description="The total sum of all contributing values")
    reasoning: str = Field(description="Short reasoning behind this total")
    is_calculation_correct: bool = Field(description="Calculate total_amount based on contributing values mark it as true if correct and false if calculation is not right mathematically")

class CheckCalculationFaithfulness(dspy.Signature):
    """Extract structured financial calculations from document text."""

    calculations: str = dspy.InputField(desc="Raw text content of one page of a financial document")
    are_all_calculations_correct: bool = dspy.OutputField()
    calculations_in_json: List[Calculation] = dspy.OutputField(desc="List of structured calculations with reasoning and total")

faithfulness = dspy.ChainOfThought(CheckCalculationFaithfulness)

def read_pdf(path):
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

def read_image(path):
    img = Image.open(path)
    text = pytesseract.image_to_string(img)
    result = faithfulness(calculations=text)
    return [{
        "text": text,
        "faithfulness_result": result
    }]

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
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
    return {
            "status": "Healthy",
    }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8010))
    uvicorn.run(
        "YOURFILENAME:app",   # Replace YOURFILENAME with this script’s name without .py
        host="0.0.0.0",
        port=port
    )
