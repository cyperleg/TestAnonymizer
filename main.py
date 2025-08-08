from logger_conf import setup_logging

# Configure logging to include timestamp, log level, and message
logging = setup_logging()

import json
from fastapi import FastAPI, HTTPException
import uvicorn
from engine import TextAnonymizer
from text_extractor import FileTextExtractor

app = FastAPI()

# Initialize the anonymization engine and file extractor
anonymizer = TextAnonymizer()

@app.get("/anonymize/text")
async def anonymize_text_endpoint(text: str):
    """
    Anonymize the provided text string.
    Query Parameters:
      - text: Input string to anonymize.
    Returns:
      - JSON containing anonymized_text, statistics, and entity_mapping.
    """
    result = anonymizer.anonymize_text(text)
    return result

@app.get("/anonymize/file")
async def anonymize_file_endpoint(file_path: str):
    """
    Anonymize text extracted from the specified file.
    Query Parameters:
      - file_path: Path to the file (*.txt, *.pdf, *.docx, *.pptx, *.xlsx).
    Returns:
      - JSON containing anonymized_text, statistics, and entity_mapping.
    """
    try:
        text = FileTextExtractor.extract(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File reading error: {e}")
    result = anonymizer.anonymize_text(text)
    return result

@app.get("/deanonymize")
async def deanonymize_endpoint(anonymized_json: str):
    """
    Restore original text from anonymized JSON.
    Query Parameters:
      - anonymized_json: JSON string with fields anonymized_text and entity_mapping.
    Returns:
      - JSON containing restored_text.
    """
    try:
        data = json.loads(anonymized_json)
        text = data.get("anonymized_text")
        mapping = data.get("entity_mapping")
        if text is None or mapping is None:
            raise ValueError("Missing required fields 'anonymized_text' or 'entity_mapping'")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    restored = anonymizer.deanonymize_text(text, mapping)
    return {"restored_text": restored}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
