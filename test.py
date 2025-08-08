import re
import time
import psutil
import pytest
from fastapi.testclient import TestClient
from main import app
from engine import TextAnonymizer
from text_extractor import FileTextExtractor

client = TestClient(app)

@pytest.fixture

def sample_text():
    return (
        """
        Dear John Smith and Mary Johnson,
        Thank you for your interest in our services at Acme Corporation.
        Please contact us at info@acme.com or call our office at 555-123-4567.
        Our headquarters are located in New York City, with additional offices
        in San Francisco and London. We serve major clients including
        Microsoft, Google, and Amazon.
        For technical support, reach out to support@acme.com or call
        1-800-SUPPORT. You can also visit our website at www.acme.com.
        Best regards,
        Robert Davis
        CEO, Acme Corporation
        robert.davis@acme.com
        Direct: 555-987-6543
        """
    )

# === BASIC FUNCTIONALITY TESTS ===

def test_simple_text_anonymization(sample_text):
    "Test basic text anonymization via API returns expected keys and types."
    response = client.get("/anonymize/text", params={"text": sample_text})
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"anonymized_text", "statistics", "entity_mapping"}



def test_correct_placeholder_generation(sample_text):
    "Ensure placeholders follow the [LABEL_index] format and map back correctly."
    tz = TextAnonymizer()
    result = tz.anonymize_text(sample_text)
    mapping = result["entity_mapping"]  # list of occurrences

    # Collect unique placeholders and validate their format
    placeholders = {m["placeholder"] for m in mapping}
    for ph in placeholders:
        assert re.match(r"^\[[A-Z]+_\d+\]$", ph)

    # Ensure mapping back yields original text
    restored = tz.deanonymize_text(result["anonymized_text"], mapping)
    assert restored == sample_text


def test_statistics_accuracy(sample_text):
    "Validate that statistics.total_entities equals sum of by_category counts."
    tz = TextAnonymizer()
    result = tz.anonymize_text(sample_text)
    stats = result["statistics"]
    total = stats["total_entities"]
    by_cat = stats["by_category"]
    assert total == sum(by_cat.values())

# === EDGE CASE TESTS ===

def test_empty_text():
    "Anonymizing empty text should return empty outputs."
    tz = TextAnonymizer()
    result = tz.anonymize_text("")
    assert result["anonymized_text"] == ""
    assert result["statistics"]["total_entities"] == 0
    assert result["entity_mapping"] == []


def test_no_entities_text():
    "Text without any identifiable entities should remain unchanged."
    text = "hello world! this is a test without names or numbers."
    tz = TextAnonymizer()
    result = tz.anonymize_text(text)
    assert result["anonymized_text"] == text
    assert result["statistics"]["total_entities"] == 0


def test_duplicate_entities():
    "Duplicate entities should map to the same placeholder."
    text = "John Doe met John Doe at the office."
    tz = TextAnonymizer()
    result = tz.anonymize_text(text)
    mapping = result["entity_mapping"]  # list of occurrences
    placeholders = {m["placeholder"] for m in mapping if m["text"] == "John Doe"}
    assert len(placeholders) == 1  # only one placeholder reused

def test_overlapping_entities():
    """Both 'John Smith' and 'Smithers' should be anonymized with PER placeholders,
    and only these two names should appear as PER entities in the mapping."""
    text = "John Smith and Smithers attended the meeting."
    tz = TextAnonymizer()
    result = tz.anonymize_text(text)
    anon = result["anonymized_text"]
    mapping = result["entity_mapping"]

    # No raw names should remain in the anonymized text
    assert "John Smith" not in anon
    assert "Smithers" not in anon

    # Expect pattern: [PER_x] and [PER_y] attended the meeting.
    assert re.search(r"\[PER_\d+\]\s+and\s+\[PER_\d+\]\s+attended the meeting\.", anon)

    # Mapping should contain exactly these two PER entities
    per_texts = {m["text"] for m in mapping if m["label"] == "PER"}
    assert per_texts == {"John Smith", "Smithers"}

    # There should be exactly 2 unique PER placeholders
    per_placeholders = {m["placeholder"] for m in mapping if m["label"] == "PER"}
    assert len(per_placeholders) == 2

# === MODEL BEHAVIOR TESTS ===

def test_model_failure(monkeypatch):
    "Simulate model failure and ensure exception is propagated."
    tz = TextAnonymizer()
    # Monkeypatch pipeline to raise error
    monkeypatch.setattr(tz, 'pipeline', lambda text: (_ for _ in ()).throw(RuntimeError("model error")))
    with pytest.raises(RuntimeError):
        tz.extract_sensitive_info("Some text")


def test_batch_processing_metrics():
    """
    Print processing time and memory usage for a large text batch without assertions.
    Run with `pytest -s` to see printed metrics.
    """
    tz = TextAnonymizer()
    try:
        text = FileTextExtractor.extract("test.docx") * 10
    except Exception as e:
        pytest.skip(f"Skipping metrics test (sample file missing or unreadable): {e}")

    process = psutil.Process()
    mem_before = process.memory_info().rss
    start = time.perf_counter()
    tz.anonymize_text(text)
    duration = time.perf_counter() - start
    mem_after = process.memory_info().rss
    mem_used_mb = (mem_after - mem_before) / (1024 ** 2)

    print(f"[METRICS] batch(10x docx) time={duration:.3f}s, mem={mem_used_mb:.2f} MB")

    assert duration < 30.0
