import re
import logging
import time
import tracemalloc
from functools import wraps
from transformers import pipeline
from langchain.text_splitter import MarkdownTextSplitter
import string
import torch
from text_extractor import FileTextExtractor

# Configure logging to output timestamp, level and message
logger = logging.getLogger("app")

def log_performance(func):
    """Decorator to log execution time and memory usage of methods."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting '{func.__name__}'")
        tracemalloc.start()
        start_time = time.time()
        start_mem, _ = tracemalloc.get_traced_memory()
        result = func(*args, **kwargs)
        end_time = time.time()
        end_mem, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = end_time - start_time
        used_mem = end_mem - start_mem
        logger.info(
            f"Finished '{func.__name__}' in {elapsed:.4f}s, memory used: {used_mem/1024:.2f} KiB"
        )
        return result
    return wrapper

class TextAnonymizer:
    """
    Class for anonymizing and deanonymizing text by replacing sensitive entities
    with placeholders and restoring the original text.
    """
    EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    PHONE_REGEX = re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{1,4}\)?[-.\s]?){1,4}\d{1,4}"
    )

    def __init__(self):
        """Initialize NER pipeline and text splitter."""
        logger.info(f"Device used: {'GPU' if torch.cuda.is_available() else 'CPU'}")

        self.pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=150,
            chunk_overlap=0
        )
        self.labels = ("PER", "ORG", "LOC", "EMAIL", "PHONE", "MISC")

    def _collect_spans(self, text: str) -> list:
        """
        Run regex and raw NER to collect entity spans (label, start, end).
        """
        spans = []
        # Emails
        for m in self.EMAIL_REGEX.finditer(text):
            spans.append({"label":"EMAIL","start":m.start(),"end":m.end()})
        # Phones
        for m in self.PHONE_REGEX.finditer(text):
            norm = m.group().strip()
            if len(re.sub(r"\D","",norm))>=6:
                spans.append({"label":"PHONE","start":m.start(),"end":m.end()})
        # NER tokens
        for ent in self.pipeline(text):
            spans.append({"label": ent["entity_group"],
                          "start": ent["start"],
                          "end": ent["end"]})
        return spans

    def _merge_spans(self, spans: list, text: str) -> list:
        """
        Merge overlapping or adjacent spans into full entity spans.
        Returns list of dicts with label, start, end, and text.
        """
        if not spans:
            return []
        # Sort by start index
        spans_sorted = sorted(spans, key=lambda s: (s['start'], -s['end']))
        merged = []
        current = spans_sorted[0].copy()
        for span in spans_sorted[1:]:
            if span['start'] <= current['end']:
                # merge
                current['end'] = max(current['end'], span['end'])
            else:
                # finalize current
                current['text'] = text[current['start']:current['end']]
                merged.append(current)
                current = span.copy()
        # append final
        current['text'] = text[current['start']:current['end']]
        merged.append(current)
        # ensure valid labels
        for e in merged:
            if e['label'] not in self.labels:
                e['label'] = 'MISC'
        return merged

    @log_performance
    def extract_sensitive_info(self, text: str) -> dict:
        """
        Extract spans and return dict of lists per label with start/end/text.
        """
        spans = self._collect_spans(text)
        merged = self._merge_spans(spans, text)
        result = {lbl: [] for lbl in self.labels}
        for e in merged:
            result[e['label']].append({
                'text': e['text'],
                'start': e['start'],
                'end': e['end']
            })
        return result

    @log_performance
    def anonymize_text(self, text: str) -> dict:
        """
        Anonymize full text by chunked span collection, merging, and placeholder insertion.
        Returns anonymized_text, statistics, and entity_mapping with positions.
        """
        # collect and merge spans from chunks with offsets
        spans = []
        pointer = 0
        for chunk in self.text_splitter.split_text(text):
            idx = text.find(chunk, pointer)
            if idx < 0:
                idx = pointer
            for s in self._collect_spans(chunk):
                s['start'] += idx
                s['end'] += idx
                spans.append(s)
            pointer = idx + len(chunk)
        merged = self._merge_spans(spans, text)
        # assign unique placeholders
        counters = {lbl: 1 for lbl in self.labels}
        placeholder_map = {}
        mapping = []
        for e in merged:
            key = (e['text'], e['label'])
            if key not in placeholder_map:
                ph = f"[{e['label']}_{counters[e['label']]}]"
                placeholder_map[key] = ph
                counters[e['label']] += 1
            else:
                ph = placeholder_map[key]
            mapping.append({
                'placeholder': ph,
                'text': e['text'],
                'start': e['start'],
                'end': e['end'],
                'label': e['label']
            })
        # build anonymized text by slicing full text
        anonymized_parts = []
        last = 0
        for m in sorted(mapping, key=lambda x: x['start']):
            anonymized_parts.append(text[last:m['start']])
            anonymized_parts.append(m['placeholder'])
            last = m['end']
        anonymized_parts.append(text[last:])
        anonymized_text = ''.join(anonymized_parts)
        # statistics on unique placeholders
        by_cat = {lbl: counters[lbl] - 1 for lbl in self.labels}
        total = sum(by_cat.values())
        return {
            'anonymized_text': anonymized_text,
            'statistics': {'total_entities': total, 'by_category': by_cat},
            'entity_mapping': mapping
        }


    @log_performance
    def deanonymize_text(self, anonymized_text: str, entity_mapping: list) -> str:
        """
        Restore original text by replacing placeholders using mapping list,
        preserving original positions.
        """
        text = anonymized_text
        # sort mapping by descending placeholder length
        for m in sorted(entity_mapping, key=lambda x: len(x['placeholder']), reverse=True):
            ph = m['placeholder']
            orig = m['text']
            text = text.replace(ph, orig)
        return text

if __name__ == "__main__":
    sample_text = """
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

    # sample_text = FileTextExtractor.extract("test.docx")

    anonymizer = TextAnonymizer()

    anonymized = anonymizer.anonymize_text(sample_text)
    print(anonymized["entity_mapping"])
    print(anonymized["statistics"])
    print(anonymized["anonymized_text"])

    restored = anonymizer.deanonymize_text(
        anonymized["anonymized_text"], anonymized["entity_mapping"]
    )
    print(restored)
