"""
Text Summarizer Backend - FastAPI + HuggingFace BART
Encoder-decoder architecture (Transformer-based)

Install:
    pip install fastapi uvicorn transformers torch sentencepiece

Run:
    uvicorn summarizer_backend:app --reload --port 8000
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from functools import lru_cache

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, pipeline

try:
    from pydantic import field_validator, model_validator

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover - fallback for Pydantic v1
    from pydantic import root_validator, validator

    PYDANTIC_V2 = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/bart-large-cnn"
FALLBACK_MODEL_NAME = "extractive-frequency-fallback"
MAX_INPUT_TOKENS = 1024  # BART context window

app = FastAPI(
    title="Text Summarizer API",
    description="Encoder-decoder (BART) text summarization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5500",
        "*",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 130
    min_length: int = 30

    if PYDANTIC_V2:

        @field_validator("text")
        @classmethod
        def text_must_not_be_empty(cls, value: str) -> str:
            value = value.strip()
            if not value:
                raise ValueError("text field must not be empty")
            return value

        @field_validator("max_length")
        @classmethod
        def max_length_range(cls, value: int) -> int:
            if not (30 <= value <= 500):
                raise ValueError("max_length must be between 30 and 500")
            return value

        @field_validator("min_length")
        @classmethod
        def min_length_range(cls, value: int) -> int:
            if not (10 <= value <= 200):
                raise ValueError("min_length must be between 10 and 200")
            return value

        @model_validator(mode="after")
        def validate_lengths(self) -> "SummarizeRequest":
            if self.min_length >= self.max_length:
                raise ValueError("min_length must be smaller than max_length")
            return self

    else:

        @validator("text")
        def text_must_not_be_empty(cls, value: str) -> str:
            value = value.strip()
            if not value:
                raise ValueError("text field must not be empty")
            return value

        @validator("max_length")
        def max_length_range(cls, value: int) -> int:
            if not (30 <= value <= 500):
                raise ValueError("max_length must be between 30 and 500")
            return value

        @validator("min_length")
        def min_length_range(cls, value: int) -> int:
            if not (10 <= value <= 200):
                raise ValueError("min_length must be between 10 and 200")
            return value

        @root_validator
        def validate_lengths(cls, values: dict) -> dict:
            min_length = values.get("min_length")
            max_length = values.get("max_length")
            if min_length is not None and max_length is not None and min_length >= max_length:
                raise ValueError("min_length must be smaller than max_length")
            return values


class SummarizeResponse(BaseModel):
    summary: str
    input_tokens: int
    output_tokens: int
    model: str


@lru_cache(maxsize=1)
def get_tokenizer():
    logger.info("Loading tokenizer: %s", MODEL_NAME)
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@lru_cache(maxsize=1)
def get_summarizer():
    logger.info("Loading summarization model: %s", MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model=MODEL_NAME, device=device)


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def fallback_summarize_text(text: str, max_length: int, min_length: int) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return text.strip()

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    if not words:
        return " ".join(sentences[:2])

    stop_words = {
        "about", "after", "again", "almost", "also", "among", "because", "before",
        "being", "between", "could", "every", "first", "found", "from", "have",
        "however", "into", "many", "more", "most", "other", "over", "same",
        "should", "since", "some", "such", "than", "that", "their", "there",
        "these", "they", "this", "those", "through", "under", "very", "were",
        "what", "when", "where", "which", "while", "with", "would",
    }
    frequencies = Counter(word for word in words if word not in stop_words)
    if not frequencies:
        return " ".join(sentences[:2])

    ranked_sentences = []
    for index, sentence in enumerate(sentences):
        sentence_words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())
        if not sentence_words:
            continue
        score = sum(frequencies[word] for word in sentence_words) / len(sentence_words)
        ranked_sentences.append((score, index, sentence))

    ranked_sentences.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[int, str]] = []
    current_word_count = 0
    target_min_words = max(20, min_length // 2)
    target_max_words = max(target_min_words, max_length)

    for _, index, sentence in ranked_sentences:
        sentence_word_count = len(sentence.split())
        if selected and current_word_count + sentence_word_count > target_max_words:
            continue
        selected.append((index, sentence))
        current_word_count += sentence_word_count
        if current_word_count >= target_min_words:
            break

    if not selected:
        selected.append((0, sentences[0]))

    selected.sort(key=lambda item: item[0])
    summary = " ".join(sentence for _, sentence in selected).strip()
    return summary or sentences[0]


def get_runtime_model_name() -> str:
    try:
        get_summarizer()
        get_tokenizer()
        return MODEL_NAME
    except Exception:
        return FALLBACK_MODEL_NAME


def chunk_text(text: str, max_tokens: int = MAX_INPUT_TOKENS) -> list[str]:
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks: list[str] = []
    stride = max_tokens - 50

    for start in range(0, len(tokens), stride):
        chunk_tokens = tokens[start : start + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        if start + max_tokens >= len(tokens):
            break

    return chunks


def summarize_text(text: str, max_length: int, min_length: int) -> str:
    try:
        tokenizer = get_tokenizer()
        summarizer = get_summarizer()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        logger.info("Input token count: %s", len(tokens))

        generation_args = {
            "max_length": max_length,
            "min_length": min_length,
            "do_sample": False,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
        }

        if len(tokens) <= MAX_INPUT_TOKENS:
            result = summarizer(text, **generation_args)
            return result[0]["summary_text"]

        logger.info("Long text detected (%s tokens). Chunking...", len(tokens))
        partial_summaries: list[str] = []

        for index, chunk in enumerate(chunk_text(text), start=1):
            logger.info("Summarizing chunk %s", index)
            result = summarizer(
                chunk,
                max_length=max_length,
                min_length=max(10, min_length // 2),
                do_sample=False,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            partial_summaries.append(result[0]["summary_text"])

        merged = " ".join(partial_summaries)
        final_result = summarizer(merged, **generation_args)
        return final_result[0]["summary_text"]
    except Exception as exc:
        logger.warning(
            "Falling back to local extractive summarizer because model loading/inference failed: %s",
            exc,
        )
        return fallback_summarize_text(text, max_length, min_length)


@app.get("/")
def root():
    return {"status": "running", "model": get_runtime_model_name()}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize_endpoint(req: SummarizeRequest):
    try:
        runtime_model = get_runtime_model_name()
        if runtime_model == MODEL_NAME:
            tokenizer = get_tokenizer()
            input_tokens = len(tokenizer.encode(req.text, add_special_tokens=False))
        else:
            input_tokens = len(req.text.split())
        logger.info("Received request: %s input tokens", input_tokens)

        summary = summarize_text(req.text, req.max_length, req.min_length)

        if runtime_model == MODEL_NAME:
            tokenizer = get_tokenizer()
            output_tokens = len(tokenizer.encode(summary, add_special_tokens=False))
        else:
            output_tokens = len(summary.split())
        logger.info("Summary generated: %s tokens", output_tokens)

        return SummarizeResponse(
            summary=summary,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=runtime_model,
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Inference error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal model error. Check server logs.",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("summarizer_backend:app", host="0.0.0.0", port=8000, reload=True)
