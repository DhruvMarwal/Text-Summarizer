# Distill — AI Text Summarizer

<p align="center">
  <img src="https://img.shields.io/badge/Model-facebook%2Fbart--large--cnn-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square"/>
  <img src="https://img.shields.io/badge/Frontend-Vanilla%20HTML-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/ATML-Lab%206-8A2BE2?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
</p>

<p align="center">
  A clean, no-framework web app that runs abstractive text summarisation using the BART encoder–decoder transformer.<br/>
  Built for ATML Lab 6 to demonstrate how the encoder–decoder architecture scales from small LSTM experiments to large pre-trained transformers.
</p>

---

## What it does

You paste any long article, paper, or block of text. The BART model reads it through its encoder, cross-attention focuses on what matters, and the decoder writes a fluent abstractive summary — not just extracted sentences, but rephrased, compressed prose.

The interface shows you the full pipeline as it runs, live compression stats (words in vs words out), token counts, and lets you tune how long the summary should be.

---

## Screenshots

<p align="center">
  <img src="screenshots/summarizer.png" alt="Summarizer tool" width="100%"/>
  <br/><em>Summarizer tab — paste text, adjust length, get a summary</em>
</p>

<br/>

<p align="center">
  <img src="screenshots/results.png" alt="Model results" width="100%"/>
  <br/><em>Model Results tab — Lab 6 training summary, architecture diagram, BLEU scores</em>
</p>

---

## Project structure

```
distill/
├── summarizer_backend.py      # FastAPI server — loads BART, exposes /summarize
├── distill.html               # Frontend — open directly in browser, no build step
├── screenshots/               # Add your own after running
│   ├── summarizer.png
│   └── results.png
└── README.md
```

No `node_modules`. No `npm install`. No build tools. Just Python and a browser.

---

## Tech stack

| Layer | Technology |
|---|---|
| Model | `facebook/bart-large-cnn` via HuggingFace Transformers |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Fallback | Extractive frequency-based summariser (works offline) |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/distill.git
cd distill
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn transformers torch sentencepiece
```

### 3. Start the backend

```bash
python -m uvicorn summarizer_backend:app --port 8000
```

> **First run:** BART (~1.6 GB) downloads automatically from HuggingFace. Takes 5–10 min once, then it's cached.

### 4. Open the frontend

Double-click `distill.html` — no local server needed.

Verify the backend is alive at `http://localhost:8000/health`:

```json
{"status": "ok"}
```

---

## Usage

1. Paste any long-form text into the **Input** panel, or click **load demo**
2. Adjust **Max / Min length** sliders to control verbosity
3. Click **Summarize** or press `Ctrl + Enter`
4. Watch the pipeline animation — Encoder → Attention → Decoder → Output
5. Copy the result with the **copy** button

Texts over ~1000 tokens are automatically chunked, summarised in parts, then merged and re-summarised in a final pass.

---

## API

**`POST /summarize`**

```json
// Request
{
  "text": "your long text here",
  "max_length": 130,
  "min_length": 30
}

// Response
{
  "summary": "concise summary...",
  "input_tokens": 312,
  "output_tokens": 87,
  "model": "facebook/bart-large-cnn"
}
```

Other endpoints: `GET /` (status), `GET /health` (health check).

---

## How BART works

BART is a **denoising autoencoder** pre-trained by corrupting text and learning to reconstruct it, then fine-tuned on CNN/DailyMail for summarisation.

```
Input text
    │
    ▼
[ Tokeniser ]     BPE tokenisation, up to 1024 tokens
    │
    ▼
[ Encoder ]       Bidirectional self-attention — every token attends to every other
    │
    ▼ cross-attention
[ Decoder ]       Auto-regressive generation, beam search (n=4)
    │
    ▼
Summary
```

If the model fails to load (no internet / low RAM), the backend silently falls back to a local extractive summariser. The `model` field in the response will say `extractive-frequency-fallback` in that case.

---

## Lab 6 — LSTM experiments

The notebook `atml-lab6_task1-3.ipynb` covers three tasks that demonstrate the same enc–dec principle at LSTM scale:

| Task | Dataset | Objective |
|---|---|---|
| Task 1 | Hindi–English Truncated Corpus | English → Hindi seq2seq translation |
| Task 2 | Same | BLEU score evaluation (smoothed, n=30 samples) |
| Task 3 | opus_books en-es (15k pairs) | English → Spanish seq2seq translation |

**Architecture:** Embedding (128d) → LSTM Encoder (256 units) → context vector → LSTM Decoder (256 units) → Dense Softmax

**BLEU score (Task 2):** ~0.18–0.28, typical for vanilla LSTM seq2seq without attention. Bahdanau attention would push this meaningfully higher.

---

## Warnings you might see (all harmless)

| Warning | Why | Fix needed? |
|---|---|---|
| `symlinks warning` | Windows symlink restriction | No — caching still works |
| `hf_xet not installed` | Optional faster HuggingFace protocol | No |
| `favicon.ico 404` | No icon file | No |

---

*ATML Lab 6 · Encoder–Decoder Architectures · 2025*
