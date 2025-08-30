---
title: Gold Investment Estimations Assistant
emoji: 🪙
colorFrom: yellow
colorTo: yellow
sdk: static
pinned: false
license: mit
short_description: Gold portfolio and investment valuation advisor
---

# Gold Investment Estimations Assistant

A voice-enabled gold valuation and portfolio investment advisor built with the **OpenAI API Client**, **Groq Whisper (`whisper-large-v3`)**, and **Gradio**, deployed on **Hugging Face Spaces**.

---

## Overview

The **Gold Investment Estimations Assistant** helps users quickly estimate the true market value of gold holdings across different karat purities (24K, 22K, 18K, 14K) and weight units (Grams, Tolas, Troy Ounces). It integrates voice input using Groq's high-speed Whisper implementation and generates structured macroeconomic investment insights using the OpenAI API client (compatible with OpenAI GPT models and OpenAI-compatible endpoints like Gemini).

### Key Features
- **Voice-to-Text via Groq Whisper**: Hands-free voice queries processed via `whisper-large-v3` with low latency.
- **Accurate Karat & Unit Valuation**: Mathematical calculations across 24K (99.9%), 22K (91.6%), 18K (75.0%), and 14K (58.3%) purity grades with support for Grams, Tolas, and Troy Ounces.
- **Custom Price Adjustments**: Flexible adjustments for making/fabrication charges, local sales taxes/VAT, and custom benchmark spot prices.
- **AI Investment Advisory via OpenAI Client**: Generates personalized analysis comparing physical bullion vs paper gold (ETFs), liquidity spreads, and asset allocation strategies.
- **Hugging Face Spaces Ready**: Pre-configured Gradio interface ready for one-click deployment to Hugging Face Spaces.

---

## System Architecture

```
User Voice / Text Query
          │
          ▼
┌─────────────────────────────────┐
│     Gradio Web Interface        │
│   (Inputs, Audio, Sliders)      │
└───────────────┬─────────────────┘
                │
        Audio?  ├───────────────┐
                ▼               ▼
      ┌──────────────────┐  ┌──────────────────┐
      │   Groq Whisper   │  │  Valuation Math  │
      │(whisper-large-v3)│  │ (Purity, Making, │
      └─────────┬────────┘  │   Taxes, Spot)   │
                │           └────────┬─────────┘
                ▼                    │
     [Transcribed Prompt]            │
                │                    │
                └────────┬───────────┘
                         ▼
        ┌─────────────────────────────────┐
        │      OpenAI Client API          │
        │(GPT-4o-mini / Gemini-compatible)│
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │  Structured Investment Advice   │
        │ & Real-time Valuation Breakdown │
        └─────────────────────────────────┘
```

---

## Local Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/iamjunaidjutt/gold-investment-assistant.git
cd gold-investment-assistant
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY="your_groq_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
# Optional: OpenAI-compatible endpoint (e.g. Gemini OpenAI endpoint)
# OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
# OPENAI_MODEL="gemini-1.5-flash"
```

### 5. Launch the Gradio application
```bash
python app.py
```
Open your browser at `http://localhost:7860` to use the assistant.

---

## Hugging Face Spaces Deployment

- **Live Space URL:** [huggingface.co/spaces/junaid48/gold-investment-assistant](https://huggingface.co/spaces/junaid48/gold-investment-assistant)
- **GitHub Repository:** [github.com/iamjunaidjutt/gold-investment-assistant](https://github.com/iamjunaidjutt/gold-investment-assistant)

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space) under `junaid48`.
2. Select **Gradio** as the Space SDK.
3. Push this repository to your Hugging Face Space git remote (`space`).
4. Add your `GROQ_API_KEY` and `GEMINI_API_KEY` in **Space Settings > Secrets**.

---

## Author

- **Muhammad Junaid** — [@iamjunaidjutt](https://github.com/iamjunaidjutt)
- Portfolio: [iamjunaidjutt.vercel.app](https://iamjunaidjutt.vercel.app)
