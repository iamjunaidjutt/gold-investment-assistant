"""
Gold Investment Estimations Assistant
=====================================
Voice-enabled gold portfolio and valuation advisor powered by:
- Groq Whisper (whisper-large-v3) for high-speed voice transcription
- Google Gemini via OpenAI-compatible endpoint with OpenAI Python SDK
- Gradio for an interactive, modern web and mobile UI

Author: Muhammad Junaid (@iamjunaidjutt)
Deployed on Hugging Face Spaces & GitHub
"""

import os
import json
import logging
from typing import Optional, Tuple
import requests
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# API Keys & Gemini via OpenAI Endpoint Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip("\"'\r\n\t ")
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY", "")).strip("\"'\r\n\t ")
METAL_PRICE_API = os.getenv("METAL_PRICE_API", "").strip("\"'\r\n\t ")
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
).strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gemini-3.1-flash-lite")

# Fallback spot prices (USD per Troy Ounce, 31.1035g)
DEFAULT_GOLD_SPOT_USD = 2750.00
TROY_OUNCE_TO_GRAMS = 31.1034768
TOLA_TO_GRAMS = 11.6638038

# Karat purity factors
PURITY_FACTORS = {
    "24K (99.9% Pure)": 0.999,
    "22K (91.6% Pure)": 0.916,
    "18K (75.0% Pure)": 0.750,
    "14K (58.3% Pure)": 0.583,
}


def fetch_live_gold_price() -> Tuple[float, str]:
    """Fetch live spot gold price in USD or fall back gracefully."""
    # 1. Try with user METAL_PRICE_API on metals.dev
    if METAL_PRICE_API:
        try:
            url = f"https://api.metals.dev/v1/latest?api_key={METAL_PRICE_API}&currency=USD&unit=toz"
            resp = requests.get(url, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                rate = data.get("metals", {}).get("gold")
                if rate and isinstance(rate, (int, float)) and rate > 1000:
                    return float(rate), "Live metals.dev API"
        except Exception as e:
            logger.warning(f"Could not fetch metals.dev rate: {e}")

        # 2. Try GoldAPI if metals.dev fails
        try:
            headers = {"x-access-token": METAL_PRICE_API, "Content-Type": "application/json"}
            resp = requests.get("https://www.goldapi.io/api/XAU/USD", headers=headers, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                price = data.get("price")
                if price and isinstance(price, (int, float)) and price > 1000:
                    return float(price), "Live GoldAPI"
        except Exception as e:
            logger.warning(f"Could not fetch GoldAPI rate: {e}")

    # 3. Public demo fallback
    try:
        url = "https://api.metals.dev/v1/latest?api_key=demo&currency=USD&unit=toz"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            rate = data.get("metals", {}).get("gold")
            if rate and isinstance(rate, (int, float)) and rate > 1000:
                return float(rate), "Demo metals.dev API"
    except Exception as e:
        logger.warning(f"Could not fetch demo rate: {e}")

    return DEFAULT_GOLD_SPOT_USD, "Reference Benchmark"


def transcribe_audio_groq(audio_path: Optional[str]) -> str:
    """Transcribe speech to text using Groq Whisper API (whisper-large-v3)."""
    if not audio_path:
        return ""

    if not GROQ_API_KEY:
        return (
            "Voice query received. (Note: Set GROQ_API_KEY to enable live "
            "whisper-large-v3 transcription.)"
        )

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), file.read()),
                model="whisper-large-v3",
                response_format="text",
                language="en",
                temperature=0.0,
            )
        return str(transcription).strip()
    except Exception as err:
        logger.error(f"Groq Whisper transcription failed: {err}")
        return f"Error transcribing audio: {err}"


def calculate_valuation(
    weight: float,
    unit: str,
    karat: str,
    making_charges_pct: float,
    tax_pct: float,
    custom_spot_price: Optional[float] = None,
) -> dict:
    """Calculate gold valuation based on purity, weight, and market spot price."""
    spot_price = (
        custom_spot_price
        if (custom_spot_price and custom_spot_price > 0)
        else fetch_live_gold_price()[0]
    )

    # Convert weight to grams
    if unit == "Troy Ounces (oz)":
        grams = weight * TROY_OUNCE_TO_GRAMS
    elif unit == "Tolas":
        grams = weight * TOLA_TO_GRAMS
    else:
        grams = weight

    purity = PURITY_FACTORS.get(karat, 0.999)
    fine_gold_grams = grams * purity
    base_gram_price_24k = spot_price / TROY_OUNCE_TO_GRAMS

    metal_value = fine_gold_grams * base_gram_price_24k
    making_charges = metal_value * (making_charges_pct / 100.0)
    tax_value = (metal_value + making_charges) * (tax_pct / 100.0)
    total_cost = metal_value + making_charges + tax_value

    return {
        "spot_price_usd_oz": spot_price,
        "input_weight": weight,
        "unit": unit,
        "weight_grams": round(grams, 3),
        "fine_gold_grams": round(fine_gold_grams, 3),
        "karat": karat,
        "purity_pct": round(purity * 100, 1),
        "metal_value_usd": round(metal_value, 2),
        "making_charges_usd": round(making_charges, 2),
        "tax_usd": round(tax_value, 2),
        "total_cost_usd": round(total_cost, 2),
    }


def analyze_with_ai(valuation_data: dict, user_question: str) -> str:
    """Provide AI investment advisory using Gemini via OpenAI-compatible endpoint."""
    system_prompt = (
        "You are an expert precious metals financial analyst and investment advisor. "
        "Provide concise, realistic, and practical gold valuation analysis and asset allocation advice."
    )

    user_prompt = f"""
The user has estimated a gold holding or prospective purchase with the following verified valuation details:

- Total Weight: {valuation_data['input_weight']} {valuation_data['unit']} ({valuation_data['weight_grams']} grams)
- Karat Purity: {valuation_data['karat']} ({valuation_data['purity_pct']}% pure)
- Pure Gold Content: {valuation_data['fine_gold_grams']} grams
- Spot Price Benchmark: ${valuation_data['spot_price_usd_oz']:,.2f} / troy oz
- Raw Metal Value: ${valuation_data['metal_value_usd']:,.2f}
- Making Charges: ${valuation_data['making_charges_usd']:,.2f}
- Taxes / Duties: ${valuation_data['tax_usd']:,.2f}
- Total Estimated Cost: ${valuation_data['total_cost_usd']:,.2f}

User Query/Context:
"{user_question if user_question.strip() else 'Provide an investment analysis and asset allocation advice for this gold purchase.'}"

Provide a concise, professional, and practical evaluation covering:
1. **Valuation Breakdown Assessment**: Is the premium (making charges + tax) within reasonable limits (typically under 8-12% for bullion vs 15-25% for jewelry)?
2. **Gold Investment Strategy**: Physical bullion vs paper gold (ETFs/sovereign gold bonds) considerations.
3. **Liquidity & Resale Outlook**: Expected spread upon liquidation and purity retention.
4. **Actionable Recommendation**: Clear final verdict in simple, direct language.

Format cleanly using Markdown headings, bullet points, and bold text. Keep it realistic and direct.
"""

    if not GEMINI_API_KEY:
        return f"""### 📊 Investment Assessment (Reference Mode)
- **Estimated Total Value:** `${valuation_data['total_cost_usd']:,.2f} USD`
- **Pure Gold Weight:** `{valuation_data['fine_gold_grams']} grams`
- **Spread & Premium:** Making charges (${valuation_data['making_charges_usd']:.2f}) and tax (${valuation_data['tax_usd']:.2f}) add a combined premium over spot value.

> **Note:** Set `GEMINI_API_KEY` (or `OPENAI_API_KEY`) to enable real-time Gemini LLM investment advisory and macroeconomic trend analysis via OpenAI endpoint.
"""

    try:
        client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
        except Exception as primary_err:
            fallback_model = "gemini-2.5-flash" if OPENAI_MODEL != "gemini-2.5-flash" else "gemini-1.5-flash"
            logger.warning(f"Primary model {OPENAI_MODEL} failed ({primary_err}). Trying fallback {fallback_model}...")
            response = client.chat.completions.create(
                model=fallback_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
        return response.choices[0].message.content or "No response received."
    except Exception as err:
        logger.error(f"Gemini OpenAI endpoint generation error: {err}")
        return f"Unable to generate AI insights at this time: {err}"


def process_assistant(
    audio_file,
    text_input,
    weight,
    unit,
    karat,
    making_charges,
    taxes,
    custom_spot,
):
    """Main Gradio processing pipeline handling speech, calculation, and AI response."""
    user_query = text_input or ""

    if audio_file:
        transcription = transcribe_audio_groq(audio_file)
        if transcription and not transcription.startswith("Error"):
            user_query = transcription

    valuation = calculate_valuation(
        weight=float(weight or 10.0),
        unit=unit,
        karat=karat,
        making_charges_pct=float(making_charges or 0.0),
        tax_pct=float(taxes or 0.0),
        custom_spot_price=float(custom_spot) if custom_spot else None,
    )

    summary_md = f"""
### 💰 Valuation Summary
| Metric | Value |
| :--- | :--- |
| **Gross Weight** | **{valuation['weight_grams']} g** ({valuation['input_weight']} {valuation['unit']}) |
| **Purity (Karat)** | **{valuation['karat']}** ({valuation['purity_pct']}%) |
| **Fine (Pure) Gold** | **{valuation['fine_gold_grams']} g** |
| **Spot Benchmark** | **${valuation['spot_price_usd_oz']:,.2f} / oz** |
| **Raw Metal Value** | **${valuation['metal_value_usd']:,.2f}** |
| **Making / Fabrication** | **${valuation['making_charges_usd']:,.2f}** |
| **Estimated Tax / VAT** | **${valuation['tax_usd']:,.2f}** |
| **Total Estimated Price** | **${valuation['total_cost_usd']:,.2f} USD** |
"""

    ai_analysis = analyze_with_ai(valuation, user_query)
    return user_query, summary_md, ai_analysis


# ============================================================================
# Gradio UI Design
# ============================================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.header-box {
    text-align: center;
    padding: 24px 16px;
    background: linear-gradient(135deg, #0b0f17 0%, #171f2f 100%);
    border-radius: 14px;
    border: 1px solid rgba(234, 179, 8, 0.3);
    margin-bottom: 24px;
}
.header-box h1 {
    color: #facc15;
    margin: 0 0 8px 0;
    font-size: 1.85rem;
}
.header-box p {
    color: #94a3b8;
    margin: 0;
    font-size: 0.95rem;
}
"""

with gr.Blocks(title="Gold Investment Estimations Assistant", css=CUSTOM_CSS) as demo:
    with gr.Column(elem_classes=["header-box"]):
        gr.Markdown(
            """
# 🪙 Gold Investment Estimations Assistant
**Voice-Enabled AI Portfolio & Valuation Advisor**
Powered by **Google Gemini (via OpenAI Endpoint)** • **Groq Whisper (whisper-large-v3)** • **Gradio**
"""
        )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ 1. Voice or Text Query")
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Voice Input (Groq Whisper-large-v3)",
            )
            text_input = gr.Textbox(
                label="Query or Custom Strategy Notes",
                placeholder="e.g., Should I buy 22K jewelry or 24K cast bars for long-term hedge?",
                lines=2,
            )

            gr.Markdown("### ⚖️ 2. Gold Holding Details")
            weight_input = gr.Number(value=10.0, label="Weight", precision=2)
            unit_input = gr.Radio(
                choices=["Grams (g)", "Tolas", "Troy Ounces (oz)"],
                value="Grams (g)",
                label="Unit of Measurement",
            )
            karat_input = gr.Dropdown(
                choices=list(PURITY_FACTORS.keys()),
                value="24K (99.9% Pure)",
                label="Karat / Purity Grade",
            )

            with gr.Accordion("Advanced Pricing Adjustments", open=False):
                making_charges = gr.Slider(
                    minimum=0,
                    maximum=30,
                    value=3.0,
                    step=0.5,
                    label="Making / Fabrication Charges (%)",
                )
                taxes = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=0.0,
                    step=0.5,
                    label="Local Sales Tax / VAT (%)",
                )
                custom_spot = gr.Number(
                    value=DEFAULT_GOLD_SPOT_USD,
                    label="Custom Spot Price ($/oz, optional)",
                )

            calc_btn = gr.Button("Calculate Valuation & Analyze with AI", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📋 3. Results & AI Advisory")
            transcribed_text = gr.Textbox(
                label="Transcribed / Active Query",
                interactive=False,
            )
            valuation_output = gr.Markdown()
            ai_output = gr.Markdown()

    calc_btn.click(
        fn=process_assistant,
        inputs=[
            audio_input,
            text_input,
            weight_input,
            unit_input,
            karat_input,
            making_charges,
            taxes,
            custom_spot,
        ],
        outputs=[transcribed_text, valuation_output, ai_output],
    )

    gr.Markdown(
        """
---
*Engineered by **Muhammad Junaid** ([@iamjunaidjutt](https://github.com/iamjunaidjutt)) • Built for fast precious metal estimations with speech-to-text and generative AI analysis.*
"""
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
