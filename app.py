"""
Gradio Web Application for Punjabi-English Neural Machine Translation (NMT)
Supports bi-directional translation (Punjabi ↔ English) using NLLB-200 models.
"""

import time
import os
import gradio as gr

# Global variables for lazy loading model & tokenizer
tokenizer = None
model = None
current_model_path = None

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-pa-en"
DEFAULT_PA_TO_EN_MODEL = "Helsinki-NLP/opus-mt-pa-en"
DEFAULT_EN_TO_PA_MODEL = "Helsinki-NLP/opus-mt-en-pa"

LANG_CODES = {
    "Punjabi (ਪੰਜਾਬੀ)": "pan_Guru",
    "English": "eng_Latn"
}


def load_nmt_model(model_name_or_path: str):
    """Load model and tokenizer lazily."""
    global tokenizer, model, current_model_path
    
    if model and current_model_path == model_name_or_path:
        return model, tokenizer

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        print(f"Loading NMT Model from: {model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        current_model_path = model_name_or_path
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_name_or_path}: {str(e)}")


def translate_text(input_text: str, source_lang: str, target_lang: str, model_path: str = DEFAULT_MODEL):
    """Translate text between Punjabi and English."""
    if not input_text or not input_text.strip():
        return "", "Please enter text to translate."

    if source_lang == target_lang:
        return input_text, "Source and Target languages are identical."

    is_pa_to_en = (source_lang == "Punjabi (ਪੰਜਾਬੀ)")

    # Resolve default model based on direction if user hasn't supplied a specific path/hub ID
    if not model_path or not model_path.strip() or model_path.strip() == DEFAULT_MODEL:
        model_to_use = DEFAULT_PA_TO_EN_MODEL if is_pa_to_en else DEFAULT_EN_TO_PA_MODEL
    else:
        model_to_use = model_path.strip()

    start_time = time.time()
    try:
        m, tok = load_nmt_model(model_to_use)
        
        is_nllb = "nllb" in model_to_use.lower()
        
        gen_kwargs = {"max_length": 256}
        
        if is_nllb:
            src_code = LANG_CODES.get(source_lang, "pan_Guru")
            tgt_code = LANG_CODES.get(target_lang, "eng_Latn")
            tok.src_lang = src_code
            inputs = tok(input_text, return_tensors="pt", padding=True, max_length=256, truncation=True)
            tgt_lang_id = tok.lang_code_to_id.get(tgt_code) if hasattr(tok, 'lang_code_to_id') and tgt_code in tok.lang_code_to_id else None
            if tgt_lang_id is not None:
                gen_kwargs["forced_bos_token_id"] = tgt_lang_id
        else:
            # Standard MarianMT (Helsinki-NLP) model
            inputs = tok(input_text, return_tensors="pt", padding=True, max_length=256, truncation=True)

        translated_tokens = m.generate(**inputs, **gen_kwargs)
        result = tok.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        elapsed = time.time() - start_time
        status_info = f"Translated in {elapsed:.2f} seconds using model '{model_to_use}'."
        return result, status_info

    except Exception as e:
        return "", f"Error during translation: {str(e)}"


# Build Gradio UI Interface
def create_ui():
    custom_css = """
    .main-title { text-align: center; color: #1E3A8A; font-family: 'Inter', sans-serif; margin-bottom: 0.5rem; }
    .sub-title { text-align: center; color: #4B5563; font-size: 1.1rem; margin-bottom: 1.5rem; }
    .translate-btn { background-color: #2563EB !important; color: white !important; font-weight: bold !important; }
    """

    with gr.Blocks(title="Punjabi-English NMT Translator", css=custom_css) as demo:
        gr.Markdown("<h1 class='main-title'>ੴ Punjabi ↔ English Neural Machine Translation</h1>")
        gr.Markdown("<p class='sub-title'>State-of-the-Art Machine Translation powered by Helsinki-NLP & OmniBind Corpus</p>")

        with gr.Row():
            with gr.Column():
                source_lang_dropdown = gr.Dropdown(
                    choices=["Punjabi (ਪੰਜਾਬੀ)", "English"],
                    value="Punjabi (ਪੰਜਾਬੀ)",
                    label="Source Language"
                )
                input_text_box = gr.Textbox(
                    lines=5,
                    placeholder="Enter Punjabi or English text here...",
                    label="Input Text"
                )
                
            with gr.Column():
                target_lang_dropdown = gr.Dropdown(
                    choices=["Punjabi (ਪੰਜਾਬੀ)", "English"],
                    value="English",
                    label="Target Language"
                )
                output_text_box = gr.Textbox(
                    lines=5,
                    placeholder="Translation will appear here...",
                    label="Translated Text",
                    interactive=False
                )

        status_bar = gr.Markdown("Ready for translation.")

        with gr.Row():
            translate_button = gr.Button("Translate Text 🚀", elem_classes=["translate-btn"])
            swap_button = gr.Button("Swap Languages 🔄")

        with gr.Accordion("Advanced Settings & Fine-Tuned Checkpoint", open=False):
            model_path_input = gr.Textbox(
                value=DEFAULT_MODEL,
                label="Model Path / HuggingFace Checkpoint",
                info="Provide path to local fine-tuned checkpoint (e.g., './punjabi_english_nllb_final') or HuggingFace hub id."
            )

        gr.Examples(
            examples=[
                ["ਮੈਂ ਪੰਜਾਬੀ ਭਾਸ਼ਾ ਸਿੱਖ ਰਿਹਾ ਹਾਂ।", "Punjabi (ਪੰਜਾਬੀ)", "English"],
                ["ਅਦਾਲਤ ਨੇ ਇਸ ਮਾਮਲੇ ਵਿਚ ਆਪਣਾ ਫੈਸਲਾ ਸੁਣਾਇਆ।", "Punjabi (ਪੰਜਾਬੀ)", "English"],
                ["Welcome to the Punjabi English machine translation project.", "English", "Punjabi (ਪੰਜਾਬੀ)"],
                ["We must work together for the promotion of language and culture.", "English", "Punjabi (ਪੰਜਾਬੀ)"]
            ],
            inputs=[input_text_box, source_lang_dropdown, target_lang_dropdown]
        )

        def swap_languages(src, tgt):
            return tgt, src

        swap_button.click(
            fn=swap_languages,
            inputs=[source_lang_dropdown, target_lang_dropdown],
            outputs=[source_lang_dropdown, target_lang_dropdown]
        )

        translate_button.click(
            fn=translate_text,
            inputs=[input_text_box, source_lang_dropdown, target_lang_dropdown, model_path_input],
            outputs=[output_text_box, status_bar]
        )

    return demo


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
