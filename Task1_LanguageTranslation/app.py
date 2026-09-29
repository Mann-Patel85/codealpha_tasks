import os
import tempfile
import gradio as gr
from deep_translator import GoogleTranslator
from gtts import gTTS

# Supported Languages List
LANGUAGES = [
    ("English", "en"),
    ("Spanish (Español)", "es"),
    ("French (Français)", "fr"),
    ("German (Deutsch)", "de"),
    ("Italian (Italiano)", "it"),
    ("Portuguese (Português)", "pt"),
    ("Hindi (हिन्दी)", "hi"),
    ("Gujarati (ગુજરાતી)", "gu"),
    ("Marathi (मराठी)", "mr"),
    ("Tamil (தமிழ்)", "ta"),
    ("Telugu (తెలుగు)", "te"),
    ("Bengali (বাংলা)", "bn"),
    ("Punjabi (ਪੰਜਾਬੀ)", "pa"),
    ("Chinese (Simplified)", "zh-CN"),
    ("Japanese (日本語)", "ja"),
    ("Korean (한국어)", "ko"),
    ("Russian (Русский)", "ru"),
    ("Arabic (العربية)", "ar"),
    ("Turkish (Türkçe)", "tr"),
    ("Dutch (Nederlands)", "nl"),
    ("Polish (Polski)", "pl"),
    ("Indonesian (Bahasa)", "id"),
    ("Vietnamese (Tiếng Việt)", "vi"),
    ("Thai (ไทย)", "th"),
    ("Swedish (Svenska)", "sv")
]

LANG_CODE_MAP = {name: code for name, code in LANGUAGES}
CODE_TO_LANG_MAP = {code: name for name, code in LANGUAGES}

# Translation Backend Logic
def translate_text(text, source_lang, target_lang):
    if not text or not text.strip():
        return "", "Characters: 0 | Words: 0", "⚠️ Please enter text to translate."
    try:
        src = 'auto' if source_lang == 'auto' else source_lang
        translator = GoogleTranslator(source=src, target=target_lang)
        result = translator.translate(text)
        
        char_count = len(result)
        word_count = len(result.split())
        stats = f"📊 Characters: {char_count} | Words: {word_count} | Source: {source_lang.upper()}"
        return result, stats, "✅ Translation successful!"
    except Exception as e:
        return "", "Characters: 0 | Words: 0", f"❌ Error: {str(e)}"

# Text-to-Speech Generation Logic
def generate_speech(translated_text, target_lang):
    if not translated_text or not translated_text.strip():
        return None, "⚠️ No translated text available for audio playback."
    try:
        # Some language codes might need normalization for gTTS (e.g. zh-CN -> zh)
        tts_lang = target_lang.split('-')[0] if '-' in target_lang else target_lang
        tts = gTTS(text=translated_text, lang=tts_lang, slow=False)
        
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, "translated_audio.mp3")
        tts.save(audio_path)
        return audio_path, "🔊 Audio generated successfully!"
    except Exception as e:
        return None, f"⚠️ Audio generation failed for language '{target_lang}': {str(e)}"

# Swap Source and Target Languages
def swap_languages(src, tgt, src_text, tgt_text):
    new_src = tgt if tgt != 'auto' else 'en'
    new_tgt = src if src != 'auto' else 'en'
    return new_src, new_tgt, tgt_text, src_text

# File Translation Handler
def translate_file(file_obj, target_lang):
    if file_obj is None:
        return "", None, "⚠️ Please upload a .txt file."
    try:
        with open(file_obj.name, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not content.strip():
            return "", None, "⚠️ Uploaded file is empty."

        translator = GoogleTranslator(source='auto', target=target_lang)
        translated = translator.translate(content)
        
        temp_dir = tempfile.gettempdir()
        out_file_path = os.path.join(temp_dir, f"translated_{target_lang}.txt")
        with open(out_file_path, 'w', encoding='utf-8') as f:
            f.write(translated)
            
        return translated, out_file_path, "✅ File translated and ready for download!"
    except Exception as e:
        return "", None, f"❌ File translation error: {str(e)}"

# Custom Theme and CSS
custom_css = """
.main-title {text-align: center; font-weight: 800; color: #2563eb; margin-bottom: 4px;}
.subtitle {text-align: center; color: #64748b; font-size: 15px; margin-bottom: 24px;}
.status-bar {font-size: 13px; font-weight: 500; padding: 4px 8px; border-radius: 6px;}
.footer-text {text-align: center; color: #94a3b8; font-size: 13px; margin-top: 20px;}
"""

# Build Modern Gradio Blocks App
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=custom_css, title="GlobalConnect AI - Pro Translator") as demo:
    
    gr.Markdown("# 🌐 GlobalConnect AI — Enterprise Neural Translator", elem_classes="main-title")
    gr.Markdown("Real-time multi-lingual neural translation, voice synthesis (TTS), and document processing.", elem_classes="subtitle")
    
    with gr.Tabs():
        # --- TAB 1: Live Interactive Translator ---
        with gr.TabItem("💬 Live Text Translation & Audio"):
            with gr.Row():
                source_lang_dropdown = gr.Dropdown(
                    choices=[("Auto Detect", "auto")] + LANGUAGES,
                    value="auto",
                    label="Source Language",
                    info="Select origin or leave on Auto Detect"
                )
                swap_btn = gr.Button("⇄ Swap", variant="secondary", scale=0)
                target_lang_dropdown = gr.Dropdown(
                    choices=LANGUAGES,
                    value="fr",
                    label="Target Language",
                    info="Select destination language"
                )

            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Source Text",
                        placeholder="Type or paste your text here to translate...",
                        lines=6,
                        show_copy_button=True
                    )
                    with gr.Row():
                        clear_btn = gr.ClearButton([input_text], variant="secondary")
                        translate_btn = gr.Button("✨ Translate", variant="primary", scale=2)

                with gr.Column():
                    output_text = gr.Textbox(
                        label="Translated Text",
                        placeholder="Translation will appear here...",
                        lines=6,
                        interactive=False,
                        show_copy_button=True
                    )
                    stats_badge = gr.Markdown("📊 Characters: 0 | Words: 0", elem_classes="status-bar")
                    status_badge = gr.Markdown("Ready", elem_classes="status-bar")

            with gr.Row():
                tts_btn = gr.Button("🔊 Listen to Translation (TTS)", variant="secondary")
                audio_output = gr.Audio(label="Voice Playback", interactive=False)

            # Example Cards
            gr.Examples(
                examples=[
                    ["Artificial Intelligence is empowering global innovation.", "auto", "es"],
                    ["Where is the nearest international airport and train terminal?", "auto", "de"],
                    ["Machine Learning enables automated decision making at scale.", "auto", "hi"],
                    ["Welcome to our technical internship project presentation.", "auto", "gu"],
                    ["Thank you for your valuable feedback and guidance.", "auto", "ja"]
                ],
                inputs=[input_text, source_lang_dropdown, target_lang_dropdown],
                label="💡 Quick Prompts:"
            )

        # --- TAB 2: Document / File Translation ---
        with gr.TabItem("📄 Document / File Translation"):
            gr.Markdown("Upload any `.txt` document to translate all paragraphs into the target language.")
            with gr.Row():
                file_input = gr.File(label="Upload Text File (.txt)", file_types=[".txt"])
                file_target_lang = gr.Dropdown(
                    choices=LANGUAGES,
                    value="es",
                    label="Translate Document To:"
                )
            file_translate_btn = gr.Button("🚀 Translate Document", variant="primary")
            
            with gr.Row():
                file_preview = gr.Textbox(label="Translated File Preview", lines=8, interactive=False)
                file_download = gr.File(label="Download Translated Document")
            file_status = gr.Markdown("Ready to process documents.")

    gr.Markdown("---")
    gr.Markdown("Designed & Developed by **Mann Patel** | CodeAlpha Artificial Intelligence Internship", elem_classes="footer-text")

    # Wire up Event Handlers
    translate_btn.click(
        fn=translate_text,
        inputs=[input_text, source_lang_dropdown, target_lang_dropdown],
        outputs=[output_text, stats_badge, status_badge]
    )
    
    tts_btn.click(
        fn=generate_speech,
        inputs=[output_text, target_lang_dropdown],
        outputs=[audio_output, status_badge]
    )
    
    swap_btn.click(
        fn=swap_languages,
        inputs=[source_lang_dropdown, target_lang_dropdown, input_text, output_text],
        outputs=[source_lang_dropdown, target_lang_dropdown, input_text, output_text]
    )
    
    file_translate_btn.click(
        fn=translate_file,
        inputs=[file_input, file_target_lang],
        outputs=[file_preview, file_download, file_status]
    )

if __name__ == "__main__":
    demo.launch()