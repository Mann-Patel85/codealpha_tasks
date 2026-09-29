import os
import tempfile
import gradio as gr
from generate import generate_music

custom_css = """
.studio-title {text-align: center; color: #8b5cf6; font-weight: 800; margin-bottom: 2px;}
.studio-subtitle {text-align: center; color: #64748b; font-size: 14px; margin-bottom: 20px;}
"""

def ui_generate_music(num_notes, tempo_choice, instrument, temperature):
    tempo_map = {
        "Fast / Allegro (0.25s)": 0.25,
        "Standard / Moderato (0.50s)": 0.50,
        "Slow / Andante (0.75s)": 0.75,
        "Largo / Very Slow (1.00s)": 1.00
    }
    tempo_offset = tempo_map.get(tempo_choice, 0.50)
    
    temp_dir = tempfile.gettempdir()
    out_midi = os.path.join(temp_dir, f"ai_melody_{instrument.replace(' ', '_')}.mid")
    
    saved_path = generate_music(
        num_notes=int(num_notes),
        tempo_offset=tempo_offset,
        instrument_name=instrument,
        temperature=float(temperature),
        output_path=out_midi
    )
    
    if not saved_path or not os.path.exists(saved_path):
        return None, "❌ Generation failed. Make sure 'music_brain.h5' and MIDI training data exist."

    approx_duration = int(num_notes * tempo_offset)
    info_msg = (
        f"✅ **Melody Generated Successfully!**\n\n"
        f"- **Instrument:** {instrument}\n"
        f"- **Notes Count:** {num_notes}\n"
        f"- **Estimated Duration:** ~{approx_duration} seconds\n"
        f"- **Creativity (Temperature):** {temperature}\n"
        f"- **File Format:** Standard MIDI (.mid)\n\n"
        f"You can download the MIDI file below and open it in VLC, Windows Media Player, GarageBand, or any DAW."
    )
    return saved_path, info_msg

with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple", neutral_hue="slate"), css=custom_css, title="AI Music Studio") as demo:
    gr.Markdown("# 🎹 NeuroMelody AI — Music Generation Studio", elem_classes="studio-title")
    gr.Markdown("Deep Learning Sequence Modeling with LSTM Recurrent Neural Networks.", elem_classes="studio-subtitle")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎛️ Composition Settings")
            num_notes_slider = gr.Slider(
                minimum=30, maximum=300, value=100, step=10,
                label="Number of Notes / Length",
                info="Determines total melody length"
            )
            
            tempo_dropdown = gr.Dropdown(
                choices=[
                    "Fast / Allegro (0.25s)",
                    "Standard / Moderato (0.50s)",
                    "Slow / Andante (0.75s)",
                    "Largo / Very Slow (1.00s)"
                ],
                value="Standard / Moderato (0.50s)",
                label="Tempo / Note Duration"
            )
            
            instrument_dropdown = gr.Dropdown(
                choices=["Piano", "Electric Piano", "Acoustic Guitar", "Violin", "Flute"],
                value="Piano",
                label="Lead Instrument"
            )
            
            temp_slider = gr.Slider(
                minimum=0.2, maximum=1.5, value=0.8, step=0.1,
                label="Creativity / Temperature",
                info="Lower = strict structure, Higher = unpredictable jazz/creativity"
            )
            
            compose_btn = gr.Button("🎼 Compose New Melody", variant="primary", scale=2)

        with gr.Column():
            gr.Markdown("### 🎧 AI Composition Output")
            status_display = gr.Markdown("Set your preferences and click **Compose New Melody**.")
            midi_file_output = gr.File(label="Download Generated MIDI File (.mid)")

    compose_btn.click(
        fn=ui_generate_music,
        inputs=[num_notes_slider, tempo_dropdown, instrument_dropdown, temp_slider],
        outputs=[midi_file_output, status_display]
    )

    gr.Markdown("---")
    gr.Markdown("<div style='text-align: center; color: #94a3b8; font-size: 12px;'>Built with TensorFlow LSTM & Music21 | Mann Patel (CodeAlpha 2025)</div>")

if __name__ == "__main__":
    demo.launch()
