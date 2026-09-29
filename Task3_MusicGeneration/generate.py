import os
import glob
import numpy as np
import music21
from tensorflow.keras.models import load_model #type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "music_data")
ROOT_DIR = os.path.dirname(BASE_DIR)

def find_model_path():
    candidates = [
        os.path.join(BASE_DIR, "music_brain.h5"),
        os.path.join(ROOT_DIR, "music_brain.h5"),
        "music_brain.h5"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def get_notes():
    notes = []
    path_to_search = os.path.join(DATA_DIR, "*.mid")
    files = glob.glob(path_to_search)
    if not files:
        files = glob.glob(os.path.join(BASE_DIR, "*.mid"))
        
    for file in files:
        try:
            midi = music21.converter.parse(file)
            elements_to_parse = midi.flat.notes
            for element in elements_to_parse:
                if isinstance(element, music21.note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, music21.chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return notes

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    if temperature <= 0:
        return np.argmax(preds)
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_music(num_notes=100, tempo_offset=0.5, instrument_name='Piano', temperature=0.8, output_path=None):
    notes = get_notes()
    if not notes:
        print("❌ No notes found in training dataset.")
        return None
        
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(pitchnames)
    
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    sequence_length = 50
    network_input = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
    
    if not network_input:
        print("❌ Training data too short for sequence generation.")
        return None

    model_path = find_model_path()
    if not model_path:
        print("❌ Trained model 'music_brain.h5' not found. Run music_ai.py first.")
        return None
        
    print(f"🧠 Loading Trained AI Model from: {model_path}")
    model = load_model(model_path)
    
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    
    prediction_output = []
    print(f"🎹 Composing {num_notes} notes (Temperature: {temperature})...")
    
    for _ in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        
        prediction = model.predict(prediction_input, verbose=0)[0]
        index = sample_with_temperature(prediction, temperature)
        result = int_to_note[index]
        prediction_output.append(result)
        
        pattern.append(index)
        pattern = pattern[1:]

    # Construct MIDI Stream
    offset = 0
    output_notes = []
    
    # Select Instrument
    inst_map = {
        'Piano': music21.instrument.Piano(),
        'Electric Piano': music21.instrument.ElectricPiano(),
        'Acoustic Guitar': music21.instrument.AcousticGuitar(),
        'Violin': music21.instrument.Violin(),
        'Flute': music21.instrument.Flute()
    }
    selected_instrument = inst_map.get(instrument_name, music21.instrument.Piano())

    for item in prediction_output:
        if ('.' in item) or item.isdigit():
            notes_in_chord = item.split('.')
            chord_notes = []
            for current_note in notes_in_chord:
                try:
                    new_note = music21.note.Note(int(current_note))
                    new_note.storedInstrument = selected_instrument
                    chord_notes.append(new_note)
                except Exception:
                    pass
            if chord_notes:
                new_chord = music21.chord.Chord(chord_notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
        else:
            try:
                new_note = music21.note.Note(item)
                new_note.offset = offset
                new_note.storedInstrument = selected_instrument
                output_notes.append(new_note)
            except Exception:
                pass
        
        offset += tempo_offset

    midi_stream = music21.stream.Stream(output_notes)
    
    if output_path is None:
        output_path = os.path.join(BASE_DIR, 'ai_generated_song.mid')
    
    midi_stream.write('midi', fp=output_path)
    
    # Also save to root directory for convenient access
    root_output = os.path.join(ROOT_DIR, 'ai_generated_song.mid')
    if output_path != root_output:
        try:
            midi_stream.write('midi', fp=root_output)
        except Exception:
            pass

    print(f"✅ Success! Music saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_music(num_notes=100, tempo_offset=0.5, temperature=0.8)