import os
import glob
import numpy as np
import music21
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout #type: ignore
from tensorflow.keras.utils import to_categorical #type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "music_data")
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "music_brain.h5")
ROOT_MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "music_brain.h5")

def get_notes():
    notes = []
    path_to_search = os.path.join(DATA_DIR, "*.mid")
    files = glob.glob(path_to_search)
    
    if not files:
        # Fallback to root or current dir
        files = glob.glob(os.path.join(BASE_DIR, "*.mid"))
        
    if not files:
        print(f"❌ Error: No .mid files found at: {path_to_search}")
        return []

    print(f"🎵 Parsing notes and chords from {len(files)} MIDI file(s)...")
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
            print(f"⚠️ Error parsing file {file}: {e}")
    
    print(f"✅ Success! Extracted {len(notes)} total notes/chords.")
    return notes

def create_model(n_vocab, input_shape):
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def train_network(epochs=5, batch_size=64):
    notes = get_notes()
    if not notes:
        print("❌ Cannot train: No notes extracted.")
        return
    
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(pitchnames)
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    sequence_length = 50
    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    if n_patterns == 0:
        print("❌ Insufficient notes to form training sequences.")
        return

    network_input_reshaped = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input_reshaped = network_input_reshaped / float(n_vocab)
    network_output = to_categorical(network_output, num_classes=n_vocab)

    print("\n🧠 Building LSTM Neural Network...")
    model = create_model(n_vocab, (network_input_reshaped.shape[1], network_input_reshaped.shape[2]))
    
    print(f"\n🏋️ Training on {len(pitchnames)} unique vocabulary tokens for {epochs} epochs...")
    model.fit(network_input_reshaped, network_output, epochs=epochs, batch_size=batch_size)
    
    model.save(MODEL_SAVE_PATH)
    model.save(ROOT_MODEL_SAVE_PATH)
    print(f"\n✅ Training Complete! Model saved to:\n- {MODEL_SAVE_PATH}\n- {ROOT_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_network(epochs=5)