# ==========================================================
# AI MUSIC GENERATION USING LSTM
# Internship Project
# Author: Your SOHAM
# ==========================================================

# Install Required Libraries
# pip install tensorflow music21 numpy

import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================================================
# STEP 1: LOAD MIDI FILES
# ==========================================================

notes = []

print("Loading MIDI files...")

for file in glob.glob("dataset/*.mid"):

    midi = converter.parse(file)

    parts = instrument.partitionByInstrument(midi)

    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:

        if isinstance(element, note.Note):
            notes.append(str(element.pitch))

        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

print("Total Notes:", len(notes))

# ==========================================================
# STEP 2: PREPROCESS DATA
# ==========================================================

pitchnames = sorted(set(notes))

note_to_int = dict(
    (note_name, number)
    for number, note_name in enumerate(pitchnames)
)

int_to_note = dict(
    (number, note_name)
    for number, note_name in enumerate(pitchnames)
)

sequence_length = 100

network_input = []
network_output = []

for i in range(len(notes) - sequence_length):

    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]

    network_input.append(
        [note_to_int[char] for char in sequence_in]
    )

    network_output.append(
        note_to_int[sequence_out]
    )

n_patterns = len(network_input)

network_input = np.reshape(
    network_input,
    (n_patterns, sequence_length, 1)
)

network_input = network_input / float(len(pitchnames))

network_output = np.array(network_output)

print("Training Patterns:", n_patterns)

# ==========================================================
# STEP 3: BUILD LSTM MODEL
# ==========================================================

print("Building Model...")

model = Sequential()

model.add(
    LSTM(
        256,
        input_shape=(
            network_input.shape[1],
            network_input.shape[2]
        ),
        return_sequences=True
    )
)

model.add(Dropout(0.3))

model.add(LSTM(256))

model.add(Dropout(0.3))

model.add(Dense(128, activation="relu"))

model.add(Dense(len(pitchnames), activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam"
)

model.summary()

# ==========================================================
# STEP 4: TRAIN MODEL
# ==========================================================

print("Training Started...")

model.fit(
    network_input,
    network_output,
    epochs=50,
    batch_size=64
)

model.save("music_generator.h5")

print("Model Saved!")

# ==========================================================
# STEP 5: GENERATE MUSIC
# ==========================================================

print("Generating Music...")

start = np.random.randint(0, len(network_input) - 1)

pattern = network_input[start]

prediction_output = []

for note_index in range(200):

    prediction_input = np.reshape(
        pattern,
        (1, len(pattern), 1)
    )

    prediction = model.predict(
        prediction_input,
        verbose=0
    )

    index = np.argmax(prediction)

    result = int_to_note[index]

    prediction_output.append(result)

    pattern = np.append(
        pattern,
        [[index / float(len(pitchnames))]],
        axis=0
    )

    pattern = pattern[1:]

print("Music Generated!")

# ==========================================================
# STEP 6: SAVE GENERATED MUSIC AS MIDI
# ==========================================================

offset = 0
output_notes = []

for pattern in prediction_output:

    if '.' in pattern:

        notes_in_chord = pattern.split('.')
        chord_notes = []

        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.offset = offset
            chord_notes.append(new_note)

        new_chord = chord.Chord(chord_notes)
        output_notes.append(new_chord)

    else:

        new_note = note.Note(pattern)
        new_note.offset = offset
        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)

midi_stream.write(
    'midi',
    fp='generated_music.mid'
)

print("===================================")
print("AI MUSIC GENERATION COMPLETED")
print("Generated File: generated_music.mid")
print("Model File: music_generator.h5")
print("===================================")