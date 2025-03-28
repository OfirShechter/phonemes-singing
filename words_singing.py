#%%
# accsess folder in "E:" drive (this code is on C: drive)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
import soundfile as sf
import os
#%%
# load TIMIT TRAIN dataset using pandas
# structure of TIMIT dataset
# data
# ├── TRAIN
# │   ├── DR1
# │   │   ├── FADG0
# │   │   │   ├── SA1.WAV.wav
# │   │   │   ├── SA1.PHN
# │   │   │   ├── SA1.TXT
# │   │   │   ├── SA1.WRD
# │   │   │   ├── SA1.WAV
# │   │   │   ├── SA2.WAV.wav
# │   │   │   ├── SA2.PHN
# │   │   │   ├── SA2.TXT
# │   │   │   ├── SA2.WRD
# │   │   │   ├── SA2.WAV
# │   │   │   ├── ...
def load_timit_files_by_path(path):
    # read lines from path.TXT file and format them
    with open(path + ".TXT", "r") as f:
        txt = f.readlines()
    txt = [line.strip().split(" ") for line in txt]
    assert len(txt) == 1
    txt = txt[0]
    sentence = {'start': int(txt[0]), 'end': int(txt[1]), 'content': " ".join(txt[2:])}
    # read lines from path.PHN and store them in a dataframe
    phonemes = pd.DataFrame([], columns=["start", "end", "phoneme"])
    with open(path + ".PHN", "r") as f:
        phn = f.readlines()
    phn = [line.strip().split(" ") for line in phn]
    for p in phn:
        phonemes.loc[len(phonemes)] = {"start": int(p[0]), "end": int(p[1]), "phoneme": p[2]}
    # read lines from path.WRD and store them in a dataframe
    words = pd.DataFrame([], columns=["start", "end", "word"])
    with open(path + ".WRD", "r") as f:
        wrd = f.readlines()
    wrd = [line.strip().split(" ") for line in wrd]
    for w in wrd:
        words.loc[len(words)] = {"start": int(w[0]), "end": int(w[1]), "word": w[2]}
    # save the path of the audio file
    audio = path + ".WAV.wav"
    # combine all the data in a dictionary
    data = {"sentence": sentence, "phonemes": phonemes, "words": words, "audio": audio}
    return data 
#%%
timit_base_path = "data/TRAIN/DR1/FVMH0/SA1"
data = load_timit_files_by_path(timit_base_path)
# # %%
# data["sentence"]
# # %%
# data["phonemes"]
# # %%
# data["words"]
# # %%
# data["audio"]
# %%
# load audio and split it into its phonemes
audio, sr = librosa.load(data["audio"], sr=None)
print("Audio shape:", audio.shape, "Sample rate:", sr)
phonemes = data["phonemes"]
words = data["words"]
words_audio = []
words_indexes = []
w_num = 0
current_word = words.iloc[w_num]
print("Word:", current_word['word'], "start:", current_word['start'], "end:", current_word['end'])
for i, row in words.iterrows():
    start = row['start']
    end = row['end']
    word = audio[start:end]
    words_audio.append(word)
    words_indexes.append((start, end))
    print(f"Word: {row['word']}, start: {start}, end: {end}")
    ipd.display(ipd.Audio(word, rate=sr))

#%%
def get_pitch(y, sr):
    # Extract fundamental frequency (f0) using librosa.pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=2000, sr=sr)
    # Remove NaN values (unvoiced frames)
    f0 = f0[~np.isnan(f0)]

    if len(f0) == 0:
        return None  # No pitch detected

    # Return the average pitch
    return np.mean(f0)

librosa.hz_to_midi(get_pitch(audio, sr))
#%%
def stretch(audio, rate):
    return librosa.effects.time_stretch(audio, rate=rate)

# Function to pitch-shift a phoneme
def pitch_shift(audio, sr, target_pitch, velocity):
    # adjust audio velocity
    audio = (audio / np.max(audio))  * (velocity / 127)
    # Calculate the number of steps to shift
    target_midi = librosa.note_to_midi(target_pitch)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    if len(pitch_values) == 0:
        print("Warning: Could not estimate pitch for phoneme. Skipping pitch shift.")
        return audio
    original_midi = librosa.hz_to_midi(np.mean(pitch_values))  # Get the median pitch
    # original_midi = 75
    n_steps = target_midi - original_midi
    # print(target_midi, original_midi, n_steps)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
# %%
# stretched_audio = np.concatenate([phonemes_audio[0], phonemes_audio[1], stretch_phoneme(phonemes_audio[2], 0.1)])
merged_audio = np.concatenate(words_audio)
ipd.display(ipd.Audio(merged_audio, rate=sr))
# %%
import mido
# Function to load MIDI file and extract notes
def load_midi_notes(midi_file):
    midi = mido.MidiFile(midi_file)
    notes = []

    tempo = 500000  # Default MIDI tempo (120 BPM) in microseconds per beat
    ticks_per_beat = midi.ticks_per_beat

    for track in midi.tracks:
        track_time = 0  # Track-specific time accumulator
        for msg in track:
            delta_time = mido.tick2second(msg.time, ticks_per_beat, tempo)
            # print("track_time:", track_time, "delta_time:", delta_time)
            track_time += delta_time  # Accumulate time per track

            if msg.type == 'set_tempo':
                tempo = msg.tempo  # Update tempo dynamically

            if msg.type == 'note_on' and msg.velocity > 0:
                print("track_time:", track_time, "delta_time:", delta_time)
                notes.append((msg.note, track_time, msg.velocity))  # Store frequency, time, and velocity
    return notes

# Load MIDI notes
midi_file = 'data\MIDIs\Barbie Girl - Chorus - only vocal-Piano.mid'
midi_notes = load_midi_notes(midi_file)

#%%
# # normalized all phonemes to the same length
# max_length = max([len(phoneme) for phoneme in phonemes_audio])
# phonemes_audio = [np.pad(phoneme, (0, max_length - len(phoneme))) for phoneme in phonemes_audio]
# %%
musical_phonemes = []
for i, midi_note in enumerate(midi_notes):
    print(i+1, len(midi_notes), i+1 < len(midi_notes))
    w_index = i % len(words_audio)
    word = words_audio[w_index]
    note, start_time, velocity = midi_note
    duration = 0.5
    if i+1 < len(midi_notes):
        duration = (midi_notes[i+1][1] - start_time)
    target_pitch = librosa.midi_to_note(note)
    word = pitch_shift(word, sr, target_pitch, velocity)
    start, end = words_indexes[w_index]
    rate = ((end - start) / sr) / (duration)
    print(duration, ((end - start)/sr), rate)
    word = stretch(word, rate)
    musical_phonemes.append(word)
# %%
musical_audio = np.concatenate(musical_phonemes)
ipd.display(ipd.Audio(musical_audio, rate=sr))
# %%
ipd.display(ipd.Audio(merged_audio, rate=sr))
# %%
