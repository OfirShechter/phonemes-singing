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
timit_base_path = "data\TRAIN\DR1\FECD0\SA2"
# data = load_timit_files_by_path(timit_base_path)
# # %%
# data["sentence"]
# # %%
# data["phonemes"]
# # %%
# data["words"]
# # %%
# data["audio"]

# load audio and split it into its phonemes
audio, sr = librosa.load('data/vocals.wav', sr=None)
# t = np.linspace(0, 5, 500000, endpoint=False)
# waveform = 0.5 * np.sin(2 * np.pi * 1000 * t)  # Simple sine wave
# ipd.display(ipd.Audio(waveform, rate=sr))
# audio = waveform
audio = audio[:int(5*sr)]


#%%
def stretch_phoneme(phoneme_audio, rate):
    return librosa.effects.time_stretch(phoneme_audio, rate=rate)

# Function to pitch-shift a phoneme
def pitch_shift_phoneme(phoneme_audio, sr, target_pitch, velocity=127):
    phoneme_audio = (phoneme_audio / np.max(phoneme_audio))  * (velocity / 127)

    # Calculate the number of steps to shift
    target_midi = librosa.note_to_midi(target_pitch)
    pitches, magnitudes = librosa.piptrack(y=phoneme_audio, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    if len(pitch_values) == 0:
        print("Warning: Could not estimate pitch for phoneme. Skipping pitch shift.")
        return phoneme_audio
    original_midi = librosa.hz_to_midi(np.mean(pitch_values))  # Get the median pitch
    original_midi = 55.8230354165873
    n_steps = target_midi - original_midi
    # print(target_midi, original_midi, n_steps)
    return librosa.effects.pitch_shift(phoneme_audio, sr=sr, n_steps=n_steps, bins_per_octave=12)
# %%
ipd.display(ipd.Audio(audio, rate=sr))
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
                notes.append((msg.note, track_time))  # Store frequency, time, and velocity
    return notes

# Load MIDI notes
midi_file = 'data\MIDIs\Barbie Girl - Chorus - only vocal-Piano.mid'
midi_notes = load_midi_notes(midi_file)

#%%
# # normalized all phonemes to the same length
# max_length = max([len(phoneme) for phoneme in phonemes_audio])
# phonemes_audio = [np.pad(phoneme, (0, max_length - len(phoneme))) for phoneme in phonemes_audio]
# %%
musical_audio= []
for i, midi_note in enumerate(midi_notes):
    print(i+1, len(midi_notes), i+1 < len(midi_notes))
    note, start_time = midi_note
    end_time = start_time + 0.5
    if i+1 < len(midi_notes):
        end_time = midi_notes[i+1][1]
    target_pitch = librosa.midi_to_note(note)
    if end_time > len(audio)/sr:
        break
    sliced_audio = audio[int(start_time*sr):int(end_time*sr)]
    
    sliced_audio = pitch_shift_phoneme(sliced_audio, sr, target_pitch)
    musical_audio.append(sliced_audio)
# %%
musical_audio_complete = np.concatenate(musical_audio)
ipd.display(ipd.Audio(musical_audio_complete, rate=sr))
# %%
