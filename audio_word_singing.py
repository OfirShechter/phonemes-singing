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
t = t = np.linspace(0, 5, len(audio), endpoint=False)
waveform = (np.min(audio)) * np.sin(2 * np.pi * 1000 * t)  # Simple sine wave
# waveform = 0
audio = audio #+ waveform
ipd.display(ipd.Audio(audio, rate=sr))
#%%
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
def stretch(audio, rate):
    return librosa.effects.time_stretch(audio, rate=rate)

# Function to pitch-shift a phoneme
def pitch_shift(audio, sr, target_pitch, velocity = 127):
    # adjust audio velocity
    # audio = (audio / np.max(audio))  * (velocity / 127)
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
    print(np.mean(pitch_values))
    original_midi = librosa.hz_to_midi(np.mean(pitch_values))  # Get the median pitch
    # original_midi = 75
    n_steps = target_midi - original_midi
    # print(target_midi, original_midi, n_steps)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
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
midi_file = 'data\MIDIs\custom score.mid'
midi_notes = load_midi_notes(midi_file)

#%%
# # normalized all phonemes to the same length
# max_length = max([len(phoneme) for phoneme in phonemes_audio])
# phonemes_audio = [np.pad(phoneme, (0, max_length - len(phoneme))) for phoneme in phonemes_audio]
#%%
merged_audio = np.concatenate(words_audio)
ipd.display(ipd.Audio(merged_audio, rate=sr))
#%%
from collections import deque
word_lifo = deque(words_audio[::-1])
wav_audios_words = [monotune_waveform[start:end] for start, end in words_indexes]
words_lifo_wav = deque(wav_audios_words[::-1])
# ipd.display(ipd.Audio(words_audio[0], rate=sr))
# ipd.display(ipd.Audio(words_lifo.pop(), rate=sr))
# %%
def align_with_midi(lifo):
    musical_audio= []
    for i, midi_note in enumerate(midi_notes):
        print(i+1, len(midi_notes), i+1 < len(midi_notes))
        note, start_time = midi_note
        end_time = start_time + 0.5
        if i+1 < len(midi_notes):
            end_time = midi_notes[i+1][1]
        target_pitch = librosa.midi_to_note(note)
        # if end_time > len(audio)/sr:
        #     break
        if not lifo:
            break
        word = lifo.pop()
        duration = end_time - start_time
        word_duration = len(word) / sr
        if duration < word_duration:
            slice_i = int((word_duration/(word_duration % duration + 1)) * sr)
            new_word = word[slice_i:]
            lifo.append(new_word)
            word = word[:slice_i]
        rate = (len(word) / sr) / (duration)
        word = stretch(word, rate)
        pitched_word = pitch_shift(word, sr, target_pitch)
        musical_audio.append(pitched_word)
    musical_audio_complete = np.concatenate(musical_audio)
    return musical_audio_complete
# %%
musical_word_complete = align_with_midi(word_lifo)
ipd.display(ipd.Audio(musical_word_complete, rate=sr, normalize=False))
musical_wav_complete = align_with_midi(words_lifo_wav)
ipd.display(ipd.Audio(musical_wav_complete, rate=sr, normalize=False))
# %%
merged_audio = np.concatenate(words_audio)
ipd.display(ipd.Audio(merged_audio, rate=sr, normalize=False))

# %%
# use matplotlib to plot pitch in time diagrama of merged_audio and musical_audio_complete on the same plot
# get pitches of merged_audio
pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
pitch_values = []
for t in range(pitches.shape[1]):
    index = magnitudes[:, t].argmax()
    pitch = pitches[index, t]
    if pitch > 0:
        pitch_values.append(pitch)
wav_audio_pitch = np.array(pitch_values)
# get pitches of musical_audio_complete
pitches, magnitudes = librosa.piptrack(y=monotune_waveform, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
pitch_values = []
for t in range(pitches.shape[1]):
    index = magnitudes[:, t].argmax()
    pitch = pitches[index, t]
    if pitch > 0:
        pitch_values.append(pitch)
musical_audio_pitch = np.array(pitch_values)
# plot pitches
plt.figure(figsize=(14, 5))
plt.plot(np.arange(len(wav_audio_pitch)) / sr, wav_audio_pitch, label="monotonic")
plt.plot(np.arange(len(musical_audio_pitch)) / sr, musical_audio_pitch, label="Musical")
plt.xlabel("Time (s)")
plt.ylabel("Pitch (Hz)")
plt.legend()
plt.show()


# %%
import librosa
import numpy as np
import soundfile as sf

def convert_to_monotone(y,sr, target_pitch=500):    
    # Harmonic-percussive source separation
    y_harmonic, _ = librosa.effects.hpss(y)
    
    # Estimate pitch using piptrack
    pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sr, win_length=10)
    
    # Find average pitch to understand baseline
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    
    avg_pitch = np.mean(pitch_values) if pitch_values else target_pitch
    print(f"Average detected pitch: {avg_pitch:.2f} Hz")

    # Calculate pitch shift to reach target pitch
    pitch_shift = target_pitch / avg_pitch if avg_pitch > 0 else 1.0
    
    # Shift pitch using librosa
    y_monotone = librosa.effects.pitch_shift(y_harmonic, sr=sr, n_steps=np.log2(pitch_shift))
    
    return y_monotone

# %%
monotune_waveform = convert_to_monotone(audio, sr)
ipd.display(ipd.Audio(monotune_waveform, rate=sr))
ipd.display(ipd.Audio(audio, rate=sr))
# %%
