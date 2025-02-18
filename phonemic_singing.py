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
# %%
timit_base_path = "data/TRAIN/DR1/FVMH0/SA1"
data = load_timit_files_by_path(timit_base_path)
# # %%
# data["sentence"]
# # %%
# data["phonemes"]
# # %%
data["words"]
# # %%
# data["audio"]
# %%
# load audio and split it into its phonemes
audio, sr = librosa.load(data["audio"], sr=None)
print("Audio shape:", audio.shape, "Sample rate:", sr)
phonemes = data["phonemes"]
words = data["words"]
phonemes_audio = []
w_num = 0
current_word = words.iloc[w_num]
print("Word:", current_word['word'], "start:", current_word['start'], "end:", current_word['end'])
for i, row in phonemes.iterrows():
    start = row['start']
    end = row['end']
    if start == current_word['start']:
        print('---------------------------------------')
        if w_num == 1:
            break
        w_num += 1
        
        print("Word:", current_word['word'])
        w_start = current_word['start']
        w_end = current_word['end']
        w_audio = audio[w_start:w_end]
        ipd.display(ipd.Audio(w_audio, rate=sr))
        try:
            current_word = words.iloc[w_num]
        except:
            print("LAST WORD")
        print("-------------------Word Phoneme:")
    phoneme = audio[start:end]
    phonemes_audio.append(phoneme)
    print(f"Phoneme: {row['phoneme']}, start: {start}, end: {end}")
    ipd.display(ipd.Audio(phoneme, rate=sr))

#%%
def stretch_phoneme(phoneme_audio, rate):
    return librosa.effects.time_stretch(phoneme_audio, rate=rate)

# %%
stretched_audio = np.concatenate([phonemes_audio[0], phonemes_audio[1], stretch_phoneme(phonemes_audio[2], 0.1)])
ipd.display(ipd.Audio(stretched_audio, rate=sr))
# %%

