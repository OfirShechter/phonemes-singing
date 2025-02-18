#%%
import mido
import librosa
import numpy as np
import sounddevice as sd

def midi_to_frequency(midi_note):
    """Convert MIDI note number to frequency in Hz."""
    return librosa.midi_to_hz(midi_note)

def load_midi_notes(midi_file):
    """Load MIDI file and extract notes with timestamps."""
    midi = mido.MidiFile(midi_file)
    notes = []

    tempo = 500000  # Default MIDI tempo (120 BPM) in microseconds per beat
    ticks_per_beat = midi.ticks_per_beat
    total_time = 0  # Global time tracker

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
                freq = midi_to_frequency(msg.note)
                notes.append((freq, track_time, msg.velocity))  # Store frequency, time, and velocity
    return notes

def synthesize_notes(notes, sr=22050, duration=0.5):
    """Generate a waveform from MIDI notes."""
    total_length = int(sr * (notes[-1][1] + duration))  # Compute required length of audio
    audio = np.zeros(total_length)  # Empty waveform

    for freq, start_time, velocity in notes:
        start_sample = int(start_time * sr)  # Convert start time to samples
        end_sample = start_sample + int(duration * sr)  # Duration per note
        t = np.linspace(0, duration, end_sample - start_sample, endpoint=False)
        waveform = 0.5 * np.sin(2 * np.pi * freq * t)  # Simple sine wave
        audio[start_sample:end_sample] += waveform * (velocity / 127)  # Scale by velocity

    return audio

def play_audio(audio, sr=22050):
    """Play synthesized audio using sounddevice."""
    sd.play(audio, samplerate=sr)
    sd.wait()  # Wait for playback to finish

# Load and play MIDI notes
midi_file = 'data/MIDIs/Barbie Girl - Chorus - only vocal-Piano.mid'
midi_notes = load_midi_notes(midi_file)
audio_waveform = synthesize_notes(midi_notes)
play_audio(audio_waveform)

# %%
