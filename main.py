import speech_recognition as sr
import numpy as np
import sounddevice as sd
import librosa
#import pygame
import tkinter as tk

#KIWIMD maybe for mobile integration

tun = False

tone_midi_guitar_dict = {
    "A1": 33,
    "B1": 35,
    "C2": 36,
    "D2": 38,
    "E2": 40,
    "F2": 41,
    "G2": 43,
    "A2": 45,
    "B2": 47,
    "C3": 48,
    "D3": 50,
    "E3": 52,
    "F3": 53,
    "G3": 55,
    "A3": 57,
    "B3": 59,
    "C4": 60,
    "D4": 62,
    "E4": 64,
    "F4": 65,
    "G4": 67,
    "A4": 69,
    "B4": 71,
    "C5": 72,
    "D5": 74,
    "E5": 76,
    "F5": 77,
    "G5": 79,
    "A5": 81,
    "B5": 83,
    "C6": 84,
    "D6": 86,
}

#or use librosa midi to hz

#n = 12 * (\log(f / 220) / \log(2)) + 57

# Callback for continuous audio input
def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata.flatten()


def analyze_pitch(audio_data):
    pitches, magnitudes, probs = librosa.pyin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('A5'), sr=44100, n_thresholds=100)
    detected_notes = [librosa.hz_to_note(pitch) for pitch, mag in zip(pitches, magnitudes) if mag]
    detected_pitches = [pitch for pitch, mag in zip(pitches, magnitudes) if mag]
    if len(detected_notes) <= 10 or probs[np.argmax(probs)] < 0.5:
        max_magnitude_note = ""
        max_magnitude_pitch = []
    else:
        average_freq = sum(detected_pitches)/len(detected_pitches)
        max_magnitude_pitch = average_freq
        max_magnitude_note = librosa.hz_to_note(max_magnitude_pitch)
    return max_magnitude_pitch, max_magnitude_note

def detect_voice(label):
    init_rec = sr.Recognizer()
    label.config(text="Let's speak!!")
    label.update()
    with sr.Microphone() as source:
        audio_data = init_rec.listen(source)
        label.config(text="Recognizing your text.............")
        label.update()
        try:
            text = init_rec.recognize_google(audio_data)
            print(text)
            return text
        except:
            label.config(text="Sorry, I did not get that")
            label.update()
            return ""


def log_data(detected_notes, detected_freqs, to_hit_freq):
    print("Notes:", detected_notes)
    print("freqs:", detected_freqs)
    print("To hit freq:", to_hit_freq)
    print("-------------------")

#def play_sound():
#    pygame.mixer.music.load("tune.wav")
#    pygame.mixer.music.play()

def generate_tune(to_hit_freq, curr_freq, duration=0.7, sr=44100):
    print(to_hit_freq, curr_freq)
    t = np.linspace(0, duration, int(sr * duration), False)
    if curr_freq > to_hit_freq:
        curr_freq *=2
    elif curr_freq < to_hit_freq:
        curr_freq /= 2
    freqs = np.linspace(curr_freq, to_hit_freq, len(t))
    tune = 0.5 * np.sin(2 * np.pi * freqs * t)
    return tune

def generate_static_tune(to_hit_freq, curr_freq, duration=0.7, sr=44100):
    static_freq = 329.6275569128699
    max_freq_top = 349.2282314330039
    max_freq_bottom = 311.1269837220809
    if curr_freq > to_hit_freq:
        return generate_tune(static_freq, max_freq_top, duration, sr)
    elif curr_freq < to_hit_freq:
        return generate_tune(static_freq, max_freq_bottom, duration, sr)

def p_s(tune):
    sd.play(tune, samplerate=44100)
    sd.wait()


def generate_kaching_sound(duration=0.4, sample_rate=44100):
    """
    Generate a simple 'ka-ching' sound using sine waves and noise.

    Args:
        duration (float): Duration of the sound in seconds.
        sample_rate (int): Number of samples per second.

    Returns:
        numpy.ndarray: Array containing the generated 'ka-ching' sound.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Bell-like sine wave (higher frequencies give the bell-like tone)
    base_freq = 1000  # Base frequency for the "chime"
    bell_tone = 0.5 * np.sin(2 * np.pi * base_freq * t)

    # Add a higher frequency to simulate the ringing (bell)
    bell_tone += 0.3 * np.sin(2 * np.pi * (base_freq * 2) * t)

    # Add white noise for metallic effect
    noise = 0.2 * np.random.normal(0, 1, len(t))

    # Mix the bell tone and the noise
    kaching = bell_tone + noise

    # Apply a fast decay (decay envelope)
    decay = np.exp(-t * 10)  # Exponential decay for a quick fade out
    kaching *= decay

    return kaching

#main loop
#pygame.mixer.init()
root = tk.Tk()
root.title("Voice controlled tuner")
root.geometry("400x300")

label = tk.Label(root, text="PLACEHOLDER", font=("Helvetica", 16))
label.pack(expand=True)
root.update()


SAMPLE_RATE = 44100
CHANNELS = 1
WINDOW_DURATION = 0.5
STEP_DURATION = 0.1
BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
STEP_SIZE = int(SAMPLE_RATE * STEP_DURATION)

audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)


midi_to_hit = -1
to_hit_freq = 0
max_dif_top = 0
max_dif_bottom = 0

no_input = 0

def voice_and_tuning():
    global tun, midi_to_hit, no_input, max_dif_top, max_dif_bottom, to_hit_freq

    if tun:
        detected_freqs, detected_notes = analyze_pitch(audio_buffer)

        log_data(detected_notes, detected_freqs, to_hit_freq)

        if len(detected_notes) == 0:
            if no_input > 20:
                no_input = 0
                tun = False
                color = (255, 255, 255)
                root.config(bg='#%02x%02x%02x' % color)
                label.config(text="Exiting tuning mode")
                root.after(150, voice_and_tuning)
            else:
                color = (255, 255, 255)
                root.config(bg='#%02x%02x%02x' % color)
                label.config(text="Tuning to: {0}".format(str(librosa.midi_to_note(midi_to_hit))))
                no_input += 1
                root.after(50, voice_and_tuning)
            return

        no_input = 0
        note_midi = tone_midi_guitar_dict.get(detected_notes)

        if note_midi == midi_to_hit + 12:
            tun = False
            color = (0, 255, 0)
            root.config(bg='#%02x%02x%02x' % color)
            label.config(text="Tuned to: {0}".format(str(librosa.midi_to_note(midi_to_hit))))
            root.after(50, voice_and_tuning)
            return
        #print(detected_freqs)



        bottom_safe_level = to_hit_freq- 0.1*max_dif_bottom
        top_safe_level = to_hit_freq + 0.1*max_dif_top

        red, green = 255, 255
        if detected_freqs > top_safe_level:
            label.config(text="Tune down")
            difference = abs(top_safe_level - detected_freqs)
            if difference > max_dif_top:
                difference = max_dif_top
            red = 0 + int(255 * (difference / max_dif_top))
            green = 255 - int(255 * (difference / max_dif_top))

            tune = generate_tune(to_hit_freq, to_hit_freq+difference)
            print("PLAYING")
            p_s(tune)

        elif detected_freqs < bottom_safe_level:
            label.config(text="Tune up")
            difference = abs(bottom_safe_level - detected_freqs)
            if difference > max_dif_bottom:
                difference = max_dif_bottom
            red = 0 + int(255 * (difference / max_dif_bottom))
            green = 255 - int(255 * (difference / max_dif_bottom))

            tune = generate_tune(to_hit_freq, to_hit_freq-difference)
            print("PLAYING")
            p_s(tune)

        elif detected_freqs >= bottom_safe_level and detected_freqs <= top_safe_level:
            label.config(text="In tune")
            red = 0
            green = 255

            p_s(generate_kaching_sound())


        color = (red, green, 30)

        print("---------")
        root.config(bg='#%02x%02x%02x' % color)

    else:
        tuning = detect_voice(label)
        root.config(bg='#%02x%02x%02x' % (255, 255, 255))
        if len(tuning) == 2 and tuning[0] in "CDEFGABC" and tuning[1].isdigit():
            label.config(text="Tuning to {0}".format(tuning))
            midi_to_hit = tone_midi_guitar_dict[tuning]
            tun = True

            note_above_freq = librosa.midi_to_hz(midi_to_hit + 1)
            note_below_freq = librosa.midi_to_hz(midi_to_hit - 1)
            to_hit_freq = librosa.midi_to_hz(midi_to_hit)

            max_dif_top = abs(to_hit_freq - note_above_freq)
            max_dif_bottom = abs(to_hit_freq - note_below_freq)

        elif tuning == "exit":
            label.config(text="Exiting tuner")
            root.after(1000, root.destroy())
            exit(0)

    root.after(50, voice_and_tuning)



stream = sd.InputStream(
    samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, blocksize=STEP_SIZE
)
stream.start()

voice_and_tuning()
root.mainloop()

