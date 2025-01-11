import time

from kivy.lang import Builder
from kivy.properties import ListProperty
from kivy.clock import Clock
from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.graphics import Color, Rectangle
from kivymd.uix.screen import MDScreen
from functools import partial

import speech_recognition as sr
import numpy as np
import sounddevice as sd
import librosa
import threading
import queue

# Constants
SAMPLE_RATE = 44100
CHANNELS = 1
WINDOW_DURATION = 0.5
STEP_DURATION = 0.1
BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
STEP_SIZE = int(SAMPLE_RATE * STEP_DURATION)



audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

# Midi dictionary for notes
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

# Audio callback for continuous input
def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata.flatten()

# Analyze pitch using Librosa
def analyze_pitch(audio_data):
    pitches, magnitudes, probs = librosa.pyin(
        audio_data,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('A5'),
        sr=SAMPLE_RATE,
        n_thresholds=100
    )
    detected_notes = [librosa.hz_to_note(pitch) for pitch, mag in zip(pitches, magnitudes) if mag]
    detected_pitches = [pitch for pitch, mag in zip(pitches, magnitudes) if mag]

    if len(detected_notes) <= 10 or probs[np.argmax(probs)] < 0.99:
        return [], ""
    else:
        average_freq = sum(detected_pitches) / len(detected_pitches)
        return average_freq, librosa.hz_to_note(average_freq)

# Calculate gradient color based on frequency difference
def calculate_color(detected_freq, target_freq, max_diff):
    diff = abs(detected_freq - target_freq)
    if diff > max_diff:
        diff = max_diff

    # Linear interpolation between red (off) and green (in tune)
    ratio = diff / max_diff
    red = int(255 * ratio)
    green = 255 - red
    if detected_freq > target_freq:
        return (red / 255.0, green / 255.0, 0, 1)
    else:
        return (0, green / 255.0, red / 255.0, 1)  # RGBA format

def generate_tune(to_hit_freq, curr_freq, duration=0.7, sr=44100):
    #print(to_hit_freq, curr_freq)
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

"""def p_s(tune):
    sd.play(tune, samplerate=44100)
    sd.wait()"""

playback_queue = queue.Queue()

# Buffer size for the stream
BUFFER_SIZE = 30869  # Adjust as needed for smoother playback

# Playback thread flag
playback_active = threading.Event()

# Global output stream
output_stream = None

def playback_worker():
    global output_stream
    while playback_active.is_set():
        if not playback_queue.empty():
            tune = playback_queue.get()
            if tune is None:  # Sentinel value to terminate playback
                break
            #print(tune)
            # Chunked writing to the audio stream
            tune_idx = 0
            #print("TUNE LEN ", len(tune))
            while tune_idx < len(tune):
                chunk = tune[tune_idx:tune_idx + BUFFER_SIZE]
                output_stream.write(chunk)
                tune_idx += BUFFER_SIZE

            #sleep for 0.7 seconds
            time.sleep(0.7)

            playback_queue.task_done()
        else:
            # Sleep briefly to reduce CPU usage when the queue is empty
            sd.sleep(10)

# Function to initialize and start the playback system
def start_playback_system():
    global output_stream
    playback_active.set()

    # Initialize the output stream
    output_stream = sd.OutputStream(samplerate=44100, channels=1, dtype='float32', blocksize=BUFFER_SIZE)
    output_stream.start()

    # Start the playback thread
    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start()

# Function to stop the playback system
def stop_playback_system():
    global output_stream
    playback_active.clear()
    playback_queue.put(None)  # Signal the playback thread to stop
    if output_stream:
        output_stream.stop()
        output_stream.close()

# Function to add a tune to the playback queue
def p_s(tune):
    # Normalize the tune to ensure it's within the valid range for audio
    tune = np.clip(tune, -1.0, 1.0).astype('float32')
    if playback_queue.unfinished_tasks == 0:
        #print(playback_queue.unfinished_tasks)
        #print("PUTTING TUNE")
        playback_queue.put(tune)

def generate_kaching_sound(duration=0.35, sample_rate=44100):
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


# KivyMD UI
KV = '''
<MyScreen>:
    md_bg_color: root.bg_color

    BoxLayout:
        orientation: 'vertical'
        padding: dp(20)
        spacing: dp(10)

        MDLabel:
            id: label
            text: "PLACEHOLDER"
            halign: "center"
            font_style: "H5"



        MDRaisedButton:
            text: "Exit"
            pos_hint: {"center_x": 0.5}
            on_release: app.stop_app()
'''

class MyScreen(MDScreen):
    bg_color = ListProperty([1, 1, 1, 1])  # Default to white

class GuitarTunerApp(MDApp):
    def build(self):
        self.title = "Voice-Controlled Guitar Tuner"
        self.tuning = False
        self.midi_to_hit = -1
        self.to_hit_freq = 0
        self.max_dif_top = 0
        self.max_dif_bottom = 0
        self.no_input = 0
        self.audio_stream = None
        Builder.load_string(KV)
        self.screen = MyScreen()
        self.tune_clock = None
        return self.screen

    def start_tuning(self, switch_text, dt):
        print("Starting tuning...")
        if switch_text:
            self.root.ids.label.text = "Listening for tuning instructions..."
        self.screen.bg_color = [1, 1, 1, 1]
        Clock.schedule_once(self.detect_voice, 0)

    def detect_voice(self, dt):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                audio_data = recognizer.listen(source)
                text = recognizer.recognize_google(audio_data)
                if text in tone_midi_guitar_dict:
                    self.tuning = True
                    self.midi_to_hit = tone_midi_guitar_dict[text]
                    self.to_hit_freq = librosa.midi_to_hz(self.midi_to_hit)
                    self.max_dif_top = abs(self.to_hit_freq - librosa.midi_to_hz(self.midi_to_hit + 1))
                    self.max_dif_bottom = abs(self.to_hit_freq - librosa.midi_to_hz(self.midi_to_hit - 1))
                    self.root.ids.label.text = f"Tuning to {text}..."
                    self.tune_clock = Clock.schedule_interval(self.tune, 0.05)
                elif text.lower() == "exit":
                    self.stop_app()
                else:
                    print(text)
                    self.root.ids.label.text = "Invalid note. Try again."
                    Clock.schedule_once(partial(self.start_tuning, False), 0.7)
                    return
            except sr.UnknownValueError:
                self.root.ids.label.text = "Sorry, I did not understand."
                Clock.schedule_once(partial(self.start_tuning, True), 0.7)
            except sr.RequestError as e:
                self.root.ids.label.text = f"API error: {e}"

    def tune(self, dt):
        detected_freqs, detected_notes = analyze_pitch(audio_buffer)
        #print(detected_freqs, detected_notes)
        if not detected_notes:
            self.no_input += 1
            if self.no_input > 20:
                self.tuning = False
                self.root.ids.label.text = "No input detected. Exiting tuning mode."
                self.tune_clock.cancel()
                Clock.schedule_once(partial(self.start_tuning, True), 1)
                self.no_input = 0
        else:
            self.no_input = 0
            diff = detected_freqs - self.to_hit_freq
            top_safe_level = self.to_hit_freq + 0.1*self.max_dif_top
            bottom_safe_level = self.to_hit_freq - 0.1*self.max_dif_bottom
            tune = generate_static_tune(self.to_hit_freq, detected_freqs)
            if detected_freqs > bottom_safe_level and detected_freqs < top_safe_level:#detected_notes == librosa.midi_to_note(self.midi_to_hit):
                self.root.ids.label.text = f"In tune: {detected_notes}"
                self.screen.bg_color = calculate_color(detected_freqs, self.to_hit_freq, max(self.max_dif_bottom, self.max_dif_top))
                tune = generate_kaching_sound()
                p_s(tune)
            elif detected_notes == librosa.midi_to_note(self.midi_to_hit + 12):
                self.tuning = False
                self.tune_clock.cancel()
                Clock.schedule_once(partial(self.start_tuning, True), 0.7)
            elif diff > 0:
                self.root.ids.label.text = "Tune down"
                self.screen.bg_color = calculate_color(detected_freqs, self.to_hit_freq, self.max_dif_top)
                p_s(tune)
            elif diff < 0:
                self.root.ids.label.text = "Tune up"
                self.screen.bg_color = calculate_color(detected_freqs, self.to_hit_freq, self.max_dif_bottom)
                p_s(tune)


            #diff = detected_freqs - self.to_hit_freq
            #max_diff = max(self.max_dif_top, self.max_dif_bottom)
            #self.screen.bg_color = calculate_color(detected_freqs, self.to_hit_freq, max_diff)

    def stop_app(self):
        if self.audio_stream:
            self.audio_stream.stop()
        stop_playback_system()
        self.stop()

    def on_start(self):
        self.audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, blocksize=STEP_SIZE
        )
        self.audio_stream.start()
        start_playback_system()
        Clock.schedule_once(partial(self.start_tuning, True), 1.0)



if __name__ == "__main__":
    GuitarTunerApp().run()
