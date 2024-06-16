import sounddevice as sd
import tkinter as tk
from tkinter import messagebox
import wavio


def record_audio(filename, duration=5, fs=16000):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)
    print("Recording finished\n*************************************************\nProcessing...")


def start_recording():
    try:
        record_audio('user_cough.ogg', duration=5)
        messagebox.showinfo("Recording", "Recording finished successfully")
    except Exception as e:
        messagebox.showerror("Error", "Error occurred while recording")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sound Recorder")
    root.geometry("300x300")
    root.resizable(False, False)

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack()

    button = tk.Button(root, text="Start Recording", command=start_recording)
    button.pack(pady=20)

    root.mainloop()
