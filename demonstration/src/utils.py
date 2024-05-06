import torch
import librosa
import numpy as np
import soundfile as sf
import sounddevice as sd
from tkinter import filedialog
from tkinter import messagebox

import torch.nn.functional as F
from pygame.mixer import music, init


from parse_config import config
from model_network import SpeechEmbedder

def show_audio(audio_file):
    init()
    music.load(audio_file)
    music.play(loops=0)

def STFT_mel_audio_data(data_config, raw_utter):
    S = librosa.stft(raw_utter, n_fft=data_config.nfft, 
                     hop_length=int(data_config.hop * data_config.sr),
                     win_length=int(data_config.window * data_config.sr))
    S = np.abs(S) ** 2
    mel_basis = librosa.filters.mel(sr=data_config.sr, n_fft=data_config.nfft, n_mels=data_config.nmels)
    S = np.log10(np.dot(mel_basis, S) + 1e-6).T
    assert S.shape[0] >= data_config.frame, f"can't generate {data_config.frame} frames."
    return S 

def pre_preprocess_tdsv(data_config, utter_path):
    utter, _ =  librosa.load(utter_path, sr=data_config.sr)
    utter_trim, _ = librosa.effects.trim(utter, top_db=30)
    duration = len(utter_trim)/data_config.sr
    utter_strech = librosa.effects.time_stretch(utter_trim, rate=duration/1)
    S = STFT_mel_audio_data(data_config, utter_strech)
    S = S[:data_config.frame, :]
    return S

def pre_preprocess_tisv(data_config, utter_path):
    utter_min_len = (data_config.frame * data_config.hop + data_config.window) * data_config.sr
    utter, _ =  librosa.load(utter_path, sr=data_config.sr)
    intervals = librosa.effects.split(utter, top_db=30)
    utter_spec = []
    for interval in intervals:
        if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
            S = STFT_mel_audio_data(data_config, utter)
            utter_spec.append(S[:data_config.frame])
            utter_spec.append(S[-data_config.frame:])

    return np.array(utter_spec)

def check_similarity(task, enrollment_path, evaluation_path, tkinter_frame):

    if task == "tdsv":
        data_config = config.data.TD_SV_data
        model_path = "./demonstration/models/tdsv/tdsv_30_4_3_10_2000.model"
    elif task == "tisv":
        data_config = config.data.TI_SV_data
        model_path = "./demonstration/models/tisv/optim_tisv_30_4_9_10_1000.model"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    speech_embedder = SpeechEmbedder().to(device)
    speech_embedder.load_state_dict(torch.load(model_path)['speech_embedder'])
    speech_embedder.eval()

    enrollment_utterance = pre_preprocess_tdsv(data_config, enrollment_path)
    evaluation_utterance = pre_preprocess_tdsv(data_config, evaluation_path)
    
    enrollment_utterance = torch.from_numpy(enrollment_utterance).to(device)
    evaluation_utterance = torch.from_numpy(evaluation_utterance).to(device)

    if task == "tdsv":
        frames, nmels = 80, 40
        enrollment_utterance = enrollment_utterance.reshape(1, frames, nmels)
        evaluation_utterance = evaluation_utterance.reshape(1, frames, nmels)

    enrollment_embedding = speech_embedder(enrollment_utterance)
    evaluation_embedding = speech_embedder(evaluation_utterance)

    similarity = F.cosine_similarity(enrollment_embedding, evaluation_embedding)
    similarity = similarity.cpu().detach().numpy()[0]

    print(similarity)
    notification = "These utterances are NOT similar" if similarity < 0.8 else "These utterances are similar"
    # return tkinter_frame.config(text=f"similarity: {similarity*100: .2f}%")
    return tkinter_frame.insert(0, notification)

def record(mode, typ):
    if mode == "TDSV":
        duration = 2
        sr = 48000  
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=2)
        sd.wait() 
        if typ == "enroll":
            filename = "./demonstration/data/tdsv/enrollment_audio.wav"
            sf.write(filename, recording, sr)
            messagebox.showinfo("Enrollment audio saved")
        elif typ == "eval":
            filename = "./demonstration/data/tdsv/evaluation_audio.wav"
            sf.write(filename, recording, sr)
            messagebox.showinfo("Evaluation audio saved")

    elif mode == "TISV":
        duration = 4
        sr = 22050 
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=2)
        sd.wait() 
        if typ == "enroll":
            filename = "./demonstration/data/tisv/enrollment_audio.wav"
            sf.write(filename, recording, sr)
            messagebox.showinfo("Enrollment audio saved")
        elif typ == "eval":
            filename = "./demonstration/data/tisv/evaluation_audio.wav"
            sf.write(filename, recording, sr)
            messagebox.showinfo("Evaluation audio saved")
        
def upload_file(mode, typ):
    filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    audio, sr = librosa.load(filename, sr=16000) 
    
    if mode == "TDSV":
        if typ == "enroll":
            filename = "./demonstration/data/tdsv/enrollment_audio.wav"
            sf.write(filename, audio, sr)
            messagebox.showinfo("Enrollment audio saved")
        elif typ == "eval":
            filename = "./demonstration/data/tdsv/evaluation_audio.wav"
            sf.write(filename, audio, sr)
            messagebox.showinfo("Evaluation audio saved")

    elif mode == "TISV":
        if typ == "enroll":
            filename = "./demonstration/data/tisv/enrollment_audio.wav"
            sf.write(filename, audio, sr)
            messagebox.showinfo("Enrollment audio saved")
        elif typ == "eval":
            filename = "./demonstration/data/tisv/evaluation_audio.wav"
            sf.write(filename, audio, sr)
            messagebox.showinfo("Evaluation audio saved")
