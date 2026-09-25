"""
Inference engine and speech synthesizer for CelerVoice.
Converts text into natural audio waveform completely on CPU.
"""

import os
import time
import numpy as np
import torch

from .text import TextTokenizer
from .audio import mel_to_wav, save_audio, play_audio, SAMPLE_RATE
from .model import CelerVoiceTTS


class Synthesizer:
    """
    High-level Speech Synthesizer.
    Usage:
        synth = Synthesizer("checkpoints/best_model.pt")
        synth.speak("Halo selamat pagi", "output.wav", play=True)
    """
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = TextTokenizer()
        self.model = CelerVoiceTTS().to(self.device)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded CelerVoice checkpoint from {model_path}")
        else:
            print("Initialized fresh CelerVoice model (untrained or using random weights).")
            
        self.model.eval()

    def synthesize(self, text: str, max_steps: int = 500, vocoder_iters: int = 24) -> tuple[np.ndarray, dict]:
        """
        Synthesize text to audio waveform.
        Returns: (audio_waveform, metadata)
        """
        t0 = time.time()
        tokens = self.tokenizer.text_to_ids(text)
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            outputs = self.model(token_tensor, mel_targets=None, max_decoder_steps=max_steps)
            mel_final = outputs["mel_final"].squeeze(0)  # [n_mels, frames]
            
        t_model = time.time() - t0
        
        # Vocoder
        t1 = time.time()
        wav = mel_to_wav(mel_final, n_iter=vocoder_iters)
        t_vocoder = time.time() - t1
        
        total_time = t_model + t_vocoder
        audio_duration = len(wav) / SAMPLE_RATE
        rtf = total_time / max(audio_duration, 1e-4)  # Real-Time Factor (<1.0 means faster than real-time!)
        
        info = {
            "model_time": t_model,
            "vocoder_time": t_vocoder,
            "total_time": total_time,
            "audio_duration": audio_duration,
            "rtf": rtf,
            "frames": mel_final.size(1),
            "alignments": outputs["alignments"].squeeze(0).cpu().numpy()
        }
        return wav, info

    def speak(self, text: str, output_path: str = "output.wav", play: bool = True, vocoder_iters: int = 24) -> str:
        """
        Synthesize text, save to WAV, and play through speakers.
        """
        wav, info = self.synthesize(text, vocoder_iters=vocoder_iters)
        save_audio(output_path, wav, sr=SAMPLE_RATE)
        print(f"Generated {info['audio_duration']:.2f}s audio in {info['total_time']:.2f}s (RTF: {info['rtf']:.2f}x) -> {output_path}")
        
        if play:
            play_audio(wav, sr=SAMPLE_RATE)
                
        return output_path
