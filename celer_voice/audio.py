"""
Audio DSP, Mel-Spectrogram extraction, and robust CPU Vocoder.
Dioptimalkan khusus untuk 16kHz audio tanpa DC offset dan bebas distorsi di Intel Celeron.
"""

import math
import os
import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
import torch
import winsound

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512
N_MELS = 80
FMIN = 50.0
FMAX = 7600.0


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS,
                         fmin: float = FMIN, fmax: float = FMAX) -> torch.Tensor:
    """Build triangular Mel filterbank matrix of shape [n_mels, n_fft // 2 + 1]."""
    weights = np.zeros((n_mels, int(n_fft // 2 + 1)), dtype=np.float32)
    m_min = hz_to_mel(fmin)
    m_max = hz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    h_pts = [mel_to_hz(m) for m in m_pts]
    bins = np.floor((n_fft + 1) * np.array(h_pts) / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center > left:
            for j in range(left, center):
                weights[i, j] = (j - left) / (center - left)
        if right > center:
            for j in range(center, right):
                weights[i, j] = (right - j) / (right - center)

    enorm = 2.0 / (np.array(h_pts[2:n_mels + 2]) - np.array(h_pts[:n_mels]))
    weights *= enorm[:, np.newaxis]
    return torch.tensor(weights, dtype=torch.float32)


MEL_BASIS = build_mel_filterbank()
MEL_BASIS_PINV = torch.pinverse(MEL_BASIS)


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file, convert to mono float32 normalized to [-1, 1], resample to target_sr."""
    sr, data = wavfile.read(path)
    
    if len(data.shape) > 1:
        data = data.mean(axis=1)
        
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
        
    if sr != target_sr:
        import scipy.signal as signal
        num_samples = int(len(data) * float(target_sr) / sr)
        data = signal.resample(data, num_samples).astype(np.float32)
        
    # Remove DC offset
    data = data - np.mean(data)
    max_val = np.max(np.abs(data))
    if max_val > 1e-4:
        data = data / max_val * 0.95
        
    return data


def save_audio(path: str, wav: np.ndarray, sr: int = SAMPLE_RATE):
    """Save float32 audio as 16-bit PCM WAV with peak normalization."""
    wav = wav - np.mean(wav)
    peak = np.percentile(np.abs(wav), 99.8)
    if peak > 1e-4:
        wav = np.clip(wav / peak * 0.9, -1.0, 1.0)
    wav_int16 = (wav * 32767.0).astype(np.int16)
    wavfile.write(path, sr, wav_int16)


def play_audio(wav_or_path, sr: int = SAMPLE_RATE):
    """
    Play audio through default speakers with stereo duplication and fallback.
    Ensures clear sound on any laptop speaker / headphone configuration.
    """
    if isinstance(wav_or_path, str):
        if not os.path.exists(wav_or_path):
            print(f"File audio tidak ditemukan: {wav_or_path}")
            return
        abs_path = os.path.abspath(wav_or_path)
        sr, data = wavfile.read(abs_path)
        wav = (data.astype(np.float32) / 32768.0) if data.dtype == np.int16 else data
    else:
        wav = wav_or_path
        abs_path = None

    if wav.ndim == 1:
        # Duplicate mono to stereo (N, 2) for multi-channel compatibility
        stereo_wav = np.column_stack([wav, wav])
    else:
        stereo_wav = wav

    played = False
    try:
        sd.play(stereo_wav, samplerate=sr)
        sd.wait()
        played = True
    except Exception as e:
        print(f"sounddevice playback notice: {e}")

    # Fallback to winsound if sounddevice did not complete
    if not played and abs_path and os.path.exists(abs_path):
        try:
            winsound.PlaySound(abs_path, winsound.SND_FILENAME)
            played = True
        except Exception as e:
            print(f"winsound playback notice: {e}")


def wav_to_mel(wav: np.ndarray) -> torch.Tensor:
    """
    Extract clean log Mel-spectrogram from raw waveform.
    Returns: Tensor of shape [n_mels, frames]
    """
    if isinstance(wav, np.ndarray):
        wav = torch.tensor(wav, dtype=torch.float32)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
        
    window = torch.hann_window(WIN_LENGTH)
    stft = torch.stft(
        wav,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        pad_mode='reflect',
        return_complex=True
    )
    
    magnitude = torch.abs(stft).squeeze(0)  # [n_fft//2 + 1, frames]
    mel = torch.matmul(MEL_BASIS, magnitude)  # [n_mels, frames]
    log_mel = torch.log(torch.clamp(mel, min=1e-5))
    return log_mel


def mel_to_wav(log_mel: torch.Tensor, n_iter: int = 24) -> np.ndarray:
    """
    Clean, robust Griffin-Lim phase reconstruction from log Mel-spectrogram on CPU.
    Produces high volume, crisp audio with zero DC offset.
    """
    if isinstance(log_mel, np.ndarray):
        log_mel = torch.tensor(log_mel, dtype=torch.float32)
        
    mel = torch.exp(log_mel.cpu())
    linear_mag = torch.matmul(MEL_BASIS_PINV, mel)
    linear_mag = torch.clamp(linear_mag, min=0.0)
    
    # Suppress DC frequency bin and lowest sub-bass (<60Hz) to prevent clicks
    linear_mag[:2, :] = 0.0
    
    window = torch.hann_window(WIN_LENGTH)
    angles = torch.exp(2j * np.pi * torch.rand(linear_mag.shape))
    spec = linear_mag.to(torch.complex64) * angles
    
    for _ in range(n_iter):
        wav = torch.istft(
            spec,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
            center=True
        )
        rebuilt = torch.stft(
            wav,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
            center=True,
            return_complex=True
        )
        spec = linear_mag * torch.exp(1j * torch.angle(rebuilt))
        
    wav_out = torch.istft(
        spec,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True
    ).squeeze().numpy()
    
    # Smooth fade-in (16ms) to eliminate boundary clicks
    fade_len = min(256, len(wav_out) // 2)
    fade_in = np.sin(np.linspace(0, np.pi / 2, fade_len)) ** 2
    wav_out[:fade_len] *= fade_in
    wav_out[-fade_len:] *= fade_in[::-1]
    
    # Volume normalization
    peak = np.percentile(np.abs(wav_out), 99.8)
    if peak > 1e-4:
        wav_out = np.clip(wav_out / peak * 0.9, -1.0, 1.0)
        
    return wav_out.astype(np.float32)
