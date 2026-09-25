"""
Generator dataset awal untuk CelerVoice.
Menghasilkan sample audio berbasis sintesis akustik formants bahasa Indonesia
sehingga model dapat langsung diuji dan dilatih di laptop Celeron.
"""

import os
import csv
import numpy as np
import scipy.signal as signal
from celer_voice.audio import save_audio, SAMPLE_RATE

# Indonesian sample sentences
SAMPLE_SENTENCES = [
    "halo selamat datang di sistem kecerdasan buatan",
    "model ini dilatih sepenuhnya di komputer lokal anda",
    "arsitektur ringan dan hemat daya untuk prosesor celeron",
    "teknologi pemrosesan sinyal suara berbasis kecerdasan buatan",
    "anda dapat merekam suara anda sendiri untuk kloning vokal",
    "belajar bahasa dan sintesis suara secara cepat",
    "kami mengembangkan model sendiri tanpa dependensi rumit",
    "suara jernih dan lancar menggunakan filterbank mel",
    "laptop celeron mampu menjalankan pelatihan model ini",
    "selamat mencoba dan berkreasi dengan suara anda"
]

# Formant frequencies (F1, F2, F3 in Hz) for Indonesian vowels
VOWEL_FORMANTS = {
    'a': (850, 1610, 2850),
    'i': (280, 2250, 2890),
    'u': (320, 800, 2240),
    'e': (530, 1840, 2480),
    'o': (500, 1000, 2500),
    ' ': (0, 0, 0)
}

def synthesize_acoustic_word(text: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Sintesis suara akustik berbasis formant dan glottal pulse.
    Menghasilkan sinyal fonetik yang menyerupai vokal bicara manusia.
    """
    audio_chunks = []
    f0 = 135.0  # Fundamental frequency (Hz)
    
    for char in text.lower():
        duration = 0.08 if char in "aiueo" else 0.05
        if char == " ":
            duration = 0.12
            silence = np.zeros(int(duration * sr), dtype=np.float32)
            audio_chunks.append(silence)
            continue
            
        num_samples = int(duration * sr)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Voiced glottal pulse train
        pulse_period = int(sr / f0)
        excitation = np.zeros(num_samples, dtype=np.float32)
        excitation[::pulse_period] = 1.0
        
        # Add slight pitch jitter and turbulence noise
        noise = np.random.normal(0, 0.05, num_samples).astype(np.float32)
        excitation = excitation + noise
        
        # Get formants for character (default to 'a' if consonant)
        f1, f2, f3 = VOWEL_FORMANTS.get(char, (500, 1500, 2500))
        
        # Formant resonant filters (2nd order bandpass / resonator)
        def formant_filter(data, freq, bw):
            if freq <= 0:
                return data
            w0 = 2 * np.pi * freq / sr
            r = np.exp(-np.pi * bw / sr)
            b = [1 - r]
            a = [1, -2 * r * np.cos(w0), r * r]
            return signal.lfilter(b, a, data)
            
        filtered = formant_filter(excitation, f1, 80)
        filtered = formant_filter(filtered, f2, 100) + 0.5 * filtered
        filtered = formant_filter(filtered, f3, 120) + 0.3 * filtered
        
        # Envelope window to avoid clicks
        win = np.hanning(num_samples)
        chunk = (filtered * win).astype(np.float32)
        audio_chunks.append(chunk)
        
    full_audio = np.concatenate(audio_chunks)
    peak = np.max(np.abs(full_audio))
    if peak > 0:
        full_audio = full_audio / peak * 0.9
    return full_audio


def setup_sample_dataset(output_dir: str = "dataset"):
    wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.csv")
    
    print(f"Membuat dataset sample di: {output_dir}")
    rows = []
    
    for i, text in enumerate(SAMPLE_SENTENCES):
        filename = f"sample_{i+1:03d}.wav"
        file_path = os.path.join(wavs_dir, filename)
        
        wav = synthesize_acoustic_word(text)
        save_audio(file_path, wav, sr=SAMPLE_RATE)
        
        # metadata format: relative_path|text
        rel_path = os.path.join("wavs", filename)
        rows.append((rel_path, text))
        print(f"  [+] Dibuat: {rel_path} -> '{text}'")
        
    with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='|')
        for row in rows:
            writer.writerow(row)
            
    print(f"\nBerhasil membuat {len(rows)} audio sample dan metadata di: {metadata_path}")
    print("Sekarang Anda dapat menjalankan pelatihan dengan: python train.py")


if __name__ == "__main__":
    setup_sample_dataset()
