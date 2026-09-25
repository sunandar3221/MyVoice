"""
Interactive Voice Recorder for CelerVoice.
Memungkinkan pengguna merekam suaranya sendiri dengan mikrofon untuk melatih model AI.
"""

import os
import csv
import sys
import time
import numpy as np
import sounddevice as sd
from celer_voice.audio import save_audio, SAMPLE_RATE

# Prompt kalimat bahasa Indonesia untuk merekam suara sendiri
PROMPT_SENTENCES = [
    "halo, ini adalah rekaman suara asli saya sendiri.",
    "saya sedang melatih model kecerdasan buatan di komputer saya.",
    "teknologi pemrosesan suara ini sangat ringan dan cepat.",
    "hari ini cuaca cukup cerah dan menyenangkan untuk belajar hal baru.",
    "kecerdasan buatan dapat menirukan intonasi dan gaya bicara kita.",
    "prosesor laptop ini mampu memproses audio dengan sangat lancar.",
    "bahasa indonesia memiliki pelafalan kata yang jelas dan mudah dipelajari.",
    "setiap rekaman akan membantu model memahami karakter vokal saya.",
    "mari kita coba mendengarkan hasil dari model suara buatan sendiri.",
    "selamat datang di era kecerdasan buatan lokal tanpa internet.",
    "selamat pagi semuanya, semoga hari kalian menyenangkan.",
    "jangan lupa untuk selalu bersyukur dan tetap semangat beraktivitas.",
    "komputer ini bekerja keras mengolah data suara menjadi model digital.",
    "kombinasi matematika dan sinyal audio menghasilkan suara yang alami.",
    "terima kasih sudah mendengarkan rekaman suara saya hari ini."
]

def record_audio_clip(duration_seconds: int = 5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Record audio from default microphone for specified duration."""
    print(f"  [>] Merekam selama {duration_seconds} detik... Bicara sekarang!")
    audio = sd.rec(int(duration_seconds * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("  [✓] Selesai merekam.")
    return audio.flatten()


def trim_silence(audio: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """Potong keheningan (silence) di awal dan akhir audio."""
    energy = np.abs(audio)
    non_silent = np.where(energy > threshold)[0]
    if len(non_silent) == 0:
        return audio
    start = max(0, non_silent[0] - int(0.1 * SAMPLE_RATE))
    end = min(len(audio), non_silent[-1] + int(0.1 * SAMPLE_RATE))
    return audio[start:end]


def main():
    wavs_dir = os.path.join("dataset", "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    metadata_path = os.path.join("dataset", "metadata.csv")

    print("=" * 65)
    print("        STUDIO PEREKAM SUARA PENGGUNA (CELERVOICE RECORDER)       ")
    print("=" * 65)
    print("Panduan:")
    print("1. Gunakan mikrofon atau headset yang jernih di tempat yang tenang.")
    print("2. Bacalah kalimat yang muncul di layar dengan artikulasi wajar.")
    print("3. Rekaman akan otomatis disimpan ke dataset/wavs dan metadata.csv.")
    print("=" * 65)

    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        dev_name = devices[default_in]['name'] if default_in >= 0 else "Default"
        print(f"Mikrofon terdeteksi: {dev_name}\n")
    except Exception as e:
        print(f"Catatan mikrofon: {e}\n")

    # Read existing metadata count
    existing_count = 0
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            existing_count = sum(1 for line in f if line.strip())

    start_idx = existing_count + 1

    for i, sentence in enumerate(PROMPT_SENTENCES):
        idx = start_idx + i
        filename = f"user_{idx:03d}.wav"
        rel_wav_path = f"wavs/{filename}"
        abs_wav_path = os.path.join("dataset", "wavs", filename)

        print("-" * 65)
        print(f"Kalimat [{i+1}/{len(PROMPT_SENTENCES)}]:")
        print(f"  >>> \"{sentence}\" <<<")
        print("-" * 65)
        
        while True:
            cmd = input("Tekan [ENTER] untuk mulai rekam (atau ketik 's' untuk lewati, 'q' untuk keluar): ").strip().lower()
            if cmd == 'q':
                print("\nSelesai merekam. Anda dapat melatih model dengan: python train.py")
                return
            if cmd == 's':
                print("Dilewati.")
                break

            # Record
            raw_audio = record_audio_clip(duration_seconds=5)
            trimmed = trim_silence(raw_audio)

            # Check energy
            if np.max(np.abs(trimmed)) < 0.05:
                print("  [!] Peringatan: Suara terlalu pelan atau tidak terdengar. Coba lagi!")
                continue

            # Save
            save_audio(abs_wav_path, trimmed, sr=SAMPLE_RATE)
            print(f"  [+] Tersimpan di: {abs_wav_path}")

            # Append to metadata.csv
            with open(metadata_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='|')
                writer.writerow([rel_wav_path, sentence])

            print(f"  [+] Ditambahkan ke: {metadata_path}")
            break

    print("\n" + "=" * 65)
    print("Selamat! Rekaman suara Anda telah berhasil disimpan ke dataset.")
    print("Sekarang jalankan perintah berikut untuk melatih model AI suara Anda:")
    print("   python train.py --epochs 50")
    print("=" * 65)

if __name__ == "__main__":
    main()
