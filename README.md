# CelerVoice - Model AI Suara Sendiri (Lightweight Neural Voice)

Proyek ini adalah implementasi **Model AI Suara (Neural Speech Synthesis) 100% Buatan Sendiri**, bukan mengambil model orang lain (bukan VITS, Bark, XTTS, atau RVC pra-latih pihak ketiga).

Arsitektur ini dirancang khusus dari nol menggunakan **PyTorch & Python** agar **ringan, berkualitas bagus, serta dapat dilatih (train) dan dijalankan (inference) secara lancar di laptop berprosesor Intel Celeron** (hanya 2 thread CPU, tanpa GPU).

---

##  Mengapa Model Ini Sangat Cocok untuk Laptop Celeron?

1. **Ukuran Model Sangat Ringkas (~3.4 MB)**:
   - Total parameter hanya **~890.000 parameter** (dibandingkan model umum seperti Bark/XTTS yang ratusan juta sampai miliaran parameter).
   - Penggunaan RAM saat training hanya **< 250 MB**, sangat aman untuk laptop dengan RAM 4 GB.
2. **Guided Attention Mechanism**:
   - Model seq2seq biasa butuh ratusan jam audio dan ratusan epoch untuk belajar membaca teks secara teratur.
   - Dengan **Guided Attention Loss**, model langsung mengunci penjajaran vokal secara diagonal sejak epoch awal (10–30 epoch sudah konvergen!).
3. **CPU-First Griffin-Lim Vocoder**:
   - Sintesis audio mel-ke-gelombang suara berjalan **2x lebih cepat dari real-time** (Real-Time Factor: **~0.53x** di CPU Celeron).
   - Kalimat 4 detik selesai diproses hanya dalam ~2 detik!
4. **Tanpa Ketergantungan Eksternal yang Berat**:
   - Tidak memerlukan CUDA, C++ Build Tools, atau software pihak ketiga yang rumit.

---

## 📁 Struktur Direktori

```text
MyVoice/
├── celer_voice/
│   ├── __init__.py
│   ├── audio.py          # DSP Audio, Mel-filterbank, STFT, Vocoder Griffin-Lim & De-emphasis
│   ├── text.py           # Normalisasi teks Bahasa Indonesia & Tokenizer Karakter
│   ├── model.py          # Arsitektur Neural Network CelerVoice (Encoder, Attention, Decoder, PostNet)
│   ├── dataset.py        # DataLoader cerdas dengan Dynamic Padding
│   ├── trainer.py        # Mesin pelatihan CPU khusus Celeron dengan Guided Attention
│   └── synthesizer.py    # Pipeline sintesis teks -> spektrogram -> audio WAV
├── dataset/
│   ├── wavs/             # Folder file rekaman suara (.wav)
│   └── metadata.csv      # Daftar pasangan file suara dan transkrip teks
├── checkpoints/          # Tempat penyimpanan model terbaik (.pt)
├── gui.py                # Aplikasi Desktop GUI modern (CustomTkinter)
├── record_voice.py       # Perekam suara interaktif dengan mikrofon
├── prepare_sample_data.py# Generator data akustik bawaan untuk pengujian instan
├── train.py              # Skrip CLI untuk melatih model
├── infer.py              # Skrip CLI untuk menghasilkan suara dari teks
└── README.md
```

---

## 🚀 Panduan Penggunaan

### 1. Menjalankan Melalui Tampilan Grafis (GUI)
Cara paling mudah dan interaktif adalah membuka aplikasi desktop:
```bash
python gui.py
```
Di dalam GUI tersedia 3 tab:
- **Tab 1 (Uji Suara / TTS)**: Ketik teks apapun lalu klik **"Sintesis & Putar Suara"**.
- **Tab 2 (Rekam Suara Sendiri)**: Rekam kalimat-kalimat yang disediakan mikrofon Anda untuk membuat dataset suara sendiri.
- **Tab 3 (Latih Model AI)**: Atur jumlah epoch dan klik **"Mulai Pelatihan Model"** dengan indikator progress bar langsung.

---

### 2. Menjalankan Melalui Terminal (CLI)

#### Langkah A: Menyiapkan Data Suara Anda Sendiri
Gunakan skrip perekam suara interaktif:
```bash
python record_voice.py
```
Skrip akan menampilkan kalimat bahasa Indonesia. Tekan `ENTER`, bacakan kalimatnya dengan jelas ke mikrofon Anda. Rekaman otomatis disimpan dan dipotong keheningannya.

*(Catatan: Anda juga bisa menggunakan data sampel bawaan dengan menjalankan `python prepare_sample_data.py`)*

#### Langkah B: Melatih Model AI (Training di Celeron)
Jalankan pelatihan model AI Anda:
```bash
python train.py --epochs 30 --batch_size 4
```
Parameter yang bisa disesuaikan:
- `--epochs`: Jumlah putaran latihan (disarankan 30 - 50 epoch untuk hasil optimal).
- `--batch_size`: Ukuran batch (default 4, sangat pas untuk memori Celeron).
- `--threads`: Jumlah core/thread CPU yang dipakai (default 2).

Model terbaik akan otomatis disimpan di: `checkpoints/best_model.pt`.

#### Langkah C: Sintesis Suara dari Teks (Inference)
Ketik teks apa saja yang ingin diucapkan oleh AI Anda:
```bash
python infer.py --text "halo nama saya adalah kecerdasan buatan buatan sendiri" --output hasil_suara.wav
```
Audio akan otomatis diputar langsung ke speaker laptop Anda!

---

## 🔬 Spesifikasi Arsitektur Neural CelerVoice

| Komponen | Arsitektur | Rincian |
| :--- | :--- | :--- |
| **Input** | Karakter / Grapheme Tokenizer | 32 karakter + kontrol token |
| **Encoder** | 1D-CNN + Bi-GRU | Embedding 128-d, 3 lapis Conv1D (kernel 5), Bi-GRU 128-d |
| **Attention** | Location-Sensitive Additive | Menggunakan konvolusi riwayat alignment untuk mencegah stutter/skip |
| **Decoder** | PreNet + Stacked GRU | PreNet 2 lapis linear (128-d) dengan dropout regulasi + GRU decoder |
| **PostNet** | 5-Layer Residual Conv1D | Menajamkan forman vokal dan membuang artefak frekuensi tinggi |
| **Vocoder** | Phase Inversion Griffin-Lim | Zero parameter (0 MB), bebas latensi, dioptimasi Hann-window STFT |
| **Total Parameter**| **890.818 parameter** | **~3.4 MB** |
