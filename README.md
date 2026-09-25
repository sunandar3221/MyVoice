# 🎙️ CelerVoice - Model AI Suara Buatan Sendiri (Lightweight Neural Voice)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunandar3221/MyVoice/blob/master/Train_CelerVoice_Colab.ipynb)

Proyek ini adalah implementasi **Model AI Suara (Neural Speech Synthesis) 100% Buatan Sendiri**, bukan mengambil atau meminjam model orang lain (bukan VITS, Bark, XTTS, atau RVC pra-latih pihak ketiga).

Arsitektur ini dirancang khusus menggunakan **PyTorch & Python** dengan arsitektur **Non-Autoregressive (Duration-Based FastVoice)** agar:
1. **Suara Jernih & Berbicara Jelas**: Setiap huruf dan kata diucapkan secara berurutan dan teratur (bebas dari masalah *attention collapse* atau suara robot yang macet/looping).
2. **Sangat Ringan**: Ukuran model hanya **~3.4 MB**, ramah memori RAM dan CPU laptop low-spec.
3. **Fleksibel**: Dapat dilatih dan dijalankan langsung di laptop Celeron, atau dilatih super cepat di **Google Colab** secara gratis dan aman.

---

## ⚡ Panduan Melatih Model di Google Colab (Paling Cepat & Aman)

Melatih model di Google Colab sangat disarankan karena menggunakan server cloud Google (bebas dari risiko banned dan tidak membebani prosesor laptop Anda). Pelatihan 80 epoch selesai hanya dalam **~1-2 menit**!

### Langkah 1: Buka Google Colab
Klik tombol di bawah ini untuk membuka notebook pelatihan langsung di browser Anda:

👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunandar3221/MyVoice/blob/master/Train_CelerVoice_Colab.ipynb)

*(Atau buka URL: https://colab.research.google.com/github/sunandar3221/MyVoice/blob/master/Train_CelerVoice_Colab.ipynb)*

### Langkah 2: Jalankan Semua Sel Pelatihan
1. Pada menu navigasi di atas notebook Colab, klik **Runtime** lalu pilih **Run all** (atau tekan tombol `Ctrl + F9`).
2. Google Colab akan secara otomatis:
   * Mengunduh kode repositori `sunandar3221/MyVoice`.
   * Memasang library yang diperlukan (`torch`, `numpy`, `scipy`).
   * Membaca 15 rekaman sampel suara vokal manusia asli bahasa Indonesia yang ada di folder `dataset/`.
   * Melatih neural network CelerVoice v2 selama 80 epoch di cloud.

### Langkah 3: Dengarkan Hasil Suara di Colab
* Pada **Sel 4**, terdapat pemutar audio interaktif (`IPython.display.Audio`).
* Anda bisa mengetik teks kalimat bahasa Indonesia apa saja, lalu klik tombol Play untuk mendengarkan hasil suara vokal AI Anda secara langsung di browser.

### Langkah 4: Unduh dan Pasang Model di Laptop Anda
* Pada **Sel 5**, file model hasil latihan (`best_model.pt`) akan otomatis terunduh ke komputer Anda.
* Pindahkan file `best_model.pt` tersebut ke dalam folder:
  ```text
  MyVoice/checkpoints/best_model.pt
  ```
* Sekarang model AI suara Anda sudah siap dijalankan di laptop secara offline tanpa internet!

---

## 💻 Cara Menjalankan di Komputer / Laptop Lokal (Celeron)

### 1. Menggunakan Antarmuka Desktop (GUI)
Jalankan perintah ini di PowerShell atau Command Prompt:
```bash
python gui.py
```
Aplikasi grafis modern akan terbuka dengan 3 tab:
* **Tab 1 (Uji Suara / TTS)**: Ketik teks apapun lalu klik **"Sintesis & Putar Suara"** untuk mendengarkan AI berbicara melalui speaker/headset laptop.
* **Tab 2 (Rekam Suara Sendiri)**: Rekam kalimat-kalimat yang disediakan mikrofon untuk membuat dataset suara vokal Anda sendiri.
* **Tab 3 (Latih Model AI)**: Melatih model langsung di CPU laptop Anda dengan progress bar live.

---

### 2. Menggunakan Terminal (CLI)

#### A. Sintesis Suara dari Teks (Inference)
```bash
python infer.py --text "halo selamat datang di sistem kecerdasan buatan suara buatan sendiri" --output hasil.wav
```
*Audio akan otomatis disintesis dalam hitungan detik dan langsung berbunyi di speaker laptop Anda.*

#### B. Merekam Suara Anda Sendiri
Jika ingin AI menirukan karakter vokal asli Anda:
```bash
python record_voice.py
```
*Ikuti petunjuk di layar, tekan ENTER dan ucapkan kalimat yang muncul ke mikrofon Anda.*

#### C. Melatih Model di Laptop Lokal
```bash
python train.py --epochs 60 --batch_size 4
```

---

## 📁 Struktur Direktori Repositori

```text
MyVoice/
├── Train_CelerVoice_Colab.ipynb # Notebook Google Colab resmi (1-klik training cloud)
├── celer_voice/
│   ├── __init__.py
│   ├── audio.py                 # DSP Audio, Mel-filterbank, STFT, Vocoder Griffin-Lim & Stereo Playback
│   ├── text.py                  # Normalisasi teks Bahasa Indonesia & Tokenizer Karakter
│   ├── model.py                 # Arsitektur Neural Non-Autoregressive CelerVoice v2
│   ├── dataset.py               # DataLoader cerdas dengan Dynamic Padding
│   ├── trainer.py               # Mesin pelatihan CPU khusus Celeron
│   └── synthesizer.py           # Pipeline sintesis teks -> spektrogram -> audio WAV
├── dataset/
│   ├── wavs/                    # File sampel audio vokal manusia (.wav)
│   └── metadata.csv             # Pasangan nama file audio dan transkrip teks
├── checkpoints/                 # Tempat penyimpanan model terbaik (best_model.pt)
├── gui.py                       # Aplikasi Desktop GUI modern (CustomTkinter)
├── record_voice.py              # Alat perekam suara interaktif dengan mikrofon
├── prepare_sample_data.py       # Generator sampel dataset suara manusia asli
├── train.py                     # Skrip CLI untuk melatih model
├── infer.py                     # Skrip CLI untuk sintesis suara
├── requirements.txt             # Daftar dependensi Python
└── README.md
```

---

## 🔬 Spesifikasi Arsitektur Neural CelerVoice v2

| Komponen | Arsitektur | Rincian |
| :--- | :--- | :--- |
| **Model Type** | Non-Autoregressive Feed-Forward | Bebas looping / bebas attention collapse |
| **Input** | Grapheme/Character Tokenizer | 32 karakter fonetik bahasa Indonesia |
| **Text Encoder** | 1D-ResNet + Bi-GRU | 3 blok Conv1D residual (128-d) + Bi-GRU (128-d) |
| **Duration Predictor** | 2-Layer 1D Convolution | Memprediksi panjang frame tiap huruf secara presisi |
| **Length Regulator** | Monotonic Sequence Expansion | Meregangkan representasi fonem sesuai durasi |
| **Mel Decoder** | 4-Layer 1D-ResNet Blocks | Memproyeksikan representasi fonem ke spektrogram Mel 80-channel |
| **PostNet** | 3-Layer Residual Conv1D | Menajamkan forman vokal dan membuang distorsi |
| **Vocoder** | Phase Inversion Griffin-Lim | Zero latency, dioptimasi tanpa DC offset, audio jernih |
| **Ukuran Model** | **~890.000 parameter** | **~3.4 MB** (Sangat ringan untuk prosesor Intel Celeron) |
