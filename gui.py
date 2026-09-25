"""
CelerVoice Studio GUI
Aplikasi grafis modern berbasis CustomTkinter untuk merekam suara,
melatih model AI sendiri, dan sintesis suara (TTS) di laptop Celeron.
"""

import os
import threading
import time
import winsound
import numpy as np
import customtkinter as ctk
import sounddevice as sd

from celer_voice.audio import save_audio, SAMPLE_RATE
from celer_voice.trainer import CelerTrainer
from celer_voice.synthesizer import Synthesizer
from record_voice import PROMPT_SENTENCES, trim_silence

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class CelerVoiceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CelerVoice - Studio Model AI Suara Sendiri")
        self.geometry("780x620")
        self.minsize(700, 550)

        # Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=15)

        self.tab_tts = self.tabview.add("1. Uji Suara (TTS)")
        self.tab_record = self.tabview.add("2. Rekam Suara Sendiri")
        self.tab_train = self.tabview.add("3. Latih Model AI")

        self.setup_tts_tab()
        self.setup_record_tab()
        self.setup_train_tab()

    # =========================================================================
    # TAB 1: TEXT-TO-SPEECH (UJI SUARA AI)
    # =========================================================================
    def setup_tts_tab(self):
        title = ctk.CTkLabel(self.tab_tts, text="Sintesis Teks ke Suara (CelerVoice TTS)", 
                             font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=10)

        subtitle = ctk.CTkLabel(self.tab_tts, text="Model AI mandiri yang berjalan sangat cepat dan hemat daya di CPU Celeron.",
                                text_color="gray")
        subtitle.pack(pady=(0, 15))

        lbl_text = ctk.CTkLabel(self.tab_tts, text="Ketik teks yang ingin diucapkan:")
        lbl_text.pack(anchor="w", padx=20)

        self.txt_input = ctk.CTkTextbox(self.tab_tts, height=120)
        self.txt_input.pack(fill="x", padx=20, pady=(5, 15))
        self.txt_input.insert("1.0", "halo selamat datang di sistem kecerdasan buatan suara buatan sendiri")

        btn_frame = ctk.CTkFrame(self.tab_tts, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=10)

        self.btn_speak = ctk.CTkButton(btn_frame, text="🔊 Sintesis & Putar Suara", command=self.on_speak_click,
                                      height=40, font=ctk.CTkFont(size=14, weight="bold"))
        self.btn_speak.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.lbl_tts_status = ctk.CTkLabel(self.tab_tts, text="Siap menghasilkan suara.", text_color="#3B8ED0")
        self.lbl_tts_status.pack(pady=15)

    def on_speak_click(self):
        text = self.txt_input.get("1.0", "end").strip()
        if not text:
            self.lbl_tts_status.configure(text="Teks tidak boleh kosong!", text_color="red")
            return

        model_path = "checkpoints/best_model.pt"
        if not os.path.exists(model_path):
            self.lbl_tts_status.configure(text="Model belum dilatih! Silakan ke Tab 'Latih Model AI' terlebih dahulu.", text_color="red")
            return

        self.btn_speak.configure(state="disabled", text="Sedang memproses audio...")
        self.lbl_tts_status.configure(text="Menjalankan neural speech decoder...", text_color="#3B8ED0")

        def run_inference():
            try:
                synth = Synthesizer(model_path=model_path)
                out_path = "output.wav"
                wav, info = synth.synthesize(text)
                save_audio(out_path, wav)
                self.after(0, lambda: self.lbl_tts_status.configure(
                    text=f"Berhasil! Durasi: {info['audio_duration']:.2f}s | Waktu: {info['total_time']:.2f}s (Kecepatan: {info['rtf']:.2f}x)",
                    text_color="#2ECC71"
                ))
                play_audio(wav, sr=SAMPLE_RATE)
            except Exception as e:
                self.after(0, lambda err=e: self.lbl_tts_status.configure(text=f"Error: {err}", text_color="red"))
            finally:
                self.after(0, lambda: self.btn_speak.configure(state="normal", text="🔊 Sintesis & Putar Suara"))

        threading.Thread(target=run_inference, daemon=True).start()

    # =========================================================================
    # TAB 2: REKAM SUARA SENDIRI
    # =========================================================================
    def setup_record_tab(self):
        title = ctk.CTkLabel(self.tab_record, text="Perekam Suara Pengguna (Dataset Generator)",
                             font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=10)

        subtitle = ctk.CTkLabel(self.tab_record, text="Rekam beberapa kalimat untuk mengajarkan AI karakter vokal Anda sendiri.",
                                text_color="gray")
        subtitle.pack(pady=(0, 15))

        self.current_prompt_idx = 0

        self.card_frame = ctk.CTkFrame(self.tab_record)
        self.card_frame.pack(fill="x", padx=20, pady=10)

        self.lbl_sentence_num = ctk.CTkLabel(self.card_frame, text=f"Kalimat 1/{len(PROMPT_SENTENCES)}",
                                             font=ctk.CTkFont(size=12, weight="bold"), text_color="gray")
        self.lbl_sentence_num.pack(anchor="w", padx=15, pady=(10, 5))

        self.lbl_prompt_text = ctk.CTkLabel(self.card_frame, text=PROMPT_SENTENCES[0],
                                            font=ctk.CTkFont(size=16, weight="bold"), wraplength=600)
        self.lbl_prompt_text.pack(fill="x", padx=15, pady=(5, 15))

        ctrl_frame = ctk.CTkFrame(self.tab_record, fg_color="transparent")
        ctrl_frame.pack(fill="x", padx=20, pady=10)

        self.btn_record = ctk.CTkButton(ctrl_frame, text="🎙️ Rekam (5 Detik)", command=self.on_record_click,
                                        fg_color="#E74C3C", hover_color="#C0392B", height=40, font=ctk.CTkFont(size=14, weight="bold"))
        self.btn_record.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_play_rec = ctk.CTkButton(ctrl_frame, text="▶ Putar Ulang", command=self.on_play_recorded,
                                          state="disabled", height=40)
        self.btn_play_rec.pack(side="left", fill="x", expand=True, padx=5)

        self.btn_next = ctk.CTkButton(ctrl_frame, text="Kalimat Berikutnya ➡", command=self.on_next_prompt, height=40)
        self.btn_next.pack(side="left", fill="x", expand=True, padx=(5, 0))

        self.lbl_rec_status = ctk.CTkLabel(self.tab_record, text="Tekan 'Rekam' lalu ucapkan kalimat di atas dengan jelas.",
                                           text_color="#3B8ED0")
        self.lbl_rec_status.pack(pady=15)

    def on_record_click(self):
        self.btn_record.configure(state="disabled", text="Sedang Merekam... (Bicara!)")
        self.lbl_rec_status.configure(text="Merekam 5 detik...", text_color="#E74C3C")

        def record_thread():
            duration = 5
            audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            raw_audio = audio.flatten()
            trimmed = trim_silence(raw_audio)

            if np.max(np.abs(trimmed)) < 0.05:
                self.after(0, lambda: self.lbl_rec_status.configure(text="Suara terlalu pelan! Coba ulangi lagi.", text_color="orange"))
                self.after(0, lambda: self.btn_record.configure(state="normal", text="🎙️ Rekam (5 Detik)"))
                return

            wavs_dir = os.path.join("dataset", "wavs")
            os.makedirs(wavs_dir, exist_ok=True)
            filename = f"user_{self.current_prompt_idx+1:03d}.wav"
            abs_path = os.path.join(wavs_dir, filename)
            rel_path = os.path.join("wavs", filename)

            save_audio(abs_path, trimmed, sr=SAMPLE_RATE)
            self.last_recorded_path = abs_path

            # Update metadata
            meta_path = os.path.join("dataset", "metadata.csv")
            with open(meta_path, 'a', newline='', encoding='utf-8') as f:
                f.write(f"{rel_path}|{PROMPT_SENTENCES[self.current_prompt_idx]}\n")

            self.after(0, lambda: self.lbl_rec_status.configure(text=f"Tersimpan: {filename}!", text_color="#2ECC71"))
            self.after(0, lambda: self.btn_record.configure(state="normal", text="🎙️ Rekam Lagi"))
            self.after(0, lambda: self.btn_play_rec.configure(state="normal"))

        threading.Thread(target=record_thread, daemon=True).start()

    def on_play_recorded(self):
        if hasattr(self, 'last_recorded_path') and os.path.exists(self.last_recorded_path):
            play_audio(self.last_recorded_path, sr=SAMPLE_RATE)

    def on_next_prompt(self):
        self.current_prompt_idx = (self.current_prompt_idx + 1) % len(PROMPT_SENTENCES)
        self.lbl_sentence_num.configure(text=f"Kalimat {self.current_prompt_idx+1}/{len(PROMPT_SENTENCES)}")
        self.lbl_prompt_text.configure(text=PROMPT_SENTENCES[self.current_prompt_idx])
        self.btn_play_rec.configure(state="disabled")
        self.lbl_rec_status.configure(text="Siap merekam kalimat baru.", text_color="#3B8ED0")

    # =========================================================================
    # TAB 3: LATIH MODEL AI (TRAINING STUDIO)
    # =========================================================================
    def setup_train_tab(self):
        title = ctk.CTkLabel(self.tab_train, text="Studio Pelatihan Model AI CelerVoice",
                             font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=10)

        desc = ctk.CTkLabel(self.tab_train, 
                             text="Melatih model neural dari nol di CPU Celeron tanpa memerlukan kartu grafis (GPU).",
                             text_color="gray")
        desc.pack(pady=(0, 15))

        settings_frame = ctk.CTkFrame(self.tab_train)
        settings_frame.pack(fill="x", padx=20, pady=10)

        lbl_epochs = ctk.CTkLabel(settings_frame, text="Jumlah Epoch Pelatihan:")
        lbl_epochs.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        self.slider_epochs = ctk.CTkSlider(settings_frame, from_=5, to=100, number_of_steps=19,
                                           command=self.on_epoch_slider)
        self.slider_epochs.set(30)
        self.slider_epochs.grid(row=0, column=1, padx=15, pady=10, sticky="ew")

        self.lbl_epoch_val = ctk.CTkLabel(settings_frame, text="30 Epochs", font=ctk.CTkFont(weight="bold"))
        self.lbl_epoch_val.grid(row=0, column=2, padx=15, pady=10)

        settings_frame.columnconfigure(1, weight=1)

        self.btn_start_train = ctk.CTkButton(self.tab_train, text="🚀 Mulai Pelatihan Model",
                                             command=self.on_start_train, height=42, 
                                             fg_color="#2ECC71", hover_color="#27AE60",
                                             font=ctk.CTkFont(size=14, weight="bold"))
        self.btn_start_train.pack(fill="x", padx=20, pady=15)

        self.progress_bar = ctk.CTkProgressBar(self.tab_train)
        self.progress_bar.pack(fill="x", padx=20, pady=5)
        self.progress_bar.set(0)

        self.lbl_train_status = ctk.CTkLabel(self.tab_train, text="Tekan 'Mulai Pelatihan Model' untuk memulai.",
                                             text_color="#3B8ED0")
        self.lbl_train_status.pack(pady=10)

    def on_epoch_slider(self, val):
        self.lbl_epoch_val.configure(text=f"{int(val)} Epochs")

    def on_start_train(self):
        epochs = int(self.slider_epochs.get())
        meta_path = "dataset/metadata.csv"

        if not os.path.exists(meta_path):
            self.lbl_train_status.configure(text="Dataset belum ada! Silakan rekam atau buat sample dataset terlebih dahulu.", text_color="red")
            return

        self.btn_start_train.configure(state="disabled", text="Sedang Melatih Model AI...")
        self.progress_bar.set(0)

        def train_worker():
            try:
                trainer = CelerTrainer(metadata_path=meta_path, batch_size=4, num_threads=2)
                for epoch in range(1, epochs + 1):
                    stats = trainer.train_epoch(epoch)
                    progress = epoch / epochs
                    self.after(0, lambda p=progress, e=epoch, s=stats: self.update_train_progress(p, e, epochs, s))
                self.after(0, lambda: self.lbl_train_status.configure(
                    text="Pelatihan Selesai! Model terbaik tersimpan di checkpoints/best_model.pt",
                    text_color="#2ECC71"
                ))
            except Exception as e:
                self.after(0, lambda err=e: self.lbl_train_status.configure(text=f"Error: {err}", text_color="red"))
            finally:
                self.after(0, lambda: self.btn_start_train.configure(state="normal", text="🚀 Mulai Pelatihan Model"))

        threading.Thread(target=train_worker, daemon=True).start()

    def update_train_progress(self, progress, epoch, total, stats):
        self.progress_bar.set(progress)
        self.lbl_train_status.configure(
            text=f"Epoch [{epoch}/{total}] - Loss: {stats['loss']:.4f} (Mel: {stats['mel_loss']:.4f}) | Waktu: {stats['time']:.2f}s",
            text_color="#3B8ED0"
        )


def main():
    app = CelerVoiceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
