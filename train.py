"""
Training CLI script for CelerVoice.
Menjalankan pelatihan model AI suara sendiri yang dioptimalkan untuk CPU Intel Celeron.
"""

import argparse
import os
import torch
from celer_voice.trainer import CelerTrainer

def main():
    parser = argparse.ArgumentParser(description="Latih Model AI Suara CelerVoice di CPU Celeron")
    parser.add_argument("--metadata", type=str, default="dataset/metadata.csv", 
                        help="Path ke file metadata.csv (default: dataset/metadata.csv)")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Jumlah epoch pelatihan (default: 50)")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Ukuran batch (rekomendasi celeron: 2 - 4)")
    parser.add_argument("--lr", type=float, default=1e-3, 
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--output_dir", type=str, default="checkpoints", 
                        help="Folder penyimpanan checkpoint model")
    parser.add_argument("--threads", type=int, default=2, 
                        help="Jumlah CPU threads yang digunakan (default: 2)")
    parser.add_argument("--sample_text", type=str, default="halo selamat datang",
                        help="Teks contoh untuk sintesis audio tiap checkpoint")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.metadata):
        print(f"Error: File metadata '{args.metadata}' tidak ditemukan.")
        print("Tip: Jalankan 'python prepare_sample_data.py' atau 'python record_voice.py' terlebih dahulu.")
        return

    print("=" * 60)
    print("      MEMULAI PELATIHAN MODEL AI SUARA SENDIRI (CELERVOICE)      ")
    print(f"  CPU Threads : {args.threads}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch Size  : {args.batch_size}")
    print(f"  Output Dir  : {args.output_dir}")
    print("=" * 60)

    trainer = CelerTrainer(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        num_threads=args.threads
    )
    
    trainer.train(
        epochs=args.epochs,
        save_every=10,
        sample_text=args.sample_text
    )

if __name__ == "__main__":
    main()
