"""
Inference CLI script for CelerVoice.
Menjalankan sintesis suara dari teks menggunakan model hasil pelatihan.
"""

import argparse
import os
from celer_voice.synthesizer import Synthesizer

def main():
    parser = argparse.ArgumentParser(description="Sintesis Suara Teks ke Audio CelerVoice")
    parser.add_argument("--text", type=str, default="halo selamat datang di sistem kecerdasan buatan",
                        help="Teks yang ingin diucapkan oleh AI")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt",
                        help="Path ke file checkpoint model (.pt)")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="File tujuan audio WAV yang dihasilkan")
    parser.add_argument("--no_play", action="store_true",
                        help="Jangan putar audio langsung ke speaker")
    parser.add_argument("--iters", type=int, default=24,
                        help="Jumlah iterasi Griffin-Lim vocoder (default: 24)")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint '{args.model}' tidak ditemukan.")
        print("Tip: Jalankan 'python train.py' terlebih dahulu untuk melatih model.")
        return

    print("=" * 60)
    print("           SINTESIS SUARA AI (CELERVOICE INFERENCE)           ")
    print(f"  Model  : {args.model}")
    print(f"  Teks   : \"{args.text}\"")
    print(f"  Output : {args.output}")
    print("=" * 60)

    synth = Synthesizer(model_path=args.model)
    synth.speak(
        text=args.text,
        output_path=args.output,
        play=(not args.no_play),
        vocoder_iters=args.iters
    )
    print("Selesai! Audio tersimpan di:", args.output)

if __name__ == "__main__":
    main()
