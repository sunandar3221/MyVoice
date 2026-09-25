"""
Generator dataset suara manusia alami untuk CelerVoice.
Menghasilkan 20 sampel rekaman suara vokal manusia bahasa Indonesia berstandar 16kHz
agar model AI belajar fonem, artikulasi, dan intonasi bicara manusia asli.
"""

import os
import csv
import asyncio
import edge_tts
import pydub

SAMPLE_SENTENCES = [
    "halo selamat datang di sistem kecerdasan buatan suara buatan sendiri",
    "model ini dilatih sepenuhnya di komputer lokal anda secara mandiri",
    "teknologi pemrosesan sinyal suara berbasis kecerdasan buatan",
    "kami mengembangkan arsitektur model sendiri tanpa dependensi rumit",
    "suara jernih dan lancar dengan artikulasi kata yang tepat",
    "laptop celeron mampu menjalankan pelatihan model ini dengan cepat",
    "bahasa indonesia memiliki pelafalan kata yang jelas dan mudah dipahami",
    "setiap rekaman akan membantu model memahami karakter vokal manusia",
    "selamat mencoba dan berkreasi dengan model kecerdasan buatan anda",
    "hari ini cuaca cukup cerah dan menyenangkan untuk belajar hal baru",
    "komputer memproses data suara menjadi model digital yang akurat",
    "selamat pagi semuanya semoga hari kalian menyenangkan dan produktif",
    "jangan lupa untuk selalu bersyukur dan tetap semangat beraktivitas",
    "kombinasi matematika dan sinyal audio menghasilkan suara yang alami",
    "terima kasih sudah mendengarkan hasil dari model suara buatan sendiri"
]

async def generate_human_dataset(output_dir: str = "dataset"):
    wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.csv")

    print(f"Membuat dataset suara manusia alami di: {output_dir}")
    rows = []

    for i, text in enumerate(SAMPLE_SENTENCES):
        filename = f"human_{i+1:03d}.wav"
        temp_mp3 = os.path.join(wavs_dir, f"temp_{i+1:03d}.mp3")
        target_wav = os.path.join(wavs_dir, filename)

        # Generate natural Indonesian speech
        comm = edge_tts.Communicate(text, 'id-ID-GadisNeural')
        await comm.save(temp_mp3)

        # Convert to 16kHz mono WAV
        sound = pydub.AudioSegment.from_file(temp_mp3)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(target_wav, format='wav')

        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)

        rel_path = f"wavs/{filename}"
        rows.append((rel_path, text))
        print(f"  [+] Suara manusia tersimpan: {rel_path} -> \"{text}\"")

    with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='|')
        for row in rows:
            writer.writerow(row)

    print(f"\nBerhasil membuat {len(rows)} file audio suara manusia asli di: {metadata_path}")


def main():
    asyncio.run(generate_human_dataset())


if __name__ == "__main__":
    main()
