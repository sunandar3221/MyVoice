"""
Dataset loader and batch collator for CelerVoice.
Reads audio and text transcript pairs with dynamic batch padding.
"""

import os
import csv
import torch
from torch.utils.data import Dataset

from .text import TextTokenizer, PAD_ID
from .audio import load_audio, wav_to_mel


class VoiceDataset(Dataset):
    """
    Dataset loader for Voice pairs.
    Format of metadata.csv:
    audio_file.wav|text transcript
    """
    def __init__(self, metadata_path: str, data_dir: str = None, precompute: bool = True):
        self.items = []
        self.tokenizer = TextTokenizer()
        
        if data_dir is None:
            data_dir = os.path.dirname(metadata_path)
            
        self.data_dir = data_dir
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if len(row) < 2:
                    continue
                wav_rel = row[0].strip()
                text = row[1].strip()
                
                # Check absolute or relative path
                if os.path.isabs(wav_rel):
                    wav_path = wav_rel
                else:
                    wav_path = os.path.join(data_dir, wav_rel)
                    
                if os.path.exists(wav_path):
                    self.items.append((wav_path, text))
                    
        print(f"Loaded {len(self.items)} audio-text pairs from {metadata_path}")
        
        # Precompute mels and tokens in RAM to maximize Celeron training throughput
        self.precomputed = []
        if precompute and len(self.items) > 0:
            print("Precomputing audio features in RAM for maximum CPU speed...")
            for wav_path, text in self.items:
                tokens = torch.tensor(self.tokenizer.text_to_ids(text), dtype=torch.long)
                try:
                    wav = load_audio(wav_path)
                    mel = wav_to_mel(wav)  # [n_mels, frames]
                    self.precomputed.append((tokens, mel))
                except Exception as e:
                    print(f"Warning: Failed to load {wav_path}: {e}")
            print(f"Successfully cached {len(self.precomputed)} samples.")

    def __len__(self):
        return len(self.precomputed) if self.precomputed else len(self.items)

    def __getitem__(self, idx):
        if self.precomputed:
            return self.precomputed[idx]
            
        wav_path, text = self.items[idx]
        tokens = torch.tensor(self.tokenizer.text_to_ids(text), dtype=torch.long)
        wav = load_audio(wav_path)
        mel = wav_to_mel(wav)
        return tokens, mel


def voice_collate_fn(batch):
    """
    Collate function with dynamic padding.
    Minimizes CPU computation by padding only to the longest sequence in the batch.
    """
    # Sort batch by mel length descending for efficient RNN processing
    batch.sort(key=lambda x: x[1].size(1), reverse=True)
    
    text_lengths = torch.tensor([x[0].size(0) for x in batch], dtype=torch.long)
    mel_lengths = torch.tensor([x[1].size(1) for x in batch], dtype=torch.long)
    
    max_text_len = text_lengths.max().item()
    max_mel_len = mel_lengths.max().item()
    n_mels = batch[0][1].size(0)
    
    batch_size = len(batch)
    
    padded_texts = torch.full((batch_size, max_text_len), PAD_ID, dtype=torch.long)
    padded_mels = torch.full((batch_size, n_mels, max_mel_len), -1.0, dtype=torch.float32)
    stop_targets = torch.zeros((batch_size, max_mel_len), dtype=torch.float32)
    
    for i, (text, mel) in enumerate(batch):
        t_len = text.size(0)
        m_len = mel.size(1)
        
        padded_texts[i, :t_len] = text
        padded_mels[i, :, :m_len] = mel
        
        # Stop token is 1.0 at the end of audio (last 3 frames)
        if m_len >= 3:
            stop_targets[i, m_len - 3:m_len] = 1.0
        else:
            stop_targets[i, m_len - 1:m_len] = 1.0
            
    return {
        "text_tokens": padded_texts,
        "text_lengths": text_lengths,
        "mel_targets": padded_mels,
        "mel_lengths": mel_lengths,
        "stop_targets": stop_targets
    }
