"""
CPU-Optimized Trainer for CelerVoice on Intel Celeron and Low-Power Laptops.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import CelerVoiceTTS, guided_attention_loss
from .dataset import VoiceDataset, voice_collate_fn
from .audio import mel_to_wav, save_audio


class CelerTrainer:
    """
    Trainer engineered specifically for fast training on 2-4 core Celeron CPU.
    """
    def __init__(self, 
                 metadata_path: str,
                 output_dir: str = "checkpoints",
                 batch_size: int = 4,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-6,
                 guided_attn_weight: float = 1.0,
                 num_threads: int = 2):
        
        # Configure CPU threads for Intel Celeron
        torch.set_num_threads(num_threads)
        self.device = torch.device("cpu")
        print(f"CelerVoice Trainer initialized on CPU with {num_threads} threads.")
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.guided_attn_weight = guided_attn_weight
        self.dataset = VoiceDataset(metadata_path)
        
        if len(self.dataset) == 0:
            raise ValueError(f"No audio files found from {metadata_path}")
            
        self.batch_size = min(batch_size, len(self.dataset))
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=voice_collate_fn,
            num_workers=0
        )
        
        self.model = CelerVoiceTTS().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)
        
        self.criterion_mel = nn.L1Loss()
        self.criterion_stop = nn.BCELoss()
        self.best_loss = float("inf")

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_mel_loss = 0.0
        total_stop_loss = 0.0
        total_attn_loss = 0.0
        total_loss = 0.0
        
        t0 = time.time()
        for batch in self.loader:
            text_tokens = batch["text_tokens"].to(self.device)
            text_lengths = batch["text_lengths"].to(self.device)
            mel_targets = batch["mel_targets"].to(self.device)
            mel_lengths = batch["mel_lengths"].to(self.device)
            stop_targets = batch["stop_targets"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward with teacher forcing
            outputs = self.model(text_tokens, mel_targets=mel_targets, teacher_forcing_ratio=1.0)
            mel_init = outputs["mel_init"]
            mel_final = outputs["mel_final"]
            stop_preds = outputs["stop_preds"]
            alignments = outputs["alignments"]
            
            # Loss calculations
            loss_init = self.criterion_mel(mel_init, mel_targets)
            loss_final = self.criterion_mel(mel_final, mel_targets)
            loss_mel = loss_init + loss_final
            
            loss_stop = self.criterion_stop(stop_preds, stop_targets)
            loss_attn = guided_attention_loss(alignments, text_lengths, mel_lengths)
            
            batch_loss = loss_mel + 0.5 * loss_stop + self.guided_attn_weight * loss_attn
            batch_loss.backward()
            
            # Gradient clipping for stable training
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_mel_loss += loss_mel.item()
            total_stop_loss += loss_stop.item()
            total_attn_loss += loss_attn.item()
            total_loss += batch_loss.item()
            
        self.scheduler.step()
        elapsed = time.time() - t0
        num_batches = len(self.loader)
        
        return {
            "epoch": epoch,
            "loss": total_loss / num_batches,
            "mel_loss": total_mel_loss / num_batches,
            "stop_loss": total_stop_loss / num_batches,
            "attn_loss": total_attn_loss / num_batches,
            "time": elapsed,
            "lr": self.optimizer.param_groups[0]["lr"]
        }

    def train(self, epochs: int = 100, save_every: int = 10, sample_text: str = None):
        print("=" * 60)
        print(f"Memulai Training CelerVoice: {epochs} Epochs, Batch Size: {self.batch_size}")
        print("=" * 60)
        
        for epoch in range(1, epochs + 1):
            stats = self.train_epoch(epoch)
            
            print(f"[Epoch {epoch:3d}/{epochs}] "
                  f"Loss: {stats['loss']:.4f} "
                  f"(Mel: {stats['mel_loss']:.4f}, Attn: {stats['attn_loss']:.4f}, Stop: {stats['stop_loss']:.4f}) | "
                  f"Waktu: {stats['time']:.2f}s | "
                  f"LR: {stats['lr']:.6f}")
            
            # Save best model
            if stats["loss"] < self.best_loss:
                self.best_loss = stats["loss"]
                best_path = os.path.join(self.output_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": self.best_loss
                }, best_path)
                
            # Periodic checkpoint and sample synthesis
            if epoch % save_every == 0 or epoch == epochs:
                ckpt_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save(self.model.state_dict(), ckpt_path)
                
                if sample_text:
                    self.generate_sample_audio(sample_text, epoch)
                    
        print("=" * 60)
        print(f"Training Selesai! Model terbaik disimpan di: {os.path.join(self.output_dir, 'best_model.pt')}")
        print("=" * 60)

    def generate_sample_audio(self, text: str, epoch: int):
        """Generate sample audio during training to monitor voice clarity."""
        self.model.eval()
        from .text import TextTokenizer
        tokenizer = TextTokenizer()
        tokens = torch.tensor([tokenizer.text_to_ids(text)], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            out = self.model(tokens, mel_targets=None, max_decoder_steps=350)
            mel = out["mel_final"].squeeze(0)
            wav = mel_to_wav(mel, n_iter=16)
            
        sample_path = os.path.join(self.output_dir, f"sample_epoch_{epoch}.wav")
        save_audio(sample_path, wav)
        print(f"   -> Sample audio tersimpan: {sample_path}")
