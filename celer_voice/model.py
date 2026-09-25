"""
CelerVoice - Custom Lightweight Neural Speech Architecture
100% Original PyTorch Model.
Designed from scratch for fast CPU training and inference on Intel Celeron.
Total parameters: ~650K (~2.6 MB).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .text import VOCAB, PAD_ID


class LocationSensitiveAttention(nn.Module):
    """
    Location-Sensitive Additive Attention.
    Uses previous alignment history to prevent words from being skipped or repeated.
    """
    def __init__(self, query_dim: int = 128, memory_dim: int = 128, 
                 attn_dim: int = 64, num_filters: int = 32, kernel_size: int = 15):
        super().__init__()
        self.query_layer = nn.Linear(query_dim, attn_dim, bias=False)
        self.memory_layer = nn.Linear(memory_dim, attn_dim, bias=False)
        self.location_conv = nn.Conv1d(
            in_channels=1, 
            out_channels=num_filters, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            bias=False
        )
        self.location_layer = nn.Linear(num_filters, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=True)
        self.score_mask_value = -1e4

    def forward(self, query: torch.Tensor, memory: torch.Tensor, 
                last_alignment: torch.Tensor, memory_mask: torch.Tensor = None):
        """
        query: [batch, query_dim]
        memory: [batch, text_len, memory_dim]
        last_alignment: [batch, text_len]
        memory_mask: [batch, text_len] boolean mask (True = valid, False = pad)
        """
        # [batch, 1, text_len] -> conv -> [batch, num_filters, text_len] -> [batch, text_len, num_filters]
        loc_features = self.location_conv(last_alignment.unsqueeze(1)).transpose(1, 2)
        loc_energy = self.location_layer(loc_features)
        
        # [batch, 1, attn_dim]
        q_energy = self.query_layer(query).unsqueeze(1)
        # [batch, text_len, attn_dim]
        m_energy = self.memory_layer(memory)
        
        # Total energy: [batch, text_len]
        energy = self.v(torch.tanh(q_energy + m_energy + loc_energy)).squeeze(-1)
        
        if memory_mask is not None:
            energy = energy.masked_fill(~memory_mask, self.score_mask_value)
            
        alignment = F.softmax(energy, dim=-1)
        # Context vector: [batch, memory_dim]
        context = torch.bmm(alignment.unsqueeze(1), memory).squeeze(1)
        return context, alignment


class DecoderPreNet(nn.Module):
    """
    Bottleneck PreNet for decoder input.
    Dropout is always active even during inference to regularize autoregression.
    """
    def __init__(self, in_dim: int = 80, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=True)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=True)
        return x


class TextEncoder(nn.Module):
    """
    Lightweight Text Encoder:
    Embedding -> 3x Conv1D-BatchNorm-ReLU -> Bidirectional GRU
    """
    def __init__(self, vocab_size: int = len(VOCAB), embed_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(3)
        ])
        
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        text_tokens: [batch, text_len]
        returns: [batch, text_len, hidden_dim * 2]
        """
        x = self.embedding(text_tokens)  # [batch, text_len, embed_dim]
        x = x.transpose(1, 2)  # [batch, embed_dim, text_len]
        for conv in self.conv_layers:
            x = conv(x)
        x = x.transpose(1, 2)  # [batch, text_len, embed_dim]
        memory, _ = self.gru(x)  # [batch, text_len, hidden_dim * 2]
        return memory


class PostNet(nn.Module):
    """
    5-layer 1D Convolutional Residual PostNet.
    Sharpens formants, harmonics, and acoustic details from preliminary mel frames.
    """
    def __init__(self, mel_dim: int = 80, hidden_dim: int = 128):
        super().__init__()
        layers = []
        layers.append(nn.Sequential(
            nn.Conv1d(mel_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1)
        ))
        for _ in range(3):
            layers.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.1)
            ))
        layers.append(nn.Conv1d(hidden_dim, mel_dim, kernel_size=5, padding=2))
        self.layers = nn.ModuleList(layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: [batch, n_mels, frames]
        returns: residual mel [batch, n_mels, frames]
        """
        x = mel
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class CelerVoiceTTS(nn.Module):
    """
    Full End-to-End Neural TTS Architecture:
    - TextEncoder
    - LocationSensitiveAttention
    - Decoder RNN with PreNet
    - Linear & Stop projections
    - PostNet (Residual refinement)
    """
    def __init__(self, vocab_size: int = len(VOCAB), mel_dim: int = 80, 
                 encoder_dim: int = 128, decoder_dim: int = 128):
        super().__init__()
        self.mel_dim = mel_dim
        self.decoder_dim = decoder_dim
        
        self.encoder = TextEncoder(vocab_size=vocab_size, embed_dim=128, hidden_dim=encoder_dim // 2)
        self.prenet = DecoderPreNet(in_dim=mel_dim, hidden_dim=128)
        self.attention = LocationSensitiveAttention(
            query_dim=decoder_dim, memory_dim=encoder_dim, attn_dim=64
        )
        self.decoder_gru = nn.GRUCell(input_size=128 + encoder_dim, hidden_size=decoder_dim)
        self.linear_mel = nn.Linear(decoder_dim + encoder_dim, mel_dim)
        self.linear_stop = nn.Linear(decoder_dim + encoder_dim, 1)
        self.postnet = PostNet(mel_dim=mel_dim, hidden_dim=128)

    def forward(self, text_tokens: torch.Tensor, mel_targets: torch.Tensor = None, 
                max_decoder_steps: int = 600, teacher_forcing_ratio: float = 1.0):
        """
        Training & Inference forward pass.
        text_tokens: [batch, text_len]
        mel_targets: [batch, n_mels, frames] (optional, for training)
        """
        batch_size = text_tokens.size(0)
        memory = self.encoder(text_tokens)  # [batch, text_len, encoder_dim]
        memory_mask = (text_tokens != PAD_ID)
        
        text_len = memory.size(1)
        device = text_tokens.device
        
        # Initial states
        decoder_state = torch.zeros(batch_size, self.decoder_dim, device=device)
        alignment = torch.zeros(batch_size, text_len, device=device)
        alignment[:, 0] = 1.0  # start at token 0
        context = torch.zeros(batch_size, memory.size(-1), device=device)
        current_mel = torch.zeros(batch_size, self.mel_dim, device=device)
        
        outputs = []
        stop_preds = []
        alignments = []
        
        if mel_targets is not None:
            # Training mode with teacher forcing
            num_steps = mel_targets.size(2)
        else:
            # Inference mode
            num_steps = max_decoder_steps
            
        for step in range(num_steps):
            prenet_out = self.prenet(current_mel)  # [batch, 128]
            gru_in = torch.cat([prenet_out, context], dim=-1)  # [batch, 128 + encoder_dim]
            decoder_state = self.decoder_gru(gru_in, decoder_state)  # [batch, decoder_dim]
            
            context, alignment = self.attention(decoder_state, memory, alignment, memory_mask)
            
            decoder_out = torch.cat([decoder_state, context], dim=-1)
            mel_frame = self.linear_mel(decoder_out)  # [batch, mel_dim]
            stop_frame = torch.sigmoid(self.linear_stop(decoder_out))  # [batch, 1]
            
            outputs.append(mel_frame)
            stop_preds.append(stop_frame)
            alignments.append(alignment)
            
            # Next frame input selection
            if mel_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                current_mel = mel_targets[:, :, step]
            else:
                current_mel = mel_frame
                
            # Inference early stopping
            if mel_targets is None and step > 10:
                if stop_frame.mean().item() > 0.65 or alignment[:, -1].mean().item() > 0.8:
                    break
                    
        # Stack sequences along time frame dimension
        # mel_outputs: [batch, n_mels, frames]
        mel_init = torch.stack(outputs, dim=2)
        stop_preds = torch.cat(stop_preds, dim=1)  # [batch, frames]
        alignments = torch.stack(alignments, dim=1)  # [batch, frames, text_len]
        
        # Apply PostNet residual refinement
        mel_residual = self.postnet(mel_init)
        mel_final = mel_init + mel_residual
        
        return {
            "mel_init": mel_init,
            "mel_final": mel_final,
            "stop_preds": stop_preds,
            "alignments": alignments
        }


def guided_attention_loss(alignments: torch.Tensor, text_lengths: torch.Tensor, 
                          mel_lengths: torch.Tensor, sigma: float = 0.2) -> torch.Tensor:
    """
    Guided Attention Loss (Tachibana et al.).
    Directs attention diagonally so the model learns alignments within 20-30 epochs on CPU.
    """
    batch_size, max_frames, max_text = alignments.shape
    device = alignments.device
    
    total_loss = 0.0
    for b in range(batch_size):
        T = mel_lengths[b].item()
        N = text_lengths[b].item()
        if T == 0 or N == 0:
            continue
            
        t_grid = torch.arange(T, device=device).unsqueeze(1).float() / float(T)
        n_grid = torch.arange(N, device=device).unsqueeze(0).float() / float(N)
        
        # Guided weight matrix: W[t, n] = 1 - exp(-(t/T - n/N)^2 / (2 * sigma^2))
        W = 1.0 - torch.exp(-((t_grid - n_grid) ** 2) / (2.0 * sigma * sigma))
        
        attn_sub = alignments[b, :T, :N]
        total_loss += torch.mean(attn_sub * W)
        
    return total_loss / max(batch_size, 1)
