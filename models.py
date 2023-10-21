import pandas as pd
import os, gc
import numpy as np
from sklearn.model_selection import KFold
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
from fastai.vision.all import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


##### Base Model

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RelativeSinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x, seq_len):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))

        # Normalize the positions by the sequence length
        x = x / seq_len

        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RNA_Model(nn.Module):
    def __init__(self, dim=192, depth=12, head_size=32, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(4,dim)
        self.pos_enc = SinusoidalPosEmb(dim)
        #self.pos_enc = RelativeSinusoidalPosEmb(dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
                dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True), depth)
        self.proj_out = nn.Linear(dim,2)

    def forward(self, x0):
        mask = x0['mask']
        Lmax = mask.sum(-1).max()
        mask = mask[:,:Lmax]
        x = x0['seq'][:,:Lmax]

        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos)
        x = self.emb(x)
        x = x + pos

        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.proj_out(x)

        return x
    





### KMER Mixture model

class KMER_Model(nn.Module):
    def __init__(self, kmer_size=88, num_kmers=65, dim=192, depth=6, head_size=40, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(4, dim)
        self.kmer_emb = nn.Embedding(num_kmers, kmer_size)  # New k-mer embedding layer
        self.pos_enc = SinusoidalPosEmb(dim + kmer_size)  # Adjusted for concatenated embeddings
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim + kmer_size,  # Adjusted for concatenated embeddings
                nhead=(dim + kmer_size) // head_size,
                dim_feedforward=4 * (dim + kmer_size),  # Adjusted for concatenated embeddings
                dropout=0.1,
                activation=nn.GELU(),
                batch_first=True,
                norm_first=True),
            depth)
        self.proj_out = nn.Linear(dim + kmer_size, 2)  # Adjusted for concatenated embeddings

    def forward(self, x):
        seq_mask = x['mask']
        Lmax = seq_mask.sum(-1).max()
        seq_mask = seq_mask[:, :Lmax]
        seq = x['seq'][:, :Lmax]
        kmers = x['kmers'][:, :Lmax]  # Extract kmers from the input dictionary

        pos = torch.arange(Lmax, device=seq.device).unsqueeze(0)
        pos = self.pos_enc(pos)
        seq_emb = self.emb(seq)
        kmer_emb = self.kmer_emb(kmers)  # Embed kmers
        #x = seq_emb + kmer_emb
        x = torch.cat((seq_emb, kmer_emb), dim=-1)  # Concatenate sequence and kmer embeddings
        x = x + pos

        x = self.transformer(x, src_key_padding_mask=~seq_mask)
        x = self.proj_out(x)

        return x








### FNET

class FFT_SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(-1).float()
        div_term = torch.exp(torch.arange(0., self.dim, 2., device=x.device).float() * -(math.log(10000.0) / self.dim))
        pos_enc = torch.zeros(x.size(1), self.dim, device=x.device).float()
        pos_enc[:, 0::2] = torch.sin(pos * div_term.unsqueeze(0))
        pos_enc[:, 1::2] = torch.cos(pos * div_term.unsqueeze(0))
        return pos_enc.unsqueeze(0)

class FNetBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 1*dim),
            nn.GELU(),
            nn.Linear(1*dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        #x = x + self.ff(self.norm1(x))
        x_fft = torch.fft.fft(torch.fft.fftshift(self.norm2(x)))
        x = x + torch.fft.ifftshift(torch.fft.ifft(x_fft)).real
        x = x + self.ff(self.norm1(x))
        return x

class RNA_FNet_Model(nn.Module):
    def __init__(self, dim=388, depth=12, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(4, dim)
        self.pos_enc = SinusoidalPosEmb(dim)
        self.layers = nn.ModuleList([FNetBlock(dim) for _ in range(depth)])
        self.proj_out = nn.Linear(dim, 2)

    def forward(self, x0):
        mask = x0['mask']
        Lmax = mask.sum(-1).max()
        mask = mask[:, :Lmax]
        x = x0['seq'][:, :Lmax]

        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos).to(x.device)  # Ensure pos is on the same device as x
        x = self.emb(x)
        x = x + pos

        for layer in self.layers:
            x = layer(x)

        x = self.proj_out(x)

        return x








### RNAFORMER
### RNAFORMER
CFG = {
    'block_size': 206,

    'rna_model_dim': 192, # 300,
    'rna_model_num_heads': 6,# 6,   32:1
    'rna_model_num_encoder_layers': 12,# 5,  32:1
    'rna_model_num_lstm_layers': 0,
    'rna_model_lstm_dropout': 0,
    'rna_model_first_dropout': 0.1,
    'rna_model_encoder_dropout': 0.1,
    'rna_model_mha_dropout': 0,
    'rna_model_ffn_multiplier': 4,

}

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()

        self.mha = nn.MultiheadAttention(CFG['rna_model_dim'], CFG['rna_model_num_heads'])

        self.layer_norm1 = nn.LayerNorm(CFG['rna_model_dim'])
        self.layer_norm2 = nn.LayerNorm(CFG['rna_model_dim'])

        # Splitting the original sequential block into individual named layers
        self.sequential = nn.Sequential(
            nn.Linear(CFG['rna_model_dim'], CFG['rna_model_dim']*CFG['rna_model_ffn_multiplier']),
            nn.GELU(),
            nn.Linear(CFG['rna_model_dim']*CFG['rna_model_ffn_multiplier'], CFG['rna_model_dim']),
            nn.Dropout(CFG['rna_model_first_dropout'])
        )
        #self.linear3 = nn.Linear(CFG['rna_model_dim'], CFG['rna_model_dim'])
        #self.dropout3 = nn.Dropout(CFG['rna_model_encoder_dropout'])
        self.activation = nn.GELU()

        self.mha_dropout = nn.Dropout(CFG['rna_model_mha_dropout'])
        self.attention_weights = None

    def forward(self, x, padding_mask=None, primer_mask=None):
        if padding_mask is not None and primer_mask is not None:
            combined_mask = padding_mask | primer_mask
        elif padding_mask is not None:
            combined_mask = padding_mask
        elif primer_mask is not None:
            combined_mask = primer_mask
        else:
            combined_mask = None

        # Using a similar approach to the provided TransformerEncoderLayer code
        # Adding a block for self-attention followed by layer normalization
        x_norm = self.layer_norm1(x)
        attn_output, att_weights = self.mha(x_norm, x_norm, x_norm, key_padding_mask=combined_mask)
        #attn_output, att_weights = self.mha(self.layer_norm1(x), self.layer_norm1(x), x, key_padding_mask=combined_mask)
        self.attention_weights = att_weights
        x = x + self.mha_dropout(attn_output)


        """# Adding a block for the feedforward network followed by layer normalization
        # Breaking down the original sequential block to individual steps
        ff_output = self.linear1(x)
        ff_output = self.activation(ff_output)
        ff_output = self.dropout1(ff_output)
        ff_output = self.linear2(ff_output)
        #ff_output = self.activation(ff_output)
        ff_output = self.dropout2(ff_output)
        #ff_output = self.linear3(ff_output)
        #ff_output = self.dropout3(ff_output)"""

        x = x + self.sequential(self.layer_norm2(x))

        return x


class RNAEncoder(nn.Module):
    def __init__(self):
        super(RNAEncoder, self).__init__()

        self.pos_enc = SinusoidalPosEmb(CFG['rna_model_dim'])   # Retained Sinusoidal Positional Embedding
        #self.pos_enc = RelativeSinusoidalPosEmb(CFG['rna_model_dim'])

        self.first_dropout = nn.Dropout(CFG['rna_model_first_dropout'])
        self.enc_layers = nn.ModuleList([EncoderLayer() for _ in range(CFG['rna_model_num_encoder_layers'])])

        self.lstm_layers = nn.ModuleList()
        for i in range(CFG['rna_model_num_lstm_layers']):
            input_dim = CFG['rna_model_dim'] * (2 if i > 0 else 1)  # Adjust the input dimensions for subsequent LSTMs
            self.lstm_layers.append(nn.LSTM(input_dim, CFG['rna_model_dim'],
                                            bidirectional=True, batch_first=True, dropout=CFG['rna_model_lstm_dropout'] if i < CFG['rna_model_num_lstm_layers'] - 1 else 0))


    def forward(self, x, mask):
        Lmax = mask.sum(-1).max()
        clipped_mask = mask[:, :Lmax]
        b_size, seq_len = clipped_mask.shape

        # Create a mask with False values
        primer_mask = torch.zeros((b_size, seq_len), dtype=torch.bool, device=clipped_mask.device)
        # Set the first 26 and last 21 positions to True to mask them
        primer_mask[:, :26] = True
        primer_mask[:, -21:] = True

        x = x[:, :Lmax]  # Adjusting the input sequence based on the mask
        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos)
        #pos = self.pos_enc(pos, Lmax)


        x = x + pos     #Add positional encoding

        x = self.first_dropout(x)
        x = x.transpose(0, 1)  # Transpose to seq_len, batch, dim for transformer

        clipped_mask = ~clipped_mask  # As we clipped the sequence earlier, we can just invert the mask
        for enc_layer in self.enc_layers:
            x = enc_layer(x, padding_mask=clipped_mask, primer_mask=None)

        x = x.transpose(0, 1)  # Revert back to batch, seq_len, dim for LSTM

        if self.lstm_layers is not None:
            lengths = mask.sum(dim=1)
            x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            for lstm in self.lstm_layers:
                x_packed, _ = lstm(x_packed)
            x, _ = pad_packed_sequence(x_packed, batch_first=True)  # Unpack sequence

        return x



class RNAFormer(nn.Module):
    def __init__(self):  # Added parameters for the embedding layer
        super(RNAFormer, self).__init__()
        self.seq_embedding = nn.Embedding(4, CFG['rna_model_dim']) #, dropout=0.1)  # Added embedding layer
        self.encoder = RNAEncoder()
        last_dim = CFG['rna_model_dim'] * (2 if CFG['rna_model_num_lstm_layers'] > 0 else 1)
        self.last_linear = nn.Linear(last_dim, 2)
        self.dropout = nn.Dropout(CFG['rna_model_first_dropout'])

    def forward(self, x):
        seq = x['seq']
        # Pass the integer-encoded k-mers through the embedding layer
        embeded_seq= self.seq_embedding(seq)#.long())  # Ensure seq is of type long
        #embedded_kmers = self.dropout(embedded_kmers)
        mask = x['mask']
        encoded_seq = self.encoder(embeded_seq, mask)  # Changed 'seq' to 'embedded_kmers'
        encoded_seq = self.dropout(encoded_seq)  # Add dropout before the final layer

        output = self.last_linear(encoded_seq)
        #print(f"Input Size: {x['seq'].size()}, mask size: {x['mask'].size()}, output size: {output.size()}")

        return output



