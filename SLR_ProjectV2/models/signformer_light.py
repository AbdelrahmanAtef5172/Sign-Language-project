# models/signformer_light.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class SignformerLight(nn.Module):
    def __init__(self, input_dim=75, vocab_size=30, d_model=128, nhead=4, num_layers=2, max_len=50):
        super().__init__()
        
        # 1. 1D Convolution
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(d_model)
        self.activation = nn.ReLU()
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # 3. Transformer Encoder (Dropout 0.1 عشان ميحفظش زيادة)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Transformer Decoder (Dropout 0.1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 5. Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # 6. Target Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, src, tgt):
        src = src.permute(0, 2, 1)
        src = self.conv1d(src)
        src = self.bn(src)
        src = self.activation(src)
        src = src.permute(0, 2, 1)
        src = self.pos_encoder(src)
        memory = self.encoder(src)
        
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)) * float('-inf'), diagonal=1).to(src.device)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output