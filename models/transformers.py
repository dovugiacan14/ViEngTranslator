import math
import torch 
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        # register initial positional encoding buffer
        self.register_buffer('pe', self._generate_pe(max_len, d_model))

    def _generate_pe(self, max_len, d_model):
        """
        Generate the positional encoding matrix for a given maximum length and model dimension.
        Shape: (max_len, 1, d_model)
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(1) 
        return pe

    def forward(self, x):
        seq_len = x.size(0)

        # If the incoming sequence is longer than the current PE, regenerate it
        if seq_len > self.pe.size(0):
            new_pe = self._generate_pe(seq_len, self.d_model).to(x.device)
            self.pe = new_pe
            self.max_len = seq_len

        # Add positional encoding to embeddings
        x = x + self.pe[:seq_len, :]

        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, config):
        super().__init__()
        self.d_model = config.d_model
        self.pad_idx = config.ignore_index
        self.config = config
        
        # embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, config.d_model, padding_idx=config.ignore_index)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, config.d_model, padding_idx=config.ignore_index)

        # positional encodings 
        self.pos_encoding = PositionalEncoding(
            d_model=config.d_model,
            dropout=config.dropout,
            max_len=config.max_length
        )

        # built Transformer by Pytorch 
        self.transformer = nn.Transformer(
            d_model = config.d_model,
            nhead = config.nhead, 
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False
        )

        self.output_projection = nn.Linear(config.d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def create_mask(self, src, tgt):
        tgt_seq_len = tgt.shape[1]

        src_padding_mask = (src == self.pad_idx)   # source padding mask 
        tgt_padding_mask = (tgt == self.pad_idx)   # target padding mask 

        # target look-ahead mask 
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device) 
        return src_padding_mask, tgt_padding_mask, tgt_mask
    
    def forward(self, src, tgt): 
        # create masks 
        src_padding_mask, tgt_padding_mask, tgt_mask = self.create_mask(src, tgt[:, :-1])
        
        # Embedding + scaling + positional encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.config.d_model)  
        tgt_emb = self.tgt_embedding(tgt[:, :-1]) * math.sqrt(self.config.d_model)  
        
        # transpose for format
        src_emb = src_emb.transpose(0, 1)  
        tgt_emb = tgt_emb.transpose(0, 1)  
        
        #add positional encoding
        src_emb = self.pos_encoding(src_emb)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # transformer forward 
        output = self.transformer(
            src_emb, 
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        output = output.transpose(0, 1) 
        output = self.output_projection(output) 
        
        return output