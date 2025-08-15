import torch 
from torch import nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size_src, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size_src, hidden_size, padding_idx= 0)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional= False
        )
    
    def init_hidden(self, batch_size):
        # initialize hidden state (h0) and cell state (c0) filled with zeros 
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)
    
    def forward(self, src_tokens, hidden):
        emb = self.embedding(src_tokens)  # convert token IDs to embeddings 
        emb = emb.transpose(0, 1)         # switch to (seq_len, batch, hidden) for LSTM
        outputs, hidden = self.lstm(emb, hidden)
        outputs = outputs.transpose(0, 1)
        return outputs, hidden


class AttentionDecoderLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size_target, num_layers, dropout):
        super(AttentionDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size_target = vocab_size_target

        self.embedding = nn.Embedding(vocab_size_target, hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # combine [context; embedding] into hidden size 
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional= False
        )

        # final output projection 
        self.out = nn.Linear(hidden_size, vocab_size_target)
    
    def forward(self, input_tokens, hidden, encoder_outputs, src_mask=None):
        embedded = self.embedding(input_tokens)   # convert token IDs to embeddings 
        embedded = self.dropout(embedded)

        # dot-product attention: score = encoder_output * decoder_hidden_last 
        h_last = hidden[0][-1]
        scores = torch.bmm(encoder_outputs, h_last.unsqueeze(2)).squeeze(2)

        if src_mask is not None: 
            scores = scores.masked_fill(src_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim= 1)   # softmax to get attention weights

        # compute context vector as weighted sum of encode outputs 
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1) 

        # concatenate context with embedding and transform 
        combo = torch.cat([embedded, context], dim=1) 
        combo = torch.tanh(self.attn_combine(combo))

        # LSTM expects (seq_len=1, batch, hidden)
        lstm_in = combo.unsqueeze(0) 
        output, hidden_next = self.lstm(lstm_in, hidden)
        output = output.squeeze(0)
        log_probs = F.log_softmax(self.out(output), dim=1)

        return log_probs, hidden_next, attn_weights