"""
Model Loading Utilities for Urdu Chatbot
Loads the trained Transformer model and tokenizer from saved files
"""

import torch
import torch.nn as nn
from pathlib import Path
import pickle
import sys
import io


class SafeUnpickler(pickle.Unpickler):
    """Custom unpickler that handles missing classes"""
    def find_class(self, module, name):
        if name == 'Counter':
            from collections import Counter
            return Counter
        if name == 'UrduTokenizer':
            return UrduTokenizer
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            print(f"Warning: Could not find class {module}.{name}")
            return dict


class UrduTokenizer:
    """Wrapper for tokenizer loaded from pickle"""
    def __init__(self, vocab=None):
        if vocab is None:
            vocab = []

        self.vocab = vocab if isinstance(vocab, list) else list(vocab) if vocab else []
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.vocab)

        self.pad_idx = 0
        self.unk_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3
        self.min_freq = 1
        self.word_freq = {}

    def encode(self, text, add_special_tokens=True):
        """Encode text to token IDs"""
        words = text.strip().split()
        tokens = []

        if add_special_tokens:
            tokens.append(self.sos_idx)

        for word in words:
            token_id = self.word2idx.get(word, self.unk_idx)
            tokens.append(token_id)

        if add_special_tokens:
            tokens.append(self.eos_idx)

        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs to text"""
        words = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in [self.pad_idx, self.sos_idx, self.eos_idx, self.unk_idx]:
                continue
            if token_id < len(self.idx2word):
                words.append(self.idx2word[token_id])
        return ' '.join(words)


def load_tokenizer(vocab_path):
    """Load the Urdu tokenizer from saved vocabulary file"""
    vocab_path = Path(vocab_path)
    tokenizer_file = vocab_path / 'tokenizer.pkl'

    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")

    try:
        with open(tokenizer_file, 'rb') as f:
            unpickler = SafeUnpickler(f)
            tokenizer_data = unpickler.load()

        if isinstance(tokenizer_data, UrduTokenizer):
            if hasattr(tokenizer_data, 'word2idx') and len(tokenizer_data.word2idx) > 0:
                tokenizer = tokenizer_data
                if not hasattr(tokenizer, 'vocab_size'):
                    tokenizer.vocab_size = len(tokenizer.word2idx)
                if not hasattr(tokenizer, 'sos_idx'):
                    tokenizer.sos_idx = 2
                if not hasattr(tokenizer, 'eos_idx'):
                    tokenizer.eos_idx = 3
                if not hasattr(tokenizer, 'pad_idx'):
                    tokenizer.pad_idx = 0
                if not hasattr(tokenizer, 'unk_idx'):
                    tokenizer.unk_idx = 1
                if not hasattr(tokenizer, 'encode'):
                    def encode(text, add_special_tokens=True):
                        words = text.strip().split()
                        tokens = []
                        if add_special_tokens:
                            tokens.append(tokenizer.sos_idx)
                        for word in words:
                            token_id = tokenizer.word2idx.get(word, tokenizer.unk_idx)
                            tokens.append(token_id)
                        if add_special_tokens:
                            tokens.append(tokenizer.eos_idx)
                        return tokens
                    tokenizer.encode = encode
                if not hasattr(tokenizer, 'decode'):
                    def decode(token_ids, skip_special_tokens=True):
                        words = []
                        for token_id in token_ids:
                            if skip_special_tokens and token_id in [tokenizer.pad_idx, tokenizer.sos_idx, tokenizer.eos_idx, tokenizer.unk_idx]:
                                continue
                            if token_id < len(tokenizer.idx2word):
                                words.append(tokenizer.idx2word.get(token_id, '<UNK>'))
                        return ' '.join(words)
                    tokenizer.decode = decode
            elif hasattr(tokenizer_data, 'vocab'):
                vocab = tokenizer_data.vocab
                if isinstance(vocab, dict):
                    vocab = list(vocab.keys())
                tokenizer = UrduTokenizer(vocab)
            else:
                vocab_list = list(tokenizer_data.word2idx.keys()) if hasattr(tokenizer_data, 'word2idx') else ['<pad>', '<unk>', '<sos>', '<eos>']
                tokenizer = UrduTokenizer(vocab_list)
        elif hasattr(tokenizer_data, 'word2idx') and len(tokenizer_data.word2idx) > 0:
            tokenizer = tokenizer_data
            if not hasattr(tokenizer, 'vocab_size'):
                tokenizer.vocab_size = len(tokenizer.word2idx)
            if not hasattr(tokenizer, 'encode') or not callable(getattr(tokenizer, 'encode', None)):
                def encode(text, add_special_tokens=True):
                    words = text.strip().split()
                    tokens = []
                    if add_special_tokens and hasattr(tokenizer, 'sos_idx'):
                        tokens.append(tokenizer.sos_idx)
                    for word in words:
                        unk_idx = getattr(tokenizer, 'unk_idx', 1)
                        token_id = tokenizer.word2idx.get(word, unk_idx)
                        tokens.append(token_id)
                    if add_special_tokens and hasattr(tokenizer, 'eos_idx'):
                        tokens.append(tokenizer.eos_idx)
                    return tokens
                tokenizer.encode = encode
        elif hasattr(tokenizer_data, 'vocab'):
            vocab = tokenizer_data.vocab
            if isinstance(vocab, dict):
                vocab = list(vocab.keys())
            tokenizer = UrduTokenizer(vocab)
        elif isinstance(tokenizer_data, dict):
            tokenizer = UrduTokenizer(list(tokenizer_data.keys()))
        elif isinstance(tokenizer_data, (list, tuple)):
            tokenizer = UrduTokenizer(list(tokenizer_data))
        else:
            tokenizer = UrduTokenizer(['<pad>', '<unk>', '<sos>', '<eos>'])
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Using minimal vocabulary...")
        tokenizer = UrduTokenizer(['<pad>', '<unk>', '<sos>', '<eos>'])

    print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    return tokenizer


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model: int, max_seq_length: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention"""
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]

        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        output, attention_weights = self.attention(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """Feed-forward network"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Encoder layer"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attention(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.norm2(x + ffn_output)

        return x


class Encoder(nn.Module):
    """Encoder"""
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


class DecoderLayer(nn.Module):
    """Decoder layer"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        attn_output, _ = self.self_attention(x, x, x, self_mask)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)

        cross_output, _ = self.cross_attention(x, encoder_output, encoder_output, cross_mask)
        cross_output = self.dropout(cross_output)
        x = self.norm2(x + cross_output)

        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.norm3(x + ffn_output)

        return x


class Decoder(nn.Module):
    """Decoder"""
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)

        return x


class Transformer(nn.Module):
    """Transformer model"""
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256,
                 n_heads: int = 2, n_encoder_layers: int = 2, n_decoder_layers: int = 2,
                 d_ff: int = 1024, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_heads, n_encoder_layers, d_ff, dropout, pad_idx)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_heads, n_decoder_layers, d_ff, dropout, pad_idx)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self.pad_idx = pad_idx

    def create_source_mask(self, src):
        """Create padding mask for source"""
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def create_target_mask(self, tgt):
        """Create causal mask for target"""
        batch_size, seq_len = tgt.shape

        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.uint8, device=tgt.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        mask = pad_mask & causal_mask
        return mask

    def forward(self, src, tgt):
        src_mask = self.create_source_mask(src)
        tgt_mask = self.create_target_mask(tgt)

        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        output = self.output_projection(decoder_output)

        return output


def load_transformer_model(model_path, device='cpu'):
    """Load the trained Transformer model from checkpoint"""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    print(f"Loading model from epoch {checkpoint.get('epoch', 'unknown')}")

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    config = checkpoint.get('config', {})

    vocab_size = config.get('src_vocab_size', 11276)
    d_model = config.get('d_model', 256)
    n_heads = config.get('n_heads', 2)
    n_encoder_layers = config.get('n_encoder_layers', 2)
    n_decoder_layers = config.get('n_decoder_layers', 2)
    d_ff = config.get('d_ff', 1024)
    dropout = config.get('dropout', 0.3)
    pad_idx = config.get('pad_idx', 0)

    print(f"Model config: vocab_size={vocab_size}, d_model={d_model}, n_heads={n_heads}")

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        dropout=dropout,
        pad_idx=pad_idx
    )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"State dict loading error: {str(e)[:100]}")
        raise

    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Best BLEU: {checkpoint.get('best_bleu', 'N/A')}")

    return model


def get_model_info(model_path):
    """Get information about a saved model without loading it fully"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'best_bleu': checkpoint.get('best_bleu', 'N/A'),
        'config': checkpoint.get('config', {}),
        'vocab_sizes': {
            'source': checkpoint.get('config', {}).get('src_vocab_size', 'unknown'),
            'target': checkpoint.get('config', {}).get('tgt_vocab_size', 'unknown'),
        }
    }

    return info


if __name__ == "__main__":
    print("Model loader implementation complete!")
    print("Expected structure:")
    print("  models/")
    print("  ├── best_model.pt")
    print("  └── vocabulary/")
    print("      └── tokenizer.pkl")
