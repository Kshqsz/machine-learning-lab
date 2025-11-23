"""
RNN Language Model (PyTorch)
----------------------------
- Builds word-level vocabulary and embeddings
- Trains a simple RNN/LSTM language model (Embedding -> RNN/LSTM -> Linear -> Softmax)
- Includes sampling function for language generation with temperature

Usage examples:

# Train on bundled tiny sample text (few epochs for demo):
python3 rnn/rnn_language_model.py --mode train --epochs 10 --seq_len 20 --batch_size 64

# Generate text from a checkpoint (after training):
python3 rnn/rnn_language_model.py --mode generate --checkpoint checkpoint.pth --seed "Once upon a time" --gen_len 50 --temperature 1.0

Requirements: torch, tqdm (optional)
"""

import argparse
import os
import random
import math
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x

# --------------------------- Utilities & Data ---------------------------

SAMPLE_TEXT = """
In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole,
filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole
with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means
comfort.
"""

SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>']


def basic_tokenize(text: str) -> List[str]:
    # Very simple whitespace tokenizer — keep punctuation attached (good enough for demo)
    return text.strip().split()


class Vocab:
    def __init__(self, min_freq: int = 1, max_size: int = None):
        self.token2idx: Dict[str, int] = {}
        self.idx2token: List[str] = []
        self.freqs: Dict[str, int] = {}
        for t in SPECIAL_TOKENS:
            self.add_token(t)
        self.min_freq = min_freq
        self.max_size = max_size

    def add_token(self, token: str):
        if token in self.token2idx:
            self.freqs[token] += 1
        else:
            idx = len(self.idx2token)
            self.token2idx[token] = idx
            self.idx2token.append(token)
            self.freqs[token] = 1

    def build_from_tokens(self, tokens: List[str]):
        for t in tokens:
            self.freqs[t] = self.freqs.get(t, 0) + 1
        # keep special tokens first
        items = [(t, c) for t, c in self.freqs.items() if t not in SPECIAL_TOKENS]
        items.sort(key=lambda x: (-x[1], x[0]))
        if self.max_size:
            items = items[:self.max_size - len(SPECIAL_TOKENS)]
        # reset
        self.token2idx = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.idx2token = list(SPECIAL_TOKENS)
        for t, c in items:
            if c >= self.min_freq and t not in self.token2idx:
                self.token2idx[t] = len(self.idx2token)
                self.idx2token.append(t)
        # update freqs only for known tokens
        self.freqs = {t: self.freqs[t] for t in self.idx2token}

    def __len__(self):
        return len(self.idx2token)

    def token_to_idx(self, token: str) -> int:
        return self.token2idx.get(token, self.token2idx['<unk>'])

    def idx_to_token(self, idx: int) -> str:
        return self.idx2token[idx]


class WordDataset(Dataset):
    def __init__(self, tokens: List[str], vocab: Vocab, seq_len: int = 20):
        # tokens: full list of tokens (words)
        self.vocab = vocab
        self.seq_len = seq_len
        self.indices = [vocab.token_to_idx(t) for t in tokens]

    def __len__(self):
        if len(self.indices) <= 1:
            return 0
        return max(0, len(self.indices) - self.seq_len)

    def __getitem__(self, idx):
        x = self.indices[idx: idx + self.seq_len]
        y = self.indices[idx + 1: idx + 1 + self.seq_len]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ------------------------------- Model ---------------------------------

class RNNLanguageModel(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 1,
                 rnn_type: str = 'lstm',
                 dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        else:
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.rnn_type = rnn_type.lower()

    def forward(self, x: torch.LongTensor, hidden=None):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        if hidden is None:
            out, hidden = self.rnn(emb)
        else:
            out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)  # (batch, seq_len, vocab_size)
        return logits, hidden


# ------------------------------ Sampling -------------------------------

def sample_text(model: RNNLanguageModel, vocab: Vocab, seed_text: str = None, gen_len: int = 50, temperature: float = 1.0, device: torch.device = torch.device('cpu')) -> str:
    model.eval()
    if seed_text is None or seed_text.strip() == '':
        seed_tokens = ['<bos>']
    else:
        seed_tokens = basic_tokenize(seed_text)
    indices = [vocab.token_to_idx(t) for t in seed_tokens]
    input_seq = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)

    hidden = None
    generated = seed_tokens.copy()

    with torch.no_grad():
        # feed seed
        logits, hidden = model(input_seq, hidden)
        last_idx = input_seq[0, -1].unsqueeze(0).unsqueeze(0)  # shape (1,1)
        for _ in range(gen_len):
            logits, hidden = model(last_idx, hidden)
            logits = logits[:, -1, :].squeeze(0)  # (vocab_size,)
            if temperature <= 0:
                probs = torch.softmax(logits, dim=-1)
                next_idx = torch.argmax(probs).item()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1).item()
            token = vocab.idx_to_token(next_idx)
            generated.append(token)
            last_idx = torch.tensor([[next_idx]], dtype=torch.long, device=device)
            if token == '<eos>':
                break
    return ' '.join(generated)


# ------------------------------- Training --------------------------------

def train_loop(model: RNNLanguageModel, dataloader: DataLoader, optimizer, criterion, device: torch.device, clip: float = 1.0):
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(x_batch)
        # logits: (batch, seq_len, vocab_size) -> reshape for CE
        batch, seq_len, vocab_size = logits.shape
        loss = criterion(logits.view(batch * seq_len, vocab_size), y_batch.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item() * batch
    return total_loss / len(dataloader.dataset)


def evaluate_loop(model: RNNLanguageModel, dataloader: DataLoader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits, _ = model(x_batch)
            batch, seq_len, vocab_size = logits.shape
            loss = criterion(logits.view(batch * seq_len, vocab_size), y_batch.view(-1))
            total_loss += loss.item() * batch
    return total_loss / len(dataloader.dataset)


# ------------------------------- Main / CLI -------------------------------

def read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def build_dataset_from_text(text: str, min_freq: int = 1, max_vocab: int = None) -> Tuple[List[str], Vocab]:
    tokens = basic_tokenize(text)
    vocab = Vocab(min_freq=min_freq, max_size=max_vocab)
    vocab.build_from_tokens(tokens)
    return tokens, vocab


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='train')
    parser.add_argument('--input', type=str, default=None, help='path to training text file')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--rnn_type', type=str, choices=['lstm', 'gru'], default='lstm')
    parser.add_argument('--seed', type=str, default='Once upon a time')
    parser.add_argument('--gen_len', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_every', type=int, default=5)
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    if args.mode == 'train':
        if args.input and os.path.exists(args.input):
            text = read_text(args.input)
        else:
            print('No input provided or file not found — using bundled sample text for demo.')
            text = SAMPLE_TEXT

        tokens, vocab = build_dataset_from_text(text)
        print(f'Vocab size: {len(vocab)}')
        dataset = WordDataset(tokens, vocab, seq_len=args.seq_len)
        if len(dataset) == 0:
            raise RuntimeError('Dataset too small. Provide longer training text or reduce seq_len.')
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = RNNLanguageModel(vocab_size=len(vocab), embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, rnn_type=args.rnn_type).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        best_loss = float('inf')
        for epoch in range(1, args.epochs + 1):
            train_loss = train_loop(model, dataloader, optimizer, criterion, device)
            val_loss = train_loss  # no separate validation for this small demo
            print(f'Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}')
            if epoch % args.save_every == 0 or epoch == args.epochs:
                ckpt = {
                    'model_state': model.state_dict(),
                    'vocab': vocab.token2idx,
                    'args': vars(args)
                }
                torch.save(ckpt, args.checkpoint)
                print(f'Saved checkpoint: {args.checkpoint}')

        print('Training finished.')

    elif args.mode == 'generate':
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        token2idx = ckpt['vocab']
        # rebuild Vocab object
        vocab = Vocab()
        vocab.token2idx = token2idx
        vocab.idx2token = [None] * len(token2idx)
        for t, i in token2idx.items():
            vocab.idx2token[i] = t

        # Use saved args if available, otherwise use command line args
        saved_args = ckpt.get('args', {})
        embed_dim = saved_args.get('embed_dim', args.embed_dim)
        hidden_dim = saved_args.get('hidden_dim', args.hidden_dim)
        num_layers = saved_args.get('num_layers', args.num_layers)
        rnn_type = saved_args.get('rnn_type', args.rnn_type)
        
        model = RNNLanguageModel(vocab_size=len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=num_layers, rnn_type=rnn_type)
        model.load_state_dict(ckpt['model_state'])
        model.to(device)

        out = sample_text(model, vocab, seed_text=args.seed, gen_len=args.gen_len, temperature=args.temperature, device=device)
        print('\n--- Generated Text ---')
        print(out)


if __name__ == '__main__':
    main()
