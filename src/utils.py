import re
import unicodedata
from collections import Counter
from pyvi import ViTokenizer
from sklearn.model_selection import train_test_split


class TranslationPreprocessor: 
    def __init__(self, min_freq_vi=1, min_freq_en=1):
        self.min_freq_vi = min_freq_vi
        self.min_freq_en = min_freq_en
        self.vi_vocab = {}
        self.en_vocab = {}
    
    # ===== Normalization ===== 
    def unicode_to_ascii(self, s):
        return "".join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    def normalize_en(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", " ", s)
        return s
    
    def normalize_vi(self, s):
        s = s.lower().strip()
        s = re.sub(r"[“”\"‘’]", "", s)
        s = re.sub(r"\s+", " ", s)
        s = ViTokenizer.tokenize(s)
        return s
    
    # ===== Pair Preprocessing =====
    def preprocess_pairs(self, vi_sentences, en_sentences):
        processed = []
        for vi, en in zip(vi_sentences, en_sentences):
            vi_norm = self.normalize_vi(vi)
            en_norm = self.normalize_en(en)
            en_final = f"<sos> {en_norm} <eos>"
            processed.append([vi_norm, en_final])
        return processed
    
    def tokenize_pairs(self, processed_data):
        tokenized_data = []
        for vi, en in processed_data:
            vi_tokens = vi.split()
            en_tokens = en.split()
            tokenized_data.append((vi_tokens, en_tokens))
        return tokenized_data
        
    # ===== Vocabulary =====
    def build_vocab(self, tokenized_data, idx, min_freq, specials):
        counter = Counter()
        for vi_tokens, en_tokens in tokenized_data:
            if idx == 0:
                counter.update(vi_tokens)
            else:
                counter.update(en_tokens)

        vocab = {}
        vocab_idx = 0
        for token in specials:
            vocab[token] = vocab_idx
            vocab_idx += 1

        for token, freq in counter.items():
            if freq >= min_freq and token not in vocab:
                vocab[token] = vocab_idx
                vocab_idx += 1
        return vocab


    def build_vocabs(self, tokenized_train):
        self.vi_vocab = self.build_vocab(tokenized_train, 0, self.min_freq_vi, ["<pad>", "<unk>"])
        self.en_vocab = self.build_vocab(tokenized_train, 1, self.min_freq_en, ["<pad>", "<unk>", "<sos>", "<eos>"])
        return self.vi_vocab, self.en_vocab
    

    # ===== Numericalization =====
    def numericalize(self, tokens, vocab):
        return [vocab.get(token, vocab["<unk>"]) for token in tokens]

    def numericalize_dataset(self, tokenized_data):
        return [
            (self.numericalize(vi, self.vi_vocab), self.numericalize(en, self.en_vocab))
            for vi, en in tokenized_data
        ]
    
    def prepare_datasets(self, vi_sentences, en_sentences, test_size=0.2, val_size=0.5, seed=42):
        processed = self.preprocess_pairs(vi_sentences, en_sentences)
        tokenized = self.tokenize_pairs(processed)

        # split data
        train_data, temp_data = train_test_split(tokenized, test_size=test_size, random_state=seed)
        val_data, test_data = train_test_split(temp_data, test_size=val_size, random_state=seed)

        # build vocabs
        self.build_vocabs(train_data)

        # numericalize
        train_numericalized = self.numericalize_dataset(train_data)
        val_numericalized = self.numericalize_dataset(val_data)
        test_numericalized = self.numericalize_dataset(test_data)

        # vocab sizes
        all_src_tokens = {t for src, _ in train_numericalized + val_numericalized + test_numericalized for t in src}
        all_tgt_tokens = {t for _, tgt in train_numericalized + val_numericalized + test_numericalized for t in tgt}

        vocab_size_src = max(all_src_tokens) + 1
        vocab_size_tgt = max(all_tgt_tokens) + 1

        return {
            "train": train_numericalized,
            "val": val_numericalized,
            "test": test_numericalized,
            "vocab_src": self.vi_vocab,
            "vocab_tgt": self.en_vocab,
            "vocab_size_src": vocab_size_src,
            "vocab_size_tgt": vocab_size_tgt
        }