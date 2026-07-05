# data_utils/data_loader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SLR_Dataset(Dataset):
    def __init__(self, data_path, max_len=50):
        # 1. Load the consolidated keypoints (الجمل القصيرة)
        self.X = np.load(os.path.join(data_path, 'X_short.npy'))
        
        # 2. Load the consolidated sentences (الجمل القصيرة)
        self.y = np.load(os.path.join(data_path, 'y_short.npy'), allow_pickle=True)
        
        print(f"✅ Loaded {len(self.X)} samples.")
        
        # 4. Build Character-Level Vocabulary
        chars = set()
        for sent in self.y:
            chars.update(str(sent).lower())
        
        self.char2idx = {ch: i+3 for i, ch in enumerate(sorted(chars))}
        self.char2idx['<PAD>'] = 0
        self.char2idx['<SOS>'] = 1
        self.char2idx['<EOS>'] = 2
        
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        self.max_len = max_len
        
    def encode_sentence(self, sentence):
        sentence = str(sentence).lower()
        indices = [self.char2idx['<SOS>']] + [self.char2idx.get(ch, 0) for ch in sentence] + [self.char2idx['<EOS>']]
        
        if len(indices) < self.max_len:
            indices += [self.char2idx['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = self.encode_sentence(self.y[idx])
        return x, y