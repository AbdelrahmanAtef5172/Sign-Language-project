# evaluate.py - النسخة النهائية المتوافقة مع النموذج الكبير (128,4,2)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from data_utils.data_loader import SLR_Dataset
from models.signformer_light import SignformerLight
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import random

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ====================================
# CONFIGURATION (مظبوطة على النموذج الجديد)
# ====================================
DATA_PATH = "extracted_5000"
CHECKPOINT_PATH = "checkpoints/model_best.pth"   # أفضل نموذج (Loss 0.016)
TEST_SIZE = 0.2
BEAM_WIDTH = 3                                    # أفضل جودة للترجمة
# ====================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Running on: {device}")

dataset = SLR_Dataset(DATA_PATH)
print(f"[INFO] Loaded {len(dataset)} samples.")

# ====== النموذج الكبير (2.5 مليون معامل) المطابق للتدريب ======
model = SignformerLight(
    input_dim=75,
    vocab_size=dataset.vocab_size,
    d_model=128,
    nhead=4,
    num_layers=2
).to(device)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print("[INFO] Model loaded successfully.")

indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=42)
print(f"[INFO] Number of test samples: {len(test_idx)}")

def translate_with_beam(model, sample_keypoints, dataset, beam_width=BEAM_WIDTH, max_len=50):
    with torch.no_grad():
        src = sample_keypoints.unsqueeze(0).to(device)
        beam = [(torch.tensor([[1]], device=device), 0.0)]
        
        for _ in range(max_len - 1):
            new_beam = []
            for seq, score in beam:
                if seq[0, -1].item() == 2:
                    new_beam.append((seq, score))
                    continue
                
                output = model(src, seq)
                log_probs = torch.log_softmax(output[0, -1, :], dim=-1)
                top_log_probs, top_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    next_token = top_indices[i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, next_token], dim=1)
                    new_score = score + top_log_probs[i].item()
                    new_beam.append((new_seq, new_score))
            
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]
            
            if all(seq[0, -1].item() == 2 for seq, _ in beam):
                break
        
        best_seq = beam[0][0]
        indices = best_seq.squeeze().tolist()
        sentence = ''.join([dataset.idx2char[idx] for idx in indices if idx not in [0, 1, 2]])
        return sentence

def decode_truth(sample_y, dataset):
    indices = sample_y.tolist()
    return ''.join([dataset.idx2char[idx] for idx in indices if idx not in [0, 1, 2]])

bleu_scores_1 = []
bleu_scores_4 = []
exact_matches = 0
total_chars = 0
correct_chars = 0
smoothie = SmoothingFunction().method4

print(f"\n[INFO] Starting evaluation on {len(test_idx)} samples with Beam Search...")
print("-" * 50)

for idx in test_idx:
    x, y = dataset[idx]
    pred = translate_with_beam(model, x, dataset, beam_width=BEAM_WIDTH)
    truth = decode_truth(y, dataset)
    
    min_len = min(len(pred), len(truth))
    correct_chars += sum(1 for i in range(min_len) if pred[i] == truth[i])
    total_chars += max(len(pred), len(truth))
    
    if pred.strip() == truth.strip():
        exact_matches += 1
    
    pred_tokens = pred.split()
    truth_tokens = truth.split()
    
    if pred_tokens:
        try:
            score1 = sentence_bleu([truth_tokens], pred_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smoothie)
            bleu_scores_1.append(score1)
            score4 = sentence_bleu([truth_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
            bleu_scores_4.append(score4)
        except:
            bleu_scores_1.append(0)
            bleu_scores_4.append(0)
    else:
        bleu_scores_1.append(0)
        bleu_scores_4.append(0)

    if len([i for i in test_idx if i <= idx]) <= 10:
        print(f"Ground Truth: {truth}")
        print(f"Predicted:    {pred}")
        print("-" * 30)

char_acc = correct_chars / total_chars if total_chars > 0 else 0
avg_bleu1 = np.mean(bleu_scores_1) if bleu_scores_1 else 0
avg_bleu4 = np.mean(bleu_scores_4) if bleu_scores_4 else 0
exact_match_acc = exact_matches / len(test_idx) if test_idx else 0

print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"Test samples: {len(test_idx)}")
print(f"Character Accuracy: {char_acc * 100:.2f}%")
print(f"Exact Match Accuracy: {exact_match_acc * 100:.2f}%")
print(f"Average BLEU-1 (Word Accuracy): {avg_bleu1:.4f}")
print(f"Average BLEU-4 (Sentence Quality): {avg_bleu4:.4f}")
print("=" * 50)