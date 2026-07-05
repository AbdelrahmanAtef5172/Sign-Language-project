# evaluate_advanced.py - تقييم متقدم (WER, ROUGE-L, Word Accuracy)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from data_utils.data_loader import SLR_Dataset
from models.signformer_light import SignformerLight
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics import edit_distance
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
# CONFIGURATION
# ====================================
DATA_PATH = "extracted_5000"
CHECKPOINT_PATH = "checkpoints/model_best.pth"
TEST_SIZE = 0.2
BEAM_WIDTH = 3
# ====================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Running on: {device}")

dataset = SLR_Dataset(DATA_PATH)
print(f"[INFO] Loaded {len(dataset)} samples.")

# ====== تحميل النموذج الكبير (128,4,2) ======
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
        return ''.join([dataset.idx2char[idx] for idx in indices if idx not in [0, 1, 2]])

def decode_truth(sample_y, dataset):
    indices = sample_y.tolist()
    return ''.join([dataset.idx2char[idx] for idx in indices if idx not in [0, 1, 2]])

# ====== حساب WER (Word Error Rate) ======
def calculate_wer(ref, hyp):
    """نسبة الكلمات اللي غلط (كل ما قلّت كل ما كانت أحسن)"""
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 0.0
    # نحسب أقل عدد من التعديلات (إضافة، حذف، تغيير) عشان نحول الـ hyp لـ ref
    distance = edit_distance(ref_words, hyp_words)
    return distance / len(ref_words)

# ====== حساب ROUGE-L (أطول جملة مشتركة متتالية) ======
def calculate_rouge_l(ref, hyp):
    """نسبة طول أطول جملة متتالية مشتركة (كل ما زادت كل ما كانت أحسن)"""
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words or not hyp_words:
        return 0.0
    # أطول تتابع مشترك (LCS)
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]
    # F1-score بين الدقة والاسترجاع
    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# ====== حساب Word Accuracy (دقة الكلمات من غير ترتيب) ======
def calculate_word_accuracy(ref, hyp):
    """نسبة الكلمات المشتركة (الموجودة في الجملتين)"""
    ref_words = set(ref.lower().split())
    hyp_words = set(hyp.lower().split())
    if not ref_words:
        return 0.0
    intersection = ref_words.intersection(hyp_words)
    return len(intersection) / len(ref_words)

# ====== التقييم الفعلي ======
bleu_scores_1, bleu_scores_4 = [], []
exact_matches = 0
total_chars, correct_chars = 0, 0
wer_scores = []
rouge_l_scores = []
word_acc_scores = []
smoothie = SmoothingFunction().method4

print(f"\n[INFO] Advanced evaluation on {len(test_idx)} samples...")
print("-" * 50)

for idx in test_idx:
    x, y = dataset[idx]
    pred = translate_with_beam(model, x, dataset, beam_width=BEAM_WIDTH)
    truth = decode_truth(y, dataset)
    
    # 1. Character Accuracy
    min_len = min(len(pred), len(truth))
    correct_chars += sum(1 for i in range(min_len) if pred[i] == truth[i])
    total_chars += max(len(pred), len(truth))
    
    # 2. Exact Match
    if pred.strip() == truth.strip():
        exact_matches += 1
    
    # 3. BLEU
    pred_tokens = pred.split()
    truth_tokens = truth.split()
    if pred_tokens:
        try:
            bleu_scores_1.append(sentence_bleu([truth_tokens], pred_tokens, weights=(1,0,0,0), smoothing_function=smoothie))
            bleu_scores_4.append(sentence_bleu([truth_tokens], pred_tokens, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie))
        except:
            bleu_scores_1.append(0); bleu_scores_4.append(0)
    else:
        bleu_scores_1.append(0); bleu_scores_4.append(0)
    
    # 4. WER (كل ما قلّ كان أحسن)
    wer_scores.append(calculate_wer(truth, pred))
    
    # 5. ROUGE-L (كل ما زاد كان أحسن)
    rouge_l_scores.append(calculate_rouge_l(truth, pred))
    
    # 6. Word Accuracy (كل ما زاد كان أحسن)
    word_acc_scores.append(calculate_word_accuracy(truth, pred))

    # Print first 10 samples
    if len([i for i in test_idx if i <= idx]) <= 10:
        print(f"Truth: {truth}")
        print(f"Pred : {pred}")
        print(f"WER : {wer_scores[-1]:.3f}, ROUGE-L: {rouge_l_scores[-1]:.3f}")
        print("-" * 30)

# ====== النتائج النهائية ======
char_acc = correct_chars / total_chars if total_chars > 0 else 0
exact_match_acc = exact_matches / len(test_idx) if test_idx else 0
avg_bleu1 = np.mean(bleu_scores_1) if bleu_scores_1 else 0
avg_bleu4 = np.mean(bleu_scores_4) if bleu_scores_4 else 0
avg_wer = np.mean(wer_scores) if wer_scores else 0
avg_rouge_l = np.mean(rouge_l_scores) if rouge_l_scores else 0
avg_word_acc = np.mean(word_acc_scores) if word_acc_scores else 0

print("\n" + "=" * 60)
print("ADVANCED EVALUATION RESULTS")
print("=" * 60)
print(f"Test samples          : {len(test_idx)}")
print(f"Character Accuracy   : {char_acc * 100:.2f}%")
print(f"Exact Match Accuracy : {exact_match_acc * 100:.2f}%")
print(f"BLEU-1               : {avg_bleu1:.4f}")
print(f"BLEU-4               : {avg_bleu4:.4f}")
print("-" * 60)
print(f"⭐ Word Accuracy      : {avg_word_acc * 100:.2f}%   (نسبة الكلمات الصح من غير ترتيب)")
print(f"⭐ ROUGE-L            : {avg_rouge_l:.4f}          (تشابه الجمل المتتابعة)")
print(f"⭐ WER (Word Error)   : {avg_wer:.4f}          (الأقل أفضل - 0 = مثالي)")
print("=" * 60)

# تفسير النتائج
print("\n📊 التفسير:")
if avg_wer < 0.1:
    print("✅ WER أقل من 0.1 => الترجمة ممتازة جداً (كلمات قليلة غلط)")
elif avg_wer < 0.3:
    print("👍 WER بين 0.1 و 0.3 => ترجمة جيدة (فيها أخطاء بسيطة)")
else:
    print("⚠️ WER أعلى من 0.3 => لسه محتاج تحسين")

if avg_word_acc > 0.9:
    print("✅ Word Accuracy أعلى من 90% => معظم الكلمات مترجمة صح")
elif avg_word_acc > 0.7:
    print("👍 Word Accuracy بين 70-90% => ترجمة مقبولة")