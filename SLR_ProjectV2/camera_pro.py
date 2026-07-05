# camera_pro.py - عرض نقاط الجسم + الأصابع مع ترجمة محسّنة
import cv2
import mediapipe as mp
import torch
import numpy as np
import sys
import os
import urllib.request
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.signformer_light import SignformerLight
from data_utils.data_loader import SLR_Dataset

# ============================================
DATA_PATH = "extracted_5000"
CHECKPOINT_PATH = "checkpoints/model_short.pth"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_PATH = "pose_landmarker.task"
HAND_MODEL_PATH = "hand_landmarker.task"

FRAMES = 12
CONFIDENCE = 0.5
TEMPERATURE = 0.7   # أقل حرارة = ترجمة أكثر تركيزاً
# ============================================

# تحميل نماذج MediaPipe
for url, path in [(POSE_MODEL_URL, POSE_MODEL_PATH), (HAND_MODEL_URL, HAND_MODEL_PATH)]:
    if not os.path.exists(path):
        print(f"📥 تحميل {path}...")
        urllib.request.urlretrieve(url, path)
        print("✅ تم التحميل.")

print("🔄 تحميل نموذج الترجمة...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = SLR_Dataset(DATA_PATH)

model = SignformerLight(
    input_dim=75,
    vocab_size=dataset.vocab_size,
    d_model=64,
    nhead=2,
    num_layers=1
).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print("✅ نموذج الترجمة جاهز!")

def translate(model, kp_seq):
    with torch.no_grad():
        src = torch.tensor(kp_seq, dtype=torch.float32).unsqueeze(0).to(device)
        tgt = torch.tensor([[1]], device=device)
        for _ in range(30):
            out = model(src, tgt)
            logits = out[0, -1, :] / TEMPERATURE
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).unsqueeze(0)
            tgt = torch.cat([tgt, nxt], dim=1)
            if nxt.item() == 2:
                break
        idx = tgt.squeeze().tolist()
        return ''.join([dataset.idx2char[i] for i in idx if i not in [0, 1, 2]])

# ====== MediaPipe Pose + Hands ======
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Pose (للجسم)
pose_opts = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
    min_pose_detection_confidence=CONFIDENCE
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_opts)

# 2. Hands (للأصابع)
hand_opts = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=CONFIDENCE
)
hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

# دالة استخراج نقاط الجسم (25 نقطة) فقط للنموذج
def extract_body(landmarks):
    if not landmarks:
        return [0.0] * 75
    lm = landmarks[0]
    mp_to_coco = {0: 0, 11: 5, 12: 2, 13: 6, 14: 3, 15: 7, 16: 4,
                  23: 11, 24: 8, 25: 12, 26: 9, 27: 13, 28: 10,
                  1: 14, 2: 15, 3: 16, 4: 17}
    coco = {}
    for m, c in mp_to_coco.items():
        p = lm[m]
        coco[c] = (p.x, p.y, p.z)
    # Neck
    if 2 in coco and 5 in coco:
        coco[1] = ((coco[5][0]+coco[2][0])/2, (coco[5][1]+coco[2][1])/2, (coco[5][2]+coco[2][2])/2)
    else:
        coco[1] = (0.0, 0.0, 0.0)
    for i in range(18, 25):
        coco[i] = (0.0, 0.0, 0.0)
    res = []
    for i in range(25):
        x, y, z = coco.get(i, (0.0, 0.0, 0.0))
        res.extend([x, y, z])
    return res

# ====== الكاميرا ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ كاميرا غير موجودة!")
    exit()

buffer = []
last_text = ""
last_time = 0
no_body = 0

print("🎥 الكاميرا شغالة...")
print("🔴 اضغط 'q' للخروج")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # كشف الجسم واليدين
    pose_res = pose_detector.detect(mp_img)
    hand_res = hand_detector.detect(mp_img)

    body_visible = pose_res.pose_landmarks is not None
    hands_visible = hand_res.hand_landmarks is not None

    # ====== الرسم ======
    # 1. رسم نقاط الجسم (أخضر)
    if body_visible:
        no_body = 0
        for lm in pose_res.pose_landmarks[0]:
            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, (0, 255, 0), -1)
        # جسم
        kp_body = extract_body(pose_res.pose_landmarks)
    else:
        no_body += 1
        kp_body = [0.0] * 75
        cv2.putText(frame, "⚠️ ابتعدي قليلاً", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 2. رسم نقاط اليدين (أصفر - Cyan) للعرض فقط
    if hands_visible:
        for hand in hand_res.hand_landmarks:
            for lm in hand:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (255, 255, 0), -1)
    else:
        cv2.putText(frame, "⚠️ ارفعي يديك", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ====== الترجمة (باستخدام نقاط الجسم فقط) ======
    if body_visible and len(kp_body) == 75:
        buffer.append(kp_body)
        if len(buffer) > FRAMES + 5:
            buffer = buffer[-FRAMES:]
        if len(buffer) >= FRAMES:
            pred = translate(model, buffer)
            if pred and len(pred) > 1:
                last_text = pred
                last_time = time.time()
            buffer = []
    else:
        if no_body > 20:
            buffer = []

    # ====== واجهة المستخدم ======
    # شريط التقدم
    progress = min(1.0, len(buffer) / FRAMES) if len(buffer) > 0 else 0
    bar_w, bar_h = int(w*0.4), 18
    bx, by = int((w-bar_w)/2), h-50
    cv2.rectangle(frame, (bx, by), (bx+bar_w, by+bar_h), (50,50,50), -1)
    cv2.rectangle(frame, (bx, by), (bx+int(progress*bar_w), by+bar_h),
                  (0, 255, 255) if progress < 1 else (0,255,0), -1)
    cv2.putText(frame, f"{int(progress*100)}%", (bx+5, by+13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # النص المترجم
    if last_text and (time.time() - last_time < 3.0):
        (tw, th), _ = cv2.getTextSize(last_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (20, 20), (20+tw+30, 20+th+30), (0,0,0), -1)
        cv2.rectangle(frame, (20, 20), (20+tw+30, 20+th+30), (0,255,0), 2)
        cv2.putText(frame, last_text, (35, 35+th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # إحصائيات
    cv2.putText(frame, f"Frames: {len(buffer)}/{FRAMES}", (20, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(frame, "Body: GREEN | Hands: CYAN", (w-300, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow('SLR - Body + Hands', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 تم الإغلاق.")