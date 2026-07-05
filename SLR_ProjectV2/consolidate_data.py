# consolidate_data.py - نسخة بتتجاهل العينات الفارغة
import os
import json
import numpy as np
import csv
import glob

BASE_FOLDER = "extracted_5000/extracted_5000"
CSV_PATH = "extracted_5000/how2sign_realigned_train.csv"
OUTPUT_X = "extracted_5000/X.npy"
OUTPUT_Y = "extracted_5000/y.npy"

print("📂 Reading CSV...")
video_data = []

with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        if len(row) >= 7:
            video_data.append((row[3], row[6]))  # folder_name, sentence

print(f"✅ Found {len(video_data)} rows.")

all_keypoints = []
all_sentences = []
missing = 0
found = 0

for folder_name, sentence in video_data:
    folder_path = os.path.join(BASE_FOLDER, folder_name)
    if os.path.exists(folder_path):
        json_files = sorted(glob.glob(os.path.join(folder_path, "*_keypoints.json")))
        if json_files:
            kp_list = []
            for jf in json_files:
                with open(jf, 'r') as f:
                    data = json.load(f)
                    kp = data.get('people', [{}])[0].get('pose_keypoints_2d', [])
                    if not kp:
                        kp = data.get('pose_keypoints_2d', [])
                    kp_list.extend(kp)
            if kp_list:
                arr = np.array(kp_list[:75*32]).reshape(-1, 75)
                if arr.shape[0] < 32:
                    pad = np.zeros((32 - arr.shape[0], 75))
                    arr = np.vstack([arr, pad])
                else:
                    arr = arr[:32]
                all_keypoints.append(arr)
                all_sentences.append(sentence)
                found += 1
                print(f"✅ Found: {found}", end='\r')
                continue
    missing += 1

print(f"\n📊 ✅ Found: {found} | ❌ Missing/Skipped: {missing}")

X = np.array(all_keypoints)
y = np.array(all_sentences)
print(f"✅ X shape: {X.shape}")

np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)
print("🎉 Done! Now run training on clean data.")