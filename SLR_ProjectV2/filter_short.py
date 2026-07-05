# filter_short.py - استخراج الجمل القصيرة (3-4 كلمات)
import numpy as np
import re

print("📂 Loading data...")

# 1. تحميل البيانات الحالية
X = np.load("extracted_5000/X.npy")
y = np.load("extracted_5000/y.npy", allow_pickle=True)

print(f"🔍 Total samples: {len(X)}")

# 2. فلترة الجمل اللي فيها 3 أو 4 كلمات
new_X = []
new_y = []

for i, sent in enumerate(y):
    # تنظيف النص من علامات الترقيم
    clean_sent = re.sub(r'[^\w\s]', '', str(sent).lower())
    words = clean_sent.split()
    
    # نحتفظ بالجمل اللي فيها 3 أو 4 كلمات بالضبط
    if len(words) >= 3 and len(words) <= 4:
        new_X.append(X[i])
        new_y.append(sent)

# 3. تحويلهم لمصفوفات
new_X = np.array(new_X)
new_y = np.array(new_y)

print(f"✅ Short sentences found: {len(new_X)}")

# 4. حفظهم في ملفات جديدة
np.save("extracted_5000/X_short.npy", new_X)
np.save("extracted_5000/y_short.npy", new_y)

print("💾 Saved to: extracted_5000/X_short.npy and y_short.npy")

# 5. طباعة 10 أمثلة عشان تتأكدي
print("\n📝 Examples:")
for i in range(min(10, len(new_y))):
    print(f"  {i+1}. {new_y[i]}")