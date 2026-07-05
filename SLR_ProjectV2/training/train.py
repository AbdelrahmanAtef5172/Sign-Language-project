# training/train.py - نسخة 80% (نموذج 128 مع Dropout 0.1)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils.data_loader import SLR_Dataset
from models.signformer_light import SignformerLight
import matplotlib.pyplot as plt

BATCH_SIZE = 8
EPOCHS = 512
LEARNING_RATE = 0.0001
DATA_PATH = "extracted_5000"
PATIENCE = 30
MIN_DELTA = 0.0005

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Running on: {device}")

dataset = SLR_Dataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# النموذج الجديد (128, 4, 2)
model = SignformerLight(
    input_dim=75,
    vocab_size=dataset.vocab_size,
    d_model=128,
    nhead=4,
    num_layers=2
).to(device)

print(f"🧠 Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

train_losses = []
best_loss = float('inf')
patience_counter = 0
best_epoch = 0

print("🔥 Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        tgt_input = y[:, :-1]
        tgt_target = y[:, 1:]
        
        optimizer.zero_grad()
        output = model(x, tgt_input)
        loss = criterion(output.reshape(-1, dataset.vocab_size), tgt_target.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    train_losses.append(avg_loss)
    scheduler.step(avg_loss)
    
    if avg_loss < best_loss - MIN_DELTA:
        best_loss = avg_loss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), "checkpoints/model_best.pth")
        print(f"📉 Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f} ✅ (Best)")
    else:
        patience_counter += 1
        print(f"📉 Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f} (Patience: {patience_counter}/{PATIENCE})")
    
    if patience_counter >= PATIENCE:
        print(f"\n🛑 Early Stopping at Epoch {epoch+1}")
        break

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/model_short.pth")
print(f"\n💾 Model saved.")

# رسم Loss
plt.figure(figsize=(10,6))
plt.plot(train_losses, label='Loss')
plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best Epoch {best_epoch}')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss')
plt.legend(); plt.grid(True)
plt.savefig('checkpoints/loss_graph.png', dpi=150)
print("📊 Graph saved to checkpoints/loss_graph.png")