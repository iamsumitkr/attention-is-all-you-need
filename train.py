import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import Transformer   

# ------------------- Hyperparams -------------------
SRC_VOCAB_SIZE = 1000     # dummy vocab size
TRG_VOCAB_SIZE = 1000
EMBED_SIZE     = 128
HEADS          = 4
FF_DIM         = 512
NUM_LAYERS     = 2
MAX_SEQ_LEN    = 20
BATCH_SIZE     = 4
EPOCHS         = 10
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- Model -------------------
model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    trg_vocab_size=TRG_VOCAB_SIZE,
    src_pad_idx=0,
    trg_pad_idx=0,
    embed_size=EMBED_SIZE,
    num_layers=NUM_LAYERS,
    forward_expansion=4,
    heads=HEADS,
    dropout=0.1,
    device=DEVICE,
    max_length=MAX_SEQ_LEN
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=0)  
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

# ------------------- Dummy Data -------------------
src = torch.randint(1, SRC_VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN)).to(DEVICE)
trg = torch.randint(1, TRG_VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN)).to(DEVICE)

# ------------------- Training Loop -------------------
print("🚀 Starting dummy training...")
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    # decoder input = trg[:, :-1], labels = trg[:, 1:]
    output = model(src, trg[:, :-1])
    
    output = output.reshape(-1, TRG_VOCAB_SIZE)
    target = trg[:, 1:].reshape(-1)
    
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

print("\n✅ Dummy training finished successfully! (Loss should decrease)")

# OUTPUT OF THE TRAINING

'''
🚀 Starting dummy training...
Epoch [1/10], Loss: 7.1196
Epoch [2/10], Loss: 6.8676
Epoch [3/10], Loss: 6.5427
Epoch [4/10], Loss: 6.2093
Epoch [5/10], Loss: 5.9883
Epoch [6/10], Loss: 5.8253
Epoch [7/10], Loss: 5.6176
Epoch [8/10], Loss: 5.3530
Epoch [9/10], Loss: 5.2584
Epoch [10/10], Loss: 5.0137

✅ Dummy training finished successfully! (Loss should decrease)

'''
