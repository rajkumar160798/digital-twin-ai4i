import torch
from torch.utils.data import DataLoader

def train_gru(model, dataset, epochs=10, batch_size=32, lr=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq, target in dataloader:
            optimizer.zero_grad()
            pred = model(seq)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")
    print("Training complete.")