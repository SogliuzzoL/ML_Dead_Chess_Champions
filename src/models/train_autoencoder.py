import numpy as np
import torch
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.config import AUTOENCODER_MODEL_PATH, logger
from models.autoencoder import Autoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


def train_autoencoder(
    train_dataset,
    input_dim,
    latent_dim=128,
    num_epochs=10,
    batch_size=1024,
    learning_rate=1e-3,
):
    model = Autoencoder(input_dim, latent_dim).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            batch = batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model


def infer_autoencoder(input_path: str, output_path: str):
    data = np.load(input_path, mmap_mode="r")
    input_dim = data.shape[1]

    model = Autoencoder(input_dim, latent_dim=128).to(DEVICE)
    model.load_state_dict(torch.load(AUTOENCODER_MODEL_PATH, map_location=DEVICE))
    model.eval()

    infer_dataset = ChessDataset(data)
    infer_loader = DataLoader(
        infer_dataset, batch_size=1024, shuffle=False, num_workers=4
    )

    encoded_vectors = []
    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="Encoding Test Vectors"):
            batch = batch.to(DEVICE, non_blocking=True)
            latent = model.encode(batch)
            encoded_vectors.append(latent.cpu().numpy())

    result = np.concatenate(encoded_vectors, axis=0)
    np.save(output_path, result)


def run_autoencoder(input_path: str, output_path: str):
    data = np.load(input_path, mmap_mode="r")
    input_dim = data.shape[1]

    train_dataset = ChessDataset(data)

    model = train_autoencoder(train_dataset, input_dim)
    torch.save(model.state_dict(), AUTOENCODER_MODEL_PATH)

    infer_autoencoder(input_path, output_path)
