import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import os
from utils.datasets import GraphDTAPklDataset, collate_graphdta
from utils.tokenizer import tokenizer
from models.gcn import GCNGraphDTA
from protein.protein_encoder import ProteinEncoder


LR = 5e-4
BATCH = 64
EPOCHS = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TORCH_SEED = 42
torch.manual_seed(TORCH_SEED)

@torch.no_grad()
def concordance_index(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes the concordance index (C-index) for the predictions.
    Args:
        y_true (torch.Tensor): True values of shape (n_samples,).
        y_pred (torch.Tensor): Predicted values of shape (n_samples,).
        Returns:
            float: The concordance index.
    """
    true_vals = y_true.view(-1)
    pred_vals = y_pred.view(-1) 
    
    true_diff = true_vals[:, None] - true_vals[None, :]
    pred_diff = pred_vals[:, None] - pred_vals[None, :]

    comparable_mask = (true_diff != 0)

    num_comparable = comparable_mask.sum().item()
    if num_comparable == 0:
        return 0.0
    
    concordant_mask = (true_diff * pred_diff) > 0

    ci = concordant_mask.masked_select(comparable_mask).float().mean().item()
    return ci

def train_one_epoch(loader, gcn_model, prot_encoder, optimizer, device) -> float:
    """ Trains the GCN model for one epoch.
    Args:
        loader (DataLoader): DataLoader for the training data.
        gcn_model (GCNGraphDTA): The GCN model to train.
        prot_encoder (ProteinEncoder): The protein encoder model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run the training on.
        Returns:
            float: Average loss for the epoch.
    """
    gcn_model.train()
    prot_encoder.train()
    running_sum, n_samples = 0, 0
    
    for graph_batch, prot_seqs, y in loader:
        
        graph_batch = graph_batch.to(device)
        y = y.to(device)

        prot_batch = torch.stack([tokenizer(seq) for seq in prot_seqs]).to(device)

        optimizer.zero_grad()

        prot_vec = prot_encoder(prot_batch)
        pred = gcn_model(graph_batch, prot_vec)

        loss = F.mse_loss(pred,y)
        loss.backward()
        optimizer.step()

        running_sum += loss.item() * y.size(0)
        n_samples += y.size(0)
    
    return running_sum / n_samples

@torch.no_grad()
def evaluate(loader, gcn_model, prot_encoder, device):
    """ Evaluates the GCN model on the validation set.
    Args:
        loader (DataLoader): DataLoader for the validation data.
        gcn_model (GCNGraphDTA): The GCN model to evaluate.
        prot_encoder (ProteinEncoder): The protein encoder model.
        device (torch.device): Device to run the evaluation on.
        Returns:
            float: Concordance index for the validation set.
    """
    gcn_model.eval()
    prot_encoder.eval()
    
    y_true, y_pred = [], []
    
    for graph_batch, prot_seqs, y in loader:
        graph_batch = graph_batch.to(device)
        prot_batch = torch.stack([tokenizer(seq) for seq in prot_seqs]).to(device)
        y = y.to(device)

        prot_vec = prot_encoder(prot_batch)
        pred = gcn_model(graph_batch, prot_vec)
        y_pred.append(pred.cpu())
        y_true.append(y.cpu())
    
    pred = torch.cat(y_pred)
    true = torch.cat(y_true)
    mse = F.mse_loss(pred, true).item()
    ci = concordance_index(true, pred)
    return mse, ci

def main():
    
    dataSet = GraphDTAPklDataset("data/processed_davis.pkl")
    len_ds = len(dataSet)
    train_size = int(0.8 * len_ds)
    val_size = int(0.1 * len_ds)
    test_size = len_ds - train_size - val_size
    g = torch.Generator().manual_seed(TORCH_SEED)
    train_ds, val_ds, test_ds = random_split(dataSet, [train_size, val_size, test_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_graphdta)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_graphdta)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_graphdta)

    proteinEncoder = ProteinEncoder().to(device)
    gcnModel = GCNGraphDTA().to(device)

    gb, ps, yb = next(iter(train_loader))
    pb = torch.stack([tokenizer(s) for s in ps]).to(device)
    proteinEncoder.eval(); gcnModel.eval()
    with torch.no_grad():
        pv = proteinEncoder(pb)                 # [B, 128]
        pr = gcnModel(gb.to(device), pv)        # [B, 1]
    print("SMOKE:", pb.shape, pv.shape, pr.shape)

    optimizer = torch.optim.Adam(list(gcnModel.parameters()) + list(proteinEncoder.parameters()), lr=LR)
    os.makedirs("checkpoints", exist_ok=True)

    best_val = float('inf')

    for epoch in range(EPOCHS):
        train_loss_mse = train_one_epoch(train_loader, gcnModel, proteinEncoder, optimizer, device)
        validation_mse, validation_ci = evaluate(val_loader, gcnModel, proteinEncoder, device)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss_mse:.4f} | Val MSE: {validation_mse:.4f} | Val CI: {validation_ci:.4f}")
        if validation_mse < best_val:
            best_val = validation_mse
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': gcnModel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'protein_encoder_state_dict': proteinEncoder.state_dict(),
            }, "checkpoints/best_model.pth")
    
    ckpt = torch.load("checkpoints/best_model.pth", map_location=device)
    gcnModel.load_state_dict(ckpt['model_state_dict'])
    proteinEncoder.load_state_dict(ckpt['protein_encoder_state_dict'])
    test_mse, test_ci = evaluate(test_loader, gcnModel, proteinEncoder, device)
    print(f"Test MSE: {test_mse:.4f} | Test CI: {test_ci:.4f}")

if __name__ == "__main__":
    main()