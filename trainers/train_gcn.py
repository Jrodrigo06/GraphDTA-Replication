import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from utils.datasets import GraphDTAPklDataset, collate_graphdta
from utils.tokenizer import tokenizer
from models.gcn import GCNGraphDTA
from protein.protein_encoder import ProteinEncoder


LR = 5e-4
BATCH = 512
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
