"""
train.py
--------
Training loop with early stopping and model checkpointing for the chest
X-ray pneumonia classification project.

Functions
---------
train(model, train_loader, val_loader, criterion, optimiser, device,
      scheduler, num_epochs, patience, save_path)
    Train a model and return the loss/accuracy history.
"""

import torch


# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------

def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimiser,
    device,
    scheduler,
    num_epochs=10,
    patience=5,
    save_path=None
):
    """
    Train a model with early stopping and optional checkpoint saving.

    The best model (lowest validation loss) is saved to disk if save_path
    is provided. Training stops early if validation loss does not improve
    for a given number of consecutive epochs.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    train_loader : DataLoader
        DataLoader for the training set.
    val_loader : DataLoader
        DataLoader for the validation set.
    criterion : torch.nn.Module
        Loss function (e.g. BCEWithLogitsLoss).
    optimiser : torch.optim.Optimizer
        Optimiser instance (e.g. AdamW).
    device : torch.device
        Device to run training on (CPU or CUDA).
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler (e.g. ReduceLROnPlateau).
    num_epochs : int, optional
        Maximum number of training epochs. Default is 10.
    patience : int, optional
        Number of epochs without improvement before early stopping.
        Default is 5.
    save_path : str, optional
        File path to save the best model checkpoint (.pth). If None,
        no checkpoint is saved.

    Returns
    -------
    history : dict
        Dictionary containing per-epoch lists:
        'train_loss', 'val_loss', 'train_acc', 'val_acc'.
    """
    history = {
        'train_loss': [],
        'val_loss':   [],
        'train_acc':  [],
        'val_acc':    []
    }

    best_val_loss   = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):

        # ------------------------------------------------------------------
        # Training Phase
        # ------------------------------------------------------------------
        model.train()
        train_loss    = 0.0
        train_correct = 0

        for batch_imgs, batch_labels in train_loader:
            batch_imgs   = batch_imgs.to(device)
            batch_labels = batch_labels.float().unsqueeze(1).to(device)

            optimiser.zero_grad()
            outputs = model(batch_imgs)
            loss    = criterion(outputs, batch_labels)
            loss.backward()
            optimiser.step()

            train_loss    += loss.item() * batch_imgs.size(0)
            predicted      = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (predicted == batch_labels).sum().item()

        train_loss_avg = train_loss    / len(train_loader.dataset)
        train_acc_avg  = train_correct / len(train_loader.dataset)

        # ------------------------------------------------------------------
        # Validation Phase
        # ------------------------------------------------------------------
        model.eval()
        val_loss    = 0.0
        val_correct = 0

        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs   = batch_imgs.to(device)
                batch_labels = batch_labels.float().unsqueeze(1).to(device)

                outputs   = model(batch_imgs)
                loss      = criterion(outputs, batch_labels)
                val_loss += loss.item() * batch_imgs.size(0)

                predicted   = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (predicted == batch_labels).sum().item()

        val_loss_avg = val_loss    / len(val_loader.dataset)
        val_acc_avg  = val_correct / len(val_loader.dataset)

        # ------------------------------------------------------------------
        # Logging and Scheduling
        # ------------------------------------------------------------------
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc_avg)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc_avg)

        scheduler.step(val_loss_avg)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss_avg:.4f} | "
            f"Val Loss: {val_loss_avg:.4f}"
        )

        # ------------------------------------------------------------------
        # Early Stopping and Checkpointing
        # ------------------------------------------------------------------
        if val_loss_avg < best_val_loss:
            best_val_loss     = val_loss_avg
            epochs_no_improve = 0

            if save_path:
                torch.save(
                    {
                        'epoch':                epoch + 1,
                        'model_state_dict':     model.state_dict(),
                        'optimizer_state_dict': optimiser.state_dict(),
                        'val_loss':             val_loss_avg,
                    },
                    save_path
                )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return history