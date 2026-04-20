import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train (model, train_loader, val_loader, criterion, optimiser, device, scheduler, num_epochs=10, patience=5, save_path=None):
    history = {'train_loss': [], 'val_loss': [],
               'val_acc': [], 'train_acc': []}
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for batch_imgs, batch_labels in train_loader:
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            batch_labels = batch_labels.float().unsqueeze(1)

            optimiser.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimiser.step()

            train_loss += loss.item() * batch_imgs.size(0)
            probs = torch.sigmoid(outputs)
            predicted = (probs >= 0.5).float()
            train_correct += (predicted == batch_labels).sum().item()
        train_loss_avg = train_loss / len(train_loader.dataset)
        train_acc_avg = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
                batch_labels = batch_labels.float().unsqueeze(1)

                outputs = model(batch_imgs)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item() * batch_imgs.size(0)

                probs = torch.sigmoid(outputs)
                predicted = (probs >= 0.5).float()
                val_correct += (predicted == batch_labels).sum().item()

        val_loss_avg = val_loss / len(val_loader.dataset)
        val_acc_avg = val_correct / len(val_loader.dataset)

        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc_avg)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc_avg)

        scheduler.step(val_loss_avg)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss_avg:.4f} | "
            f"Val Loss: {val_loss_avg:.4f}")
        
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_improve = 0
            
            if save_path:
                torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimiser.state_dict(),
                        'val_loss': val_loss_avg
                    }, save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


    return history
