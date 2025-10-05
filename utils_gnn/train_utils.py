import math
import time
import torch
import wandb
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import os


def mape_criterion(inputs, targets):
    eps = 1e-5
    return 100 * torch.mean(torch.abs(targets - inputs) / (targets + eps))


def mse_criterion(inputs, targets):
    """Simple MSE loss for speedup prediction"""
    return torch.nn.functional.mse_loss(inputs, targets)


def log_mse_loss(predictions, targets, eps=1e-8):
    """
    MSE loss in log space to handle wide-range speedup values.
    Both predictions and targets should be raw speedup values.
    """
    # Ensure positive values before taking log
    safe_predictions = torch.clamp(predictions, min=eps)
    safe_targets = torch.clamp(targets, min=eps)
    
    # Apply log transformation
    log_predictions = torch.log(safe_predictions)
    log_targets = torch.log(safe_targets)
    
    # MSE in log space
    return torch.nn.functional.mse_loss(log_predictions, log_targets)


def smape_metric(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return 100.0 * torch.abs(tgt - pred) / (torch.abs(tgt) + torch.abs(pred) + eps)

def safe_mape_metric(pred: torch.Tensor, tgt: torch.Tensor, floor: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    denom = torch.clamp(torch.abs(tgt), min=floor) + eps
    return 100.0 * torch.abs(tgt - pred) / denom

####################################################
# Main GNN training function
####################################################
def train_gnn_model(
    config,
    model,
    criterion,
    optimizer,
    max_lr,
    dataloader_dict,
    num_epochs,
    log_every,
    logger,
    train_device,
    validation_device,
):
    """
    Trains a PyG-based GNN model using a train/val dataloader.
    """
    since = time.time()
    best_loss = math.inf
    best_model = None
    
    # Early stopping parameters
    patience = 50  # Stop if no improvement for 50 epochs
    patience_counter = 0
    min_delta = 1e-4  # Minimum change to qualify as improvement for MSE loss
    stop_training = False  # Flag to control early stopping

    # Checkpoint paths
    ckpt_dir = os.path.join(config.experiment.base_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(ckpt_dir, f"{config.experiment.name}_latest.pt")

    dataloader_size = {}
    for phase in ["train", "val"]:
        total_samples = 0
        for batch in dataloader_dict[phase]:
            # Handle (batch_data, attrs) tuple from collate_with_attrs
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                batch_data, attrs = batch
                total_samples += batch_data.num_graphs
            else:
                total_samples += batch.num_graphs
        dataloader_size[phase] = total_samples

    # Baseline MAPE with const=1.0
    try:
        baseline_mapes = {}
        for phase in ["train", "val"]:
            baseline_mape = 0.0
            num_samples = 0
            for batch in dataloader_dict[phase]:
                # Handle (batch_data, attrs) tuple from collate_with_attrs
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    batch_data, attrs = batch
                    batch = batch_data.to(train_device if phase == "train" else validation_device)
                else:
                    batch = batch.to(train_device if phase == "train" else validation_device)
                labels = batch.y
                const_pred = torch.ones_like(labels)
                batch_mape = safe_mape_metric(const_pred, labels).mean()
                baseline_mape += batch_mape.item() * batch.num_graphs
                num_samples += batch.num_graphs
            baseline_mapes[phase] = baseline_mape / num_samples
        print(f"Baseline MAPE (const=1.0): Train={baseline_mapes['train']:.2f}%, Val={baseline_mapes['val']:.2f}%")
    except Exception as e:
        print(f"[WARNING] Could not compute baseline MAPE: {e}")
        baseline_mapes = {"train": 0.0, "val": 0.0}

    scheduler = OneCycleLR(
        optimizer, max_lr=max_lr, steps_per_epoch=len(dataloader_dict["train"]), epochs=num_epochs
    )

    # Load checkpoint if it exists
    start_epoch = 0
    if config.training.continue_training and os.path.exists(latest_ckpt_path):
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint.get("best_loss", math.inf)
            best_mape_safe = checkpoint.get("best_mape_safe", math.inf)
            print(f"[INFO] Resumed from epoch {start_epoch}, best_loss: {best_loss:.4f}, best_mape_safe: {best_mape_safe:.4f}%")
        except Exception as e:
            print(f"[WARNING] Could not load checkpoint: {e}")
            start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                device = train_device
            else:
                model.eval()
                device = validation_device

            model = model.to(device)
            model.device = device

            running_loss = 0.0
            num_samples_processed = 0
            smape_sum = 0.0
            mape_safe_sum = 0.0

            pbar = tqdm(dataloader_dict[phase], desc=f"{phase} Epoch {epoch+1}")

            for batch in pbar:
                # Handle (batch_data, attrs) tuple from collate_with_attrs
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    batch_data, attrs = batch
                    batch = batch_data.to(device)
                else:
                    batch = batch.to(device)
                
                labels = batch.y
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(batch)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # Calculate metrics
                batch_smape = smape_metric(outputs, labels).mean()
                batch_mape_safe = safe_mape_metric(outputs, labels).mean()

                running_loss += loss.item() * batch.num_graphs
                smape_sum += batch_smape.item() * batch.num_graphs
                mape_safe_sum += batch_mape_safe.item() * batch.num_graphs
                num_samples_processed += batch.num_graphs

                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'SMAPE': f'{batch_smape.item():.2f}%',
                    'MAPE_safe': f'{batch_mape_safe.item():.2f}%'
                })

            # Calculate epoch metrics
            epoch_loss = running_loss / num_samples_processed
            epoch_smape = smape_sum / num_samples_processed
            epoch_mape_safe = mape_safe_sum / num_samples_processed

            if phase == "train":
                train_loss = epoch_loss
                train_smape = epoch_smape
                train_mape_safe = epoch_mape_safe
            else:
                val_loss = epoch_loss
                val_smape = epoch_smape
                val_mape_safe = epoch_mape_safe

        # Early stopping logic based on validation loss
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
            
            # Save best model
            try:
                best_model_path = os.path.join(ckpt_dir, f"{config.experiment.name}_best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state": best_model,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "val_loss": val_loss,
                    "val_smape": val_smape,
                    "val_mape_safe": val_mape_safe,
                }, best_model_path)
            except Exception as e:
                print(f"[Checkpoint] Failed to save best model: {e}")
            
            print(f"[INFO] New best model saved with val_loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered after {epoch+1} epochs")
                print(f"[INFO] Best validation loss: {best_loss:.4f}")
                stop_training = True
                break

        # Log to wandb
        if config.wandb.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_smape": train_smape,
                "val_smape": val_smape,
                "train_mape_safe": train_mape_safe,
                "val_mape_safe": val_mape_safe,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch,
            })

        epoch_time = time.time() - epoch_start_time
        print(
            f"Epoch {epoch+1}/{num_epochs} => Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Loss: {best_loss:.4f}, "
            f"SMAPE (T/V): {train_smape:.2f}/{val_smape:.2f}, MAPE_safe (T/V): {train_mape_safe:.2f}/{val_mape_safe:.2f}, Time: {epoch_time:.2f}s"
        )
        if epoch % log_every == 0:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} => Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Loss: {best_loss:.4f}, "
                f"SMAPE (T/V): {train_smape:.2f}/{val_smape:.2f}, MAPE_safe (T/V): {train_mape_safe:.2f}/{val_mape_safe:.2f}, Time: {epoch_time:.2f}s"
            )

        # Check for early stopping
        if stop_training:
            break

        # Save latest checkpoint at epoch end
        try:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_loss": best_loss,
            }, latest_ckpt_path)
        except Exception as e:
            print(f"[Checkpoint] Failed to save latest checkpoint: {e}")

    # Done
    total_time = time.time() - since
    print(f"Training complete in {total_time//60:.0f}m {total_time%60:.0f}s, best val loss: {best_loss:.4f}")
    logger.info(f"Training done in {total_time//60:.0f}m {total_time%60:.0f}s, best val loss: {best_loss:.4f}")

    return best_loss, best_model