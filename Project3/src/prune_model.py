# src/prune_model.py
import torch
import torch.nn.utils.prune as prune
import pytorch_lightning as pl
from src.model.LightningModule import BodyFatRegressor
from src.model.DataModule import BodyFatDataModule
import pandas as pd
import time

def load_model(checkpoint_path='bodyfat_model.ckpt'):
    model = BodyFatRegressor.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def apply_weight_pruning(model, pruning_amount=0.2):
    """
    Apply L1 unstructured pruning to all linear layers in the model.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
            prune.remove(module, 'weight')
    return model

def save_pruned_model(model, path='bodyfat_model_pruned.ckpt'):
    torch.save(model.state_dict(), path)

def evaluate_model(model, data, batch_size=32):
    """
    Evaluate the model on the validation set and return MSE.
    """
    data_module = BodyFatDataModule(data, batch_size=batch_size)
    data_module.setup()
    trainer = pl.Trainer(logger=False)
    results = trainer.validate(model, datamodule=data_module)
    mse = results[0]['val_loss']
    return mse

def measure_inference_time(model, data, batch_size=32, num_batches=10):
    """
    Measure the average inference time over a number of batches.
    """
    data_module = BodyFatDataModule(data, batch_size=batch_size)
    data_module.setup()
    val_loader = data_module.val_dataloader()
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_batches:
                break
            features = batch['features'].to(device)
            _ = model(features)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_batches
    return avg_time

def main():
    data = pd.read_csv("data/bodyfat.csv")

    model = load_model('bodyfat_model.ckpt')

    original_mse = evaluate_model(model, data)
    original_inference_time = measure_inference_time(model, data)
    print(f"Original Model - MSE: {original_mse:.4f}, Avg Inference Time per Batch: {original_inference_time:.6f} seconds")

    pruned_model = apply_weight_pruning(model, pruning_amount=0.2)

    pruned_mse = evaluate_model(pruned_model, data)
    pruned_inference_time = measure_inference_time(pruned_model, data)
    print(f"Pruned Model - MSE: {pruned_mse:.4f}, Avg Inference Time per Batch: {pruned_inference_time:.6f} seconds")

    save_pruned_model(pruned_model, 'bodyfat_model_pruned.ckpt')

if __name__ == "__main__":
    main()
