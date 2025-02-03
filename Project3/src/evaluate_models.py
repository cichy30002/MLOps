# src/evaluate_models.py
import torch
import pytorch_lightning as pl
from src.model.LightningModule import BodyFatRegressor
from src.model.DataModule import BodyFatDataModule
import pandas as pd
import time

def load_model(checkpoint_path, model_class=BodyFatRegressor):
    model = model_class.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def evaluate_model(model, data, batch_size=32):
    """
    Evaluate the model on the validation set and return MSE.
    """
    data_module = BodyFatDataModule(data, batch_size=batch_size)
    data_module.setup()
    trainer = pl.Trainer(gpus=0, logger=False)
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
    # Load data
    data = pd.read_csv("data/bodyfat.csv")

    # Define model variants
    models = {
        "Original": "bodyfat_model.ckpt",
        "Pruned": "bodyfat_model_pruned.ckpt",
        "Layer_Pruned": "bodyfat_model_layer_pruned.ckpt",
        "Quantized": "bodyfat_model_quantized.ckpt",
    }

    # Load all models
    loaded_models = {}
    for name, path in models.items():
        if name == "Original" or name == "Layer_Pruned":
            loaded_models[name] = load_model(path)
        elif name == "Pruned":
            # For pruned model saved as state_dict
            model = BodyFatRegressor.load_from_checkpoint('bodyfat_model.ckpt')
            model.load_state_dict(torch.load(path))
            loaded_models[name] = model
        elif name == "Quantized":
            # Load quantized model
            model = BodyFatRegressor.load_from_checkpoint('bodyfat_model.ckpt')
            model.load_state_dict(torch.load(path))
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            loaded_models[name] = model

    # Evaluate all models
    results = {}
    for name, model in loaded_models.items():
        mse = evaluate_model(model, data, batch_size=32)
        inference_time = measure_inference_time(model, data, batch_size=32, num_batches=10)
        results[name] = {
            "MSE": mse,
            "Avg_Inference_Time_sec": inference_time
        }

    # Print results
    for name, metrics in results.items():
        print(f"{name} Model - MSE: {metrics['MSE']:.4f}, Avg Inference Time per Batch: {metrics['Avg_Inference_Time_sec']:.6f} seconds")

if __name__ == "__main__":
    main()
