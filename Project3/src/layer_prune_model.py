# src/layer_prune_model.py
import torch
import pytorch_lightning as pl
from src.model.LightningModule import BodyFatRegressor
from src.model.DataModule import BodyFatDataModule
import pandas as pd
import time


def load_model(checkpoint_path='bodyfat_model.ckpt'):
    model = BodyFatRegressor.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def reduce_layer_size(model, reduction_factor=0.5):
    """
    Reduce the number of neurons in hidden layers by a reduction factor.
    """
    hidden_layer_1 = model.hparams.hidden_layer_1
    hidden_layer_2 = model.hparams.hidden_layer_2

    new_hidden_layer_1 = max(1, int(hidden_layer_1 * reduction_factor))
    new_hidden_layer_2 = max(1, int(hidden_layer_2 * reduction_factor))

    print(f"Reducing hidden_layer_1 from {hidden_layer_1} to {new_hidden_layer_1}")
    print(f"Reducing hidden_layer_2 from {hidden_layer_2} to {new_hidden_layer_2}")

    new_model = BodyFatRegressor(
        input_dim=model.hparams.input_dim,
        learning_rate=model.hparams.learning_rate,
        hidden_layer_1=new_hidden_layer_1,
        hidden_layer_2=new_hidden_layer_2,
        dropout_rate=model.hparams.dropout_rate,
        weight_decay=model.hparams.weight_decay,
    )

    # Transfer weights where possible
    with torch.no_grad():
        # Copy weights and biases for the first layer
        new_model.model[0].weight[:new_hidden_layer_1, :] = model.model[0].weight[:new_hidden_layer_1, :]
        new_model.model[0].bias[:new_hidden_layer_1] = model.model[0].bias[:new_hidden_layer_1]

        # Copy weights and biases up to the new hidden_layer_2 and new_hidden_layer_1
        new_model.model[3].weight[:new_hidden_layer_2, :new_hidden_layer_1] = model.model[3].weight[:new_hidden_layer_2, :new_hidden_layer_1]
        new_model.model[3].bias[:new_hidden_layer_2] = model.model[3].bias[:new_hidden_layer_2]

        # Copy weights up to the new hidden_layer_2
        new_model.model[6].weight[:, :new_hidden_layer_2] = model.model[6].weight[:, :new_hidden_layer_2]
        new_model.model[6].bias = model.model[6].bias

    return new_model


def save_layer_pruned_model(model, path='bodyfat_model_layer_pruned.ckpt'):
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

    pruned_model = reduce_layer_size(model, reduction_factor=0.5)

    pruned_mse = evaluate_model(pruned_model, data)
    pruned_inference_time = measure_inference_time(pruned_model, data)
    print(f"Layer-Pruned Model - MSE: {pruned_mse:.4f}, Avg Inference Time per Batch: {pruned_inference_time:.6f} seconds")

    save_layer_pruned_model(pruned_model, 'bodyfat_model_layer_pruned.ckpt')


if __name__ == "__main__":
    main()
