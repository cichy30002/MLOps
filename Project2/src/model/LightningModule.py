import pytorch_lightning as pl
import torch
from torch import nn


class BodyFatRegressor(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        learning_rate=1e-3,
        hidden_layer_1=64,
        hidden_layer_2=32,
        dropout_rate=0.2,
        activation_function=nn.ReLU,
        weight_decay=1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_1),
            activation_function(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_1, hidden_layer_2),
            activation_function(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_2, 1),
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, target = batch["features"], batch["target"]
        predictions = self(features).squeeze(1)
        loss = self.criterion(predictions, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, target = batch["features"], batch["target"]
        predictions = self(features).squeeze(1)
        loss = self.criterion(predictions, target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
