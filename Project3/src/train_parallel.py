# src/train_parallel.py
import joblib
import optuna
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.model.DataModule import BodyFatDataModule
from src.model.LightningModule import BodyFatRegressor

data = pd.read_csv("data/bodyfat.csv")

wandb_logger = WandbLogger(project="bodyfat_regression")

class OptunaPruningCallback(Callback):
    def __init__(self, trial, monitor):
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
        self.trial.report(current_score, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial was pruned at epoch {trainer.current_epoch}"
            )

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    hidden_layer_1 = trial.suggest_int("hidden_layer_1", 32, 128, step=32)
    hidden_layer_2 = trial.suggest_int("hidden_layer_2", 32, 128, step=32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    model = BodyFatRegressor(
        input_dim=data.shape[1] - 1,
        learning_rate=learning_rate,
        hidden_layer_1=hidden_layer_1,
        hidden_layer_2=hidden_layer_2,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
    )

    data_module = BodyFatDataModule(data, batch_size=batch_size)

    # Add ModelCheckpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=50,
        callbacks=[OptunaPruningCallback(trial, monitor="val_loss"), checkpoint_callback],
        enable_checkpointing=True,
    )
    trainer.fit(model, data_module)
    return trainer.callback_metrics["val_loss"].item()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20, n_jobs=2)

print(f"Best hyperparameters: {study.best_params}")

best_model_path = study.best_trial.user_attrs.get("checkpoint_path")
if best_model_path is None:
    best_model_path = "checkpoints/best-checkpoint.ckpt"

best_model = BodyFatRegressor.load_from_checkpoint(best_model_path)
data_module = BodyFatDataModule(data, batch_size=32)
trainer = pl.Trainer(logger=wandb_logger, max_epochs=50)
trainer.fit(best_model, data_module)

# Save the best model state_dict
trainer.save_checkpoint('bodyfat_model.ckpt')
