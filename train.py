import os
import json
from dotenv import load_dotenv
from types import SimpleNamespace

import wandb
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from model.lightning import FaceRecognitionModel
from dataset.utk_datamodule import FaceDataModule
from utils.dir import data_dir, checkpoint_dir, config_file

load_dotenv()
seed_everything(103)

# Load config from file
with open(config_file, "r") as f:
    config = SimpleNamespace(**json.load(f))

# Set W&B API key and login
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
wandb.login()

# Define callbacks and logger
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=checkpoint_dir,
    filename="{epoch}-{val_loss:.4f}",
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=3,
    verbose=False,
    mode="min",
)

wandb_logger = WandbLogger(
    project="Face Recognition",
    name="ViT Tiny 1",
    log_model="all",
)

# Create trainer instance
trainer = Trainer(
    max_epochs=config.max_epochs,
    check_val_every_n_epoch=config.check_val_every_n_epoch,
    gradient_clip_val=config.gradient_clip_val,
    precision=config.precision,
    num_sanity_val_steps=config.num_sanity_val_steps,
    log_every_n_steps=config.log_every_n_steps,
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=wandb_logger,
)

if __name__ == "__main__":
    # Create model and data module instances
    # Add code to load pretrained weights from the 'weights/pretrained' directory
    model = FaceRecognitionModel(config)
    datamodule = FaceDataModule(data_dir, config)

    # Train the model
    trainer.fit(model, datamodule=datamodule)
