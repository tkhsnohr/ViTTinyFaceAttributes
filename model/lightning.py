import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import Accuracy

import lightning as L
from vit_classifier import ViTClassifier
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import Accuracy
import lightning as L
from vit_classifier import ViTClassifier


class FaceRecognitionModel(L.LightningModule):
    def __init__(self, config) -> None:
        super(FaceRecognitionModel, self).__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Initialize the ViTClassifier model
        self.model = ViTClassifier(config)

        # Define loss functions
        self.l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()

        # Initialize accuracy metrics for validation
        self.val_gender_accuracy = Accuracy(
            task="multiclass",
            num_classes=config.gender_classes,
        )
        self.val_ethnicity_accuracy = Accuracy(
            task="multiclass",
            num_classes=config.ethnicity_classes,
        )

    def forward(self, x):
        # Forward pass through the model
        return self.model(x)

    def training_step(self, batch, batch_idx) -> None:
        # Unpack the batch
        x, age, gender, ethnicity = batch

        # Perform forward pass through the model
        age_logits, gender_logits, ethnicity_logits = self.model(x)

        # Calculate losses
        age_loss = self.l1(age_logits, age)
        gender_loss = self.cross_entropy(gender_logits, gender)
        ethnicity_loss = self.cross_entropy(ethnicity_logits, ethnicity)

        # Calculate total loss
        loss = age_loss + gender_loss + ethnicity_loss

        # Log training losses
        self.log_dict(
            {
                "train_age_loss": age_loss.item(),
                "train_gender_loss": gender_loss.item(),
                "train_ethnicity_loss": ethnicity_loss.item(),
                "train_loss": loss.item(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        # Unpack the batch
        x, age, gender, ethnicity = batch

        # Perform forward pass through the model
        age_logits, gender_logits, ethnicity_logits = self.model(x)
        age_logits = self.model(x)

        # Calculate losses
        age_loss = self.l1(age_logits, age)
        gender_loss = self.cross_entropy(gender_logits, gender)
        ethnicity_loss = self.cross_entropy(ethnicity_logits, ethnicity)

        # Calculate total loss
        loss = age_loss + gender_loss + ethnicity_loss

        # Calculate predicted gender and ethnicity
        gender_preds = torch.argmax(gender_logits, dim=1)
        ethnicity_preds = torch.argmax(ethnicity_logits, dim=1)

        # Update accuracy metrics
        self.val_gender_accuracy.update(gender_preds, gender)
        self.val_ethnicity_accuracy.update(ethnicity_preds, ethnicity)

        # Log validation losses
        self.log_dict(
            {
                "val_age_loss": age_loss.item(),
                "val_gender_loss": gender_loss.item(),
                "val_ethnicity_loss": ethnicity_loss.item(),
                "val_loss": loss.item(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Log validation accuracies
        self.log_dict(
            {
                "val_gender_accuracy": self.val_gender_accuracy.compute().item(),
                "val_ethnicity_accuracy": self.val_ethnicity_accuracy.compute().item(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_validation_epoch_end(self) -> None:
        # Reset accuracy metrics at the end of each validation epoch
        self.val_gender_accuracy.reset()
        self.val_ethnicity_accuracy.reset()

    def configure_optimizers(self) -> None:
        # Configure optimizer with different learning rates for ViT and classifier parameters
        optimizer = AdamW(
            [
                {"params": self.model.vit.parameters(), "lr": 1e-3},
                {"params": self.model.classifier.parameters(), "lr": 1e-2},
            ]
        )

        # Configure learning rate scheduler
        scheduler = LambdaLR(
            optimizer,
            lambda epoch: 0.95**epoch,
            verbose=True,
        )

        return [optimizer], [scheduler]
