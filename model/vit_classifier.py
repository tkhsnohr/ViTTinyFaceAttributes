import torch
import torch.nn as nn

from vit import ViTModel


class Classifier(nn.Module):
    def __init__(self, config) -> None:
        super(Classifier, self).__init__()

        # Initialize age prediction module
        self.age = nn.Sequential(
            # Linear layer for age prediction
            nn.Linear(config.hidden_size, 1),
            # Apply ReLU activation to prevent age from going below zero
            nn.ReLU(),
        )

        # Initialize gender prediction module
        self.gender = nn.Linear(config.hidden_size, config.gender_classes)

        # Initialize ethnicity prediction module
        self.ethnicity = nn.Linear(config.hidden_size, config.ethnicity_classes)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Extract the class token from the input
        cls_token = x[:, 0, :]

        # Predict age using the age prediction module
        age = self.age(cls_token)

        # Predict gender using the gender prediction module
        gender = self.gender(cls_token)

        # Predict ethnicity using the ethnicity prediction module
        ethnicity = self.ethnicity(cls_token)

        return age, gender, ethnicity


class ViTClassifier(nn.Module):
    def __init__(self, config) -> None:
        super(ViTClassifier, self).__init__()

        # Vision Transformer model
        self.vit = ViTModel(config)

        # Classifier module for age, gender, and ethnicity prediction
        self.classifier = Classifier(config)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pass the input through the Vision Transformer model
        x = self.vit(x)
        # Pass the output of the Vision Transformer through the classifier module
        x = self.classifier(x)

        return x
