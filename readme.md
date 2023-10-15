# Age, Gender, and Ethnicity Prediction Model

## Project Overview

This project aims to predict the age, gender, and ethnicity of a person given a cropped image of their face. The model is based on the Vision Transformer (ViT) Tiny model, with an additional classification head for age, gender, and ethnicity.

## Model Details

The model is based on the ViT Tiny architecture and used pretrained weights from huggingface. It was then fine-tuned using the UTKFace dataset, with a classification head added for age, gender, and ethnicity. The results of the fine-tuning process can be found in the `fig` folder.

## Features

The model predicts the following attributes given an image of a person's face:

- Age (Regression)
- Gender (Classification)
- Ethnicity (Classification)

## Instructions for Use

1. Place your data in the `data` directory.
2. Place the pretrained weights in the `pretrained` directory within the `weights` directory.
3. Checkpoints during training are stored in the `checkpoints` directory within the `weights` directory.
4. Run the `train.py` file to train the model.

## Future Work

This project is ongoing, and future updates will aim to improve the accuracy and efficiency of the model.
