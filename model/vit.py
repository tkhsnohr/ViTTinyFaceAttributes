import torch
from torch import nn

from types import SimpleNamespace
from einops import rearrange, repeat


class PatchEmbeddings(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super(PatchEmbeddings, self).__init__()

        # Projection layer to convert image patches to hidden size
        self.projection = nn.Conv2d(
            in_channels=config.image_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.encoder_stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the projection layer to the input tensor
        x = self.projection(x)

        return x


class Embeddings(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super(Embeddings, self).__init__()

        # Calculate the number of positions based on image size and encoder stride
        self.positions = (config.image_size // config.encoder_stride) ** 2 + 1

        # Learnable parameter for the classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Learnable parameter for position embeddings
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.positions, config.hidden_size)
        )

        # Patch embeddings module
        self.patch_embeddings = PatchEmbeddings(config)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create classification tokens and repeat them for each input sample
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.shape[0])

        # Apply patch embeddings to input image
        x = self.patch_embeddings(x)  # b, c, h, w -> b, c', h/p, w/p

        # Rearrange the dimensions of the tensor for self-attention
        x = rearrange(x, "b c h w -> b (h w) c")

        # Concatenate classification tokens with patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embeddings to the input tensor
        x += self.position_embeddings

        # Apply dropout for regularization
        x = self.dropout(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super(SelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = config.hidden_size**-0.5

        # Linear layers for query, key, and value projections
        self.query = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.qkv_bias,
        )
        self.key = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.qkv_bias,
        )
        self.value = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.qkv_bias,
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input tensor to query, key, and value tensors
        query = rearrange(
            self.query(x),
            "b n (h d) -> b h n d",
            h=self.num_attention_heads,
        )
        key = rearrange(
            self.key(x),
            "b n (h d) -> b h n d",
            h=self.num_attention_heads,
        )
        value = rearrange(
            self.value(x),
            "b n (h d) -> b h n d",
            h=self.num_attention_heads,
        )

        # Compute attention scores using matrix multiplication (n = m)
        attention = torch.einsum("bhnd,bhdm->bhnm", query, key.transpose(-1, -2))
        attention = attention * self.scale

        # Apply softmax activation and dropout for attention scores
        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        # Compute weighted sum of value tensors using attention scores (n = m)
        x = torch.einsum("bhnm,bhmd->bhnd", attention, value)

        # Rearrange dimensions and return the output tensor
        x = rearrange(x, "b h n d -> b n (h d)")

        return x


class SelfOutput(nn.Module):
    # SelfOutput module applies a linear transformation and dropout to the input tensor
    def __init__(self, config: SimpleNamespace) -> None:
        super(SelfOutput, self).__init__()

        # Linear transformation
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply linear transformation
        x = self.dense(x)
        # Apply dropout
        x = self.dropout(x)

        return x


class Attention(nn.Module):
    # Attention module combines SelfAttention and SelfOutput modules
    def __init__(self, config: SimpleNamespace) -> None:
        super(Attention, self).__init__()

        # SelfAttention module
        self.attention = SelfAttention(config)
        # SelfOutput module
        self.output = SelfOutput(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply self-attention mechanism
        x = self.attention(x)
        # Apply linear transformation and dropout
        x = self.output(x)

        return x


class Intermediate(nn.Module):
    # Intermediate module applies a linear transformation and activation function to the input tensor
    def __init__(self, config: SimpleNamespace) -> None:
        super(Intermediate, self).__init__()

        # Linear transformation
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # GELU activation function
        self.intermediate_act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply linear transformation
        x = self.dense(x)
        # Apply activation function
        x = self.intermediate_act_fn(x)

        return x


class Output(nn.Module):
    # Output module applies a linear transformation and dropout to the input tensor
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()

        # Linear transformation
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply linear transformation
        x = self.dense(x)
        # Apply dropout
        x = self.dropout(x)

        return x


class Layer(nn.Module):
    # Layer module represents a single layer in the ViT model
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()

        # Layer normalization before the attention mechanism
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        # Self-attention mechanism
        self.attention = Attention(config)
        # Layer normalization after the attention mechanism
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        # Intermediate module applies a linear transformation and activation function
        self.intermediate = Intermediate(config)
        # Output module applies a linear transformation and dropout
        self.output = Output(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply layer normalization before the attention mechanism
        x = self.attention(self.layernorm_before(x)) + x

        # Apply layer normalization after the attention mechanism
        x = self.output(self.intermediate(self.layernorm_after(x))) + x

        return x


class Encoder(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()

        # Create a list of Layer modules based on the number of hidden layers in the configuration
        self.layer = nn.ModuleList(
            [Layer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply each layer in the list to the input tensor sequentially
        for layer in self.layer:
            x = layer(x)

        return x


class ViTModel(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()

        # Embeddings module for input tokenization and positional encoding
        self.embeddings = Embeddings(config)
        # Encoder module for processing the input sequence
        self.encoder = Encoder(config)
        # Layer normalization module after the encoder
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the embeddings module to the input tensor
        x = self.embeddings(x)
        # Apply the encoder module to the embedded input tensor
        x = self.encoder(x)
        # Apply layer normalization to the output of the encoder
        x = self.layernorm(x)

        return x
