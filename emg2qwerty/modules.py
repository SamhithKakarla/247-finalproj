# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


# YOUR CODE HERE ==============================


class CNNGRUEncoder(nn.Module):
    """A CNN + GRU hybrid encoder"""

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int],
        kernel_width: int,
        gru_hidden_size: int,
        gru_layers: int,
        gru_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # CNN front-end
        self.cnn = TDSConvEncoder(
            num_features=num_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
        )

        # GRU back-end
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            bidirectional=True,
            batch_first=False,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )

        self.proj = nn.Linear(gru_hidden_size * 2, num_features)
        self.norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        x = self.cnn(inputs)
        x, _ = self.gru(x)

        x = self.proj(x)
        x = self.norm(x)

        return x


class CNNLSTMEncoder(nn.Module):
    """A CNN + LSTM hybrid encoder"""

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int],
        kernel_width: int,
        lstm_hidden_size: int,
        lstm_layers: int,
        lstm_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # CNN front-end (temporal convolution encoder)
        self.cnn = TDSConvEncoder(
            num_features=num_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
        )

        # LSTM back-end
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=False,  # expects (T, N, F)
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # Project bidirectional output back to num_features
        self.proj = nn.Linear(lstm_hidden_size * 2, num_features)
        self.norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (T, N, F)
        x = self.cnn(inputs)

        # LSTM output: (T, N, 2 * hidden_size)
        x, _ = self.lstm(x)

        # Project back to original feature dimension
        x = self.proj(x)

        # Stabilize representation
        x = self.norm(x)

        return x


class CNNPyramidalLSTMEncoder(nn.Module):
    """CNN + Pyramidal BiLSTM encoder for CTC"""

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int],
        kernel_width: int,
        lstm_hidden_size: int,
        lstm_layers: int,
        lstm_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.lstm_layers = lstm_layers

        # CNN front-end
        self.cnn = TDSConvEncoder(
            num_features=num_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
        )

        # Build pyramidal LSTM stack manually
        self.lstm_stack = nn.ModuleList()

        input_dim = num_features
        for layer in range(lstm_layers):
            self.lstm_stack.append(
                nn.LSTM(
                    input_size=input_dim,
                    hidden_size=lstm_hidden_size,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=False,
                )
            )

            # After pyramidal reduction, input dimension doubles
            input_dim = lstm_hidden_size * 2 * 2  # biLSTM output * concat(2 timesteps)

        self.proj = nn.Linear(lstm_hidden_size * 2, num_features)
        self.norm = nn.LayerNorm(num_features)
        self.dropout = nn.Dropout(lstm_dropout)

    def _pyramid_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduce time dimension by factor of 2.
        Input:  (T, N, F)
        Output: (T//2, N, 2F)
        """
        T, N, F = x.size()

        if T % 2 != 0:
            x = x[:-1]
        x = x.view(T // 2, 2, N, F)
        x = torch.cat([x[:, 0], x[:, 1]], dim=-1)
        return x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (T, N, F)
        x = self.cnn(inputs)

        for i, lstm in enumerate(self.lstm_stack):
            # BiLSTM
            x, _ = lstm(x)

            # Apply dropout between layers
            x = self.dropout(x)

            # Apply pyramidal reduction except after last layer
            if i < self.lstm_layers - 1:
                x = self._pyramid_reduce(x)

        # Project back to original feature dimension
        x = self.proj(x)
        x = self.norm(x)

        return x


class CNNTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int],
        kernel_width: int,
        max_seq_len: int = 1000,
        transformer_layers: int = 2,
        transformer_heads: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features

        # --- CNN ---
        self.cnn = TDSConvEncoder(
            num_features=num_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
        )

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=transformer_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Positional encoding after conv downsampling
        self.positional_encoding = nn.Parameter(0.01 * torch.randn(max_seq_len, 1, num_features))
        self.output_norm = nn.LayerNorm(num_features)

    def forward(self, x):
        T, N, F_in = x.shape
        x_ = self.cnn(x)
        T_down = x_.shape[0]

        pe = self.positional_encoding

        if T_down <= pe.shape[0]:
            x_ = x_ + pe[:T_down]
        else:
            # Repeat for longer sequence
            repeat_factor = (T_down + pe.shape[0] - 1) // pe.shape[0] 
            pe_extended = pe.repeat(repeat_factor, N, 1)[:T_down, :N, :]
            x_ = x_ + pe_extended

        x_ = x_.permute(1, 0, 2)
        out = self.transformer(x_)
        out = self.output_norm(out)
        return out.permute(1, 0, 2)


# END YOUR CODE HERE =======================================
