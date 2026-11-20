"""ReactEMG (Any2Any) encoder-only multimodal Transformer model."""

from __future__ import annotations

import torch
from torch import nn


class Any2Any_Model(nn.Module):
    """Encoder-only Transformer jointly modeling masked EMG and masked intent tokens.

    Exact architectural match to ``roamlab/reactemg/nn_models.py::Any2Any_Model`` for
    per-timestep 6-class (or 3-class) intent segmentation and EMG reconstruction.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        nhead: int = 4,
        dropout: float = 0.15,
        activation: str = "gelu",
        num_layers: int = 2,
        window_size: int = 600,
        embedding_method: str = "linear_projection",
        mask_alignment: str = "aligned",
        share_pe: bool = True,
        tie_weight: bool = False,
        use_input_layernorm: bool = True,
        num_classes: int = 6,
        output_reduction_method: str = "none",
        chunk_size: int = 0,
        channels: int = 8,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.embedding_method = str(embedding_method)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.mask_alignment = str(mask_alignment)
        self.share_pe = bool(share_pe)
        self.tie_weight = bool(tie_weight)
        self.use_input_layernorm = bool(use_input_layernorm)
        self.output_reduction_method = str(output_reduction_method)
        self.chunk_size = int(chunk_size)
        self.window_size = int(window_size)

        # Intent token vocabulary: [0..num_classes-1] valid intents + mask token at index num_classes
        self.action_vocab_size = self.num_classes + 1
        self.mask_token_id = self.num_classes
        self.action_embedding = nn.Embedding(self.action_vocab_size, self.embedding_dim)
        self.action_modality_specific_embedding = nn.Parameter(torch.empty(1, 1, self.embedding_dim))
        nn.init.uniform_(self.action_modality_specific_embedding, a=-0.02, b=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.nhead,
            dim_feedforward=4 * self.embedding_dim,
            dropout=float(dropout),
            activation=str(activation),
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )

        self.action_output_projection = nn.Linear(self.embedding_dim, self.action_vocab_size)

        self.emg_embedding = nn.Conv1d(self.channels, self.embedding_dim, kernel_size=1, stride=1)
        self.linear_projection_learnable_mask = nn.Parameter(torch.empty(1, self.embedding_dim, 1))
        nn.init.uniform_(self.linear_projection_learnable_mask, a=-0.02, b=0.02)

        if self.use_input_layernorm:
            self.emg_in_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.emg_modality_specific_embedding = nn.ParameterDict(
            {"emg_id_embedding": nn.Parameter(torch.empty(1, 1, self.embedding_dim))}
        )
        nn.init.uniform_(self.emg_modality_specific_embedding["emg_id_embedding"], a=-0.02, b=0.02)

        if self.share_pe:
            self.emg_positional_encoding = nn.Parameter(torch.empty(1, self.window_size, self.embedding_dim))
            nn.init.uniform_(self.emg_positional_encoding, a=-0.02, b=0.02)
            self.action_positional_encoding = self.emg_positional_encoding
        else:
            self.emg_positional_encoding = nn.Parameter(torch.empty(1, self.window_size, self.embedding_dim))
            self.action_positional_encoding = nn.Parameter(torch.empty(1, self.window_size, self.embedding_dim))
            nn.init.uniform_(self.emg_positional_encoding, a=-0.02, b=0.02)
            nn.init.uniform_(self.action_positional_encoding, a=-0.02, b=0.02)

        self.emg_output_projection = nn.Conv1d(self.embedding_dim, self.channels, kernel_size=1, stride=1)

        if self.tie_weight:
            self.action_output_projection.weight = self.action_embedding.weight

        if self.output_reduction_method == "learned":
            self.chunk_aggregator = nn.Linear(self.chunk_size * self.embedding_dim, self.embedding_dim)
        else:
            self.chunk_aggregator = None

    def _reduce_sequence_by_pooling(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        bsz, steps, dim = x.size()
        return x.reshape(bsz, steps // chunk_size, chunk_size, dim).mean(dim=2)

    def _reduce_sequence_by_learned(self, x: torch.Tensor, chunk_size: int, aggregator: nn.Module) -> torch.Tensor:
        bsz, steps, dim = x.size()
        return aggregator(x.reshape(bsz, steps // chunk_size, chunk_size * dim))

    def forward(
        self,
        masked_emg: torch.Tensor,
        masked_actions: torch.Tensor,
        task_idx: torch.Tensor,
        mask_positions_emg: torch.Tensor,
        return_output: bool = False,
        emg_window: torch.Tensor | None = None,
        action_window: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass over paired EMG ``(B, W, C)`` and intent tokens ``(B, W)``."""
        _ = (emg_window, action_window)
        if masked_emg.dim() != 3 or masked_emg.shape[1] != self.window_size or masked_emg.shape[2] != self.channels:
            raise ValueError(
                f"Expected masked_emg of shape (B, {self.window_size}, {self.channels}), got {tuple(masked_emg.shape)}"
            )
        batch_size = masked_emg.size(0)

        emg_proj = self.emg_embedding(masked_emg.transpose(1, 2))  # (B, D, W)
        if self.use_input_layernorm:
            emg_proj = self.emg_in_layer_norm(emg_proj.transpose(1, 2)).transpose(1, 2)

        if self.mask_alignment != "aligned":
            raise ValueError(f"Unsupported mask_alignment: {self.mask_alignment!r}")

        mask_tokens = self.linear_projection_learnable_mask.expand(batch_size, -1, self.window_size)
        mask_pos_t = mask_positions_emg.transpose(1, 2)[:, 0, :].unsqueeze(1).expand(-1, self.embedding_dim, -1)
        mask_pos_f = mask_pos_t.type_as(mask_tokens)
        emg_proj = emg_proj * (1.0 - mask_pos_f) + mask_tokens * mask_pos_f

        emg_tokens = (
            emg_proj.transpose(1, 2)
            + self.emg_modality_specific_embedding["emg_id_embedding"]
            + self.emg_positional_encoding
        )
        action_tokens = (
            self.action_embedding(masked_actions)
            + self.action_modality_specific_embedding
            + self.action_positional_encoding
        )

        src = torch.cat([emg_tokens, action_tokens], dim=1)  # (B, 2W, D)
        seq_len = src.size(1)

        if isinstance(task_idx, int):
            task_idx_t = torch.full((batch_size,), task_idx, dtype=torch.long, device=src.device)
        elif not torch.is_tensor(task_idx):
            task_idx_t = torch.as_tensor(task_idx, dtype=torch.long, device=src.device).reshape(-1)
        else:
            task_idx_t = task_idx.reshape(-1)

        task3_rows = (task_idx_t == 3).nonzero(as_tuple=False).flatten()
        if task3_rows.numel() > 0:
            # Block EMG->action and action->EMG cross-attention cleanly, and allow action tokens
            # self-attention to avoid NaN rows in PyTorch MHA softmax.
            attention_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=src.device)
            attention_mask[task3_rows, : self.window_size, self.window_size :] = True
            attention_mask[task3_rows, self.window_size :, : self.window_size] = True
            attention_mask = attention_mask.repeat_interleave(self.nhead, dim=0)
            encoded = self.transformer_encoder(src, mask=attention_mask)
        else:
            encoded = self.transformer_encoder(src)
        if return_output:
            return encoded

        emg_encoded = encoded[:, : self.window_size, :]
        action_encoded = encoded[:, self.window_size :, :]

        emg_output = self.emg_output_projection(emg_encoded.transpose(1, 2)).transpose(1, 2)

        if self.output_reduction_method == "none":
            action_output = self.action_output_projection(action_encoded)
        elif self.output_reduction_method == "pooling":
            pooled = self._reduce_sequence_by_pooling(action_encoded, self.chunk_size)
            action_output = self.action_output_projection(pooled)
        elif self.output_reduction_method == "learned":
            assert self.chunk_aggregator is not None
            learned = self._reduce_sequence_by_learned(action_encoded, self.chunk_size, self.chunk_aggregator)
            action_output = self.action_output_projection(learned)
        else:
            raise ValueError(f"Unknown output_reduction_method: {self.output_reduction_method!r}")

        return emg_output, action_output

    def predict_action_logits(self, emg_window: torch.Tensor) -> torch.Tensor:
        """Predict valid class logits ``(B, W, num_classes)`` from unmasked EMG ``(B, W, C)``."""
        batch_size = emg_window.size(0)
        device = emg_window.device
        masked_actions = torch.full((batch_size, self.window_size), self.mask_token_id, dtype=torch.long, device=device)
        task_idx = torch.zeros((batch_size,), dtype=torch.long, device=device)
        mask_positions_emg = torch.zeros_like(emg_window, dtype=torch.bool)
        _, action_logits = self.forward(emg_window, masked_actions, task_idx, mask_positions_emg)
        return action_logits[..., : self.num_classes]


Model = Any2Any_Model
