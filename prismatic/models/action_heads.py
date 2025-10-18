"""
action_heads.py

Implementations of various action heads, which serve as alternatives to VLM sequential token prediction.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS



class ActionTokenPooling(nn.Module):
    """
    A module for pooling action tokens.
    It aggregates a sequence of tokens (B, NUM_TOKENS, D) into a smaller
    sequence of action chunks (B, NUM_ACTIONS_CHUNK, D) using various pooling strategies.
    """
    def __init__(self, pooling_type: str, input_dim: int, num_tokens: int = NUM_TOKENS, num_chunks: int = NUM_ACTIONS_CHUNK):
        super().__init__()
        self.pooling_type = pooling_type
        self.num_tokens = num_tokens
        self.num_chunks = num_chunks
        self.input_dim = input_dim

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_chunks, self.input_dim))

        if pooling_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        elif pooling_type == "weighted":
            # Implements an uneven pooling scheme where the first action chunk gets more tokens.
            # For example, with 64 tokens and 8 chunks, the first chunk might be pooled from 32 tokens,
            # and the rest from the remaining 32 tokens evenly. This prioritizes the first action.
            if self.num_chunks > 1:
                first_chunk_size = self.num_tokens // 2
                remaining_tokens = self.num_tokens - first_chunk_size
                base_size = remaining_tokens // (self.num_chunks - 1)
                rem = remaining_tokens % (self.num_chunks - 1)
                
                self.chunk_sizes = [first_chunk_size] + [base_size] * (self.num_chunks - 1)
                for i in range(rem):
                    self.chunk_sizes[i + 1] += 1
            else:
                self.chunk_sizes = [self.num_tokens]
            
            assert sum(self.chunk_sizes) == self.num_tokens, "Sum of chunk sizes must equal total number of tokens."
        elif pooling_type == "linear_fusion":
            if self.num_tokens % self.num_chunks != 0:
                raise ValueError("For linear_fusion pooling, num_tokens must be divisible by num_chunks.")
            tokens_per_chunk = self.num_tokens // self.num_chunks
            self.fusion_layer = nn.Linear(tokens_per_chunk, 1)
        elif pooling_type == "mixer":
            self.mixer_mlp = nn.Sequential(
                nn.LayerNorm(self.num_tokens),
                nn.Linear(self.num_tokens, self.num_tokens * 2),
                nn.ReLU(),
                nn.Linear(self.num_tokens * 2, self.num_chunks)
            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, NUM_TOKENS, D).

        Returns:
            torch.Tensor: Pooled tensor of shape (B, NUM_ACTIONS_CHUNK, D).
        """
        B, _, D = x.shape

        if self.pooling_type in ["mean", "max"]:
            if self.num_tokens % self.num_chunks != 0:
                raise ValueError("For mean/max pooling, num_tokens must be divisible by num_chunks.")
            tokens_per_chunk = self.num_tokens // self.num_chunks
            
            x = x.reshape(B, self.num_chunks, tokens_per_chunk, D)
            if self.pooling_type == "mean":
                pooled = x.mean(dim=2)
            else: # max
                pooled, _ = x.max(dim=2)

        elif self.pooling_type == "attention":
            if self.num_tokens % self.num_chunks != 0:
                raise ValueError("For attention pooling, num_tokens must be divisible by num_chunks.")
            tokens_per_chunk = self.num_tokens // self.num_chunks
            x_reshaped = x.reshape(B * self.num_chunks, tokens_per_chunk, D)

            attn_weights = torch.softmax(self.attention(x_reshaped), dim=1)
            pooled_flat = torch.sum(x_reshaped * attn_weights, dim=1)
            
            pooled = pooled_flat.reshape(B, self.num_chunks, D)

        elif self.pooling_type == "weighted":
            outputs = []
            start_idx = 0
            for chunk_size in self.chunk_sizes:
                end_idx = start_idx + chunk_size
                chunk = x[:, start_idx:end_idx, :]
                outputs.append(chunk.mean(dim=1, keepdim=True))
                start_idx = end_idx
            pooled = torch.cat(outputs, dim=1)
        
        elif self.pooling_type == "linear_fusion":
            tokens_per_chunk = self.num_tokens // self.num_chunks
            
            # Reshape to (B, NUM_CHUNKS, TOKENS_PER_CHUNK, D)
            x_reshaped = x.reshape(B, self.num_chunks, tokens_per_chunk, D)
            
            # Transpose to (B, NUM_CHUNKS, D, TOKENS_PER_CHUNK) to apply linear layer across tokens
            x_transposed = x_reshaped.transpose(2, 3)
            
            # Fuse tokens in each chunk -> (B, NUM_CHUNKS, D, 1)
            fused = self.fusion_layer(x_transposed)
            
            # Squeeze to get final shape (B, NUM_CHUNKS, D)
            pooled = fused.squeeze(-1)
            
        elif self.pooling_type == "mixer":
            # (B, NUM_TOKENS, D) -> (B, D, NUM_TOKENS)
            x_transposed = x.transpose(1, 2)
            # (B, D, NUM_TOKENS) -> (B, D, NUM_ACTIONS_CHUNK)
            pooled_transposed = self.mixer_mlp(x_transposed)
            # (B, D, NUM_ACTIONS_CHUNK) -> (B, NUM_ACTIONS_CHUNK, D)
            pooled = pooled_transposed.transpose(1, 2)

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        return pooled + self.position_embedding


class SimpleActionHead(nn.Module):
    """
    A lightweight, stackable FFN head with residual connections.
    Can be used as a full action head or as a latent feature refiner.
    """
    def __init__(self, hidden_dim: int, action_dim: int, num_layers: int = 8, ffn_dim_multiplier: int = 1):
        super().__init__()
        
        self.net = nn.ModuleList()
        for _ in range(num_layers):
            self.net.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * ffn_dim_multiplier),
                nn.ReLU(),
                nn.Linear(hidden_dim * ffn_dim_multiplier, hidden_dim),
            ))

        self.connector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, NUM_ACTIONS_CHUNK, D).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - coarse_action (torch.Tensor): Coarse action prediction of shape (B, NUM_ACTIONS_CHUNK, ACTION_DIM).
                - representation (torch.Tensor): Intermediate representation of shape (B, NUM_ACTIONS_CHUNK, D).
        """
        representation = self.connector(x)
        for block in self.net:
            representation = representation + block(representation)

        coarse_action = self.action_predictor(representation)
        return coarse_action, representation


class FeedForward(nn.Module):
    """A simple feed-forward network with ReLU activation and no dropout."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PreNormResidual(nn.Module):
    """Applies layer normalization before a function and adds a residual connection."""
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs) + x


class MlpMixerHead(nn.Module):
    """An MLP-Mixer based action head for action prediction."""

    def __init__(self, num_chunks: int, dim: int, depth: int, action_dim: int, expansion_factor: float = 1.0, expansion_factor_token: float = 1.0):
        super().__init__()

        # An MLP for token subsampling.
        # subsampler_hidden_dim = NUM_TOKENS * 2
        self.token_subsampler = nn.Sequential(
            nn.LayerNorm(NUM_TOKENS),
            # nn.Linear(NUM_TOKENS, subsampler_hidden_dim),
            # nn.ReLU(),
            nn.Linear(NUM_TOKENS, num_chunks),
        )

        class TokenMixer(nn.Module):
            """A wrapper for token mixing that handles transposing."""
            def __init__(self, num_chunks: int, expansion_factor: float):
                super().__init__()
                self.ff = FeedForward(num_chunks, int(expansion_factor * num_chunks))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.ff(x.transpose(1, 2)).transpose(1, 2)
            

        chan_ff = FeedForward(dim, int(expansion_factor_token * dim))

        self.mixer_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    PreNormResidual(dim, TokenMixer(num_chunks, expansion_factor)),
                    PreNormResidual(dim, chan_ff),
                )
                for _ in range(depth)
            ]
        )

        self.layer_norm = nn.LayerNorm(dim)
        self.action_predictor = nn.Linear(dim, action_dim)

    def predict_action(
        self,
        actions_hidden_states: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector: Optional[nn.Module] = None,
        phase: str = "Inference",
        ground_truth_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        
        # We only use the last layer's hidden states for the mixer.
        # actions_hidden_states shape: (B, num_layers, NUM_TOKENS + num_task_tokens, D)
        # We need the action tokens from the last layer: (B, NUM_TOKENS, D)
        last_layer_hidden_states = actions_hidden_states[:, -1, -NUM_TOKENS:, :]

        # Subsample tokens using a linear projection (token-mixing).
        # (B, NUM_TOKENS, D) -> (B, D, NUM_TOKENS)
        x_transposed = last_layer_hidden_states.transpose(1, 2)
        # (B, D, NUM_TOKENS) -> (B, D, NUM_ACTIONS_CHUNK)
        subsampled = self.token_subsampler(x_transposed)
        # (B, D, NUM_ACTIONS_CHUNK) -> (B, NUM_ACTIONS_CHUNK, D)
        x = subsampled.transpose(1, 2)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        predicted_actions = self.action_predictor(x)
        
        return predicted_actions, None, ground_truth_actions


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function. See https://arxiv.org/pdf/2002.05202.pdf.
    This is a wrapper for the F.silu function.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=512,
        use_pro_version=False,
        action_probing=False,
        action_pooling_type="attention",
        only_simple_action_head=False,
        ensemble_hidden_state=False,
        sim_expert_v2=False,
        add_sink_token=False,
        use_deep_recursion=False,
        n_recursion=6,
        T_recursion=3,
        perturbation_type: str = None,
        perturbation_std: float = 0.02,
        perturbation_dropout_p: float = 0.1,
        adversarial_step_size: float = 1e-3,
        condition_aware_scale: float = 0.01,
        mixup_alpha: float = 0.4,
        token_dropout_p: float = 0.1,
        deep_supervise: bool = False,
        deep_supervise_ensemble: bool = False,
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_probing = action_probing
        self.action_pooling_type = action_pooling_type
        self.only_simple_action_head = only_simple_action_head
        self.ensemble_hidden_state = ensemble_hidden_state
        self.sim_expert_v2 = sim_expert_v2
        self.add_sink_token = add_sink_token
        self.use_deep_recursion = use_deep_recursion
        self.n_recursion = n_recursion
        self.T_recursion = T_recursion
        self.perturbation_type = perturbation_type
        self.perturbation_std = perturbation_std
        self.adversarial_step_size = adversarial_step_size
        self.condition_aware_scale = condition_aware_scale
        self.mixup_alpha = mixup_alpha
        self.token_dropout_p = token_dropout_p
        self.deep_supervise = deep_supervise
        self.deep_supervise_ensemble = deep_supervise_ensemble

        if self.use_deep_recursion:
            self.token_pooler = ActionTokenPooling(
                pooling_type=self.action_pooling_type,
                input_dim=hidden_dim,
                num_tokens=NUM_TOKENS,
                num_chunks=NUM_ACTIONS_CHUNK,
            )
            self.recursive_action_head = RecursiveActionHead(
                vlm_dim=hidden_dim,
                action_dim=self.action_dim,
                latent_dim=hidden_dim,
                num_chunks=NUM_ACTIONS_CHUNK,
                mixer_depth=4, # A reasonable default
            )
            return

        # Create heads for deep supervision, probing, or simple head mode
        if self.deep_supervise or self.deep_supervise_ensemble:
            self.token_pooler_low = ActionTokenPooling(
                pooling_type=self.action_pooling_type,
                input_dim=hidden_dim,
                num_tokens=NUM_TOKENS,
                num_chunks=NUM_ACTIONS_CHUNK,
            )
            self.coarse_action_head_low = SimpleActionHead(hidden_dim, self.action_dim)
            self.token_pooler_mid = ActionTokenPooling(
                pooling_type=self.action_pooling_type,
                input_dim=hidden_dim,
                num_tokens=NUM_TOKENS,
                num_chunks=NUM_ACTIONS_CHUNK,
            )
            self.coarse_action_head_mid = SimpleActionHead(hidden_dim, self.action_dim)
            self.token_pooler_high = ActionTokenPooling(
                pooling_type=self.action_pooling_type,
                input_dim=hidden_dim,
                num_tokens=NUM_TOKENS,
                num_chunks=NUM_ACTIONS_CHUNK,
            )
            self.coarse_action_head = SimpleActionHead(hidden_dim, self.action_dim)

        elif self.action_probing or self.only_simple_action_head:
            self.token_pooler = ActionTokenPooling(
                pooling_type=self.action_pooling_type,
                input_dim=hidden_dim,
                num_tokens=NUM_TOKENS,
                num_chunks=NUM_ACTIONS_CHUNK,
            )
            self.coarse_action_head = SimpleActionHead(hidden_dim, self.action_dim)

        # Determine perturbation dimension based on the architecture.
        # If not using the simple head and not probing, the perturbation dimension is different.
        pert_dim = self.action_dim * hidden_dim if not self.only_simple_action_head and not self.action_probing else hidden_dim
        if self.perturbation_type == "learnable_gaussian":
            self.perturbations = nn.Parameter(torch.zeros(NUM_ACTIONS_CHUNK, pert_dim))
            nn.init.normal_(self.perturbations, mean=0.0, std=self.perturbation_std)
        elif self.perturbation_type == "dropout":
            self.dropout = nn.Dropout(p=perturbation_dropout_p)
        elif self.perturbation_type == "condition_aware":
            self.noise_generator = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, pert_dim),
            )

        if not self.only_simple_action_head:
            self.model = MLPResNet(
                num_blocks=24,
                input_dim=self.action_dim * input_dim if not self.action_probing else hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=action_dim,
                use_pro_version=use_pro_version,
                sim_expert_v2=sim_expert_v2,
                add_sink_token=add_sink_token,
            )

    def _apply_perturbations(self, x: torch.Tensor, ground_truth_actions: Optional[torch.Tensor] = None, proprio: Optional[torch.Tensor] = None, proprio_projector: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Applies a configured perturbation to the input tensor as a regularization technique.

        Args:
            x (torch.Tensor): The input tensor to be perturbed.
            ground_truth_actions (Optional[torch.Tensor]): Ground truth actions, required for adversarial perturbations.
            proprio (Optional[torch.Tensor]): Proprioceptive state, required for condition-aware perturbations.
            proprio_projector (Optional[nn.Module]): Proprioceptive projector, required for condition-aware perturbations.

        Returns:
            torch.Tensor: The perturbed tensor.
        """
        if not self.training or self.perturbation_type == "none":
            return x

        if self.perturbation_type == "learnable_gaussian":
            return x + self.perturbations
        elif self.perturbation_type == "random_gaussian":
            noise = torch.randn_like(x) * self.perturbation_std
            return x + noise
        elif self.perturbation_type == "dropout":
            return self.dropout(x)
        elif self.perturbation_type == "adversarial":
            if ground_truth_actions is None:
                # Cannot compute adversarial perturbation without ground truth, so we return the input as is.
                return x

            # FGSM-like single-step adversarial perturbation
            x_adv = x.clone().detach().requires_grad_(True)
            
            # A single forward and backward pass is needed to get the gradient w.r.t. the input.
            with torch.enable_grad():
                # We only need the prediction from the model to compute the loss.
                # The full context (h_a, p, h_t) is not necessary for this simplified adversarial step.
                predicted_actions, _ = self.model(x_adv, h_a=None, p=None, h_t=None)
                loss = F.l1_loss(predicted_actions, ground_truth_actions)

            # Compute gradients with respect to the input
            loss.backward()
            
            # Add the perturbation
            perturbation = self.adversarial_step_size * x_adv.grad.sign()
            return x + perturbation
        elif self.perturbation_type == "condition_aware":
            if proprio is None or proprio_projector is None:
                return x
            
            # Project proprioceptive state
            batch_size = x.shape[0]
            proprio = proprio.reshape(batch_size, -1).to(x.dtype)
            proprio_features = proprio_projector(proprio)  # (B, D_llm)

            # Generate and scale noise
            noise = self.noise_generator(proprio_features)  # (B, D_pert)
            # Reshape noise to match input shape for broadcasting
            noise = noise.unsqueeze(1).expand_as(x) # (B, NUM_ACTIONS_CHUNK, D_pert)
            return x + noise * self.condition_aware_scale
        elif self.perturbation_type == "token_dropout":
            if not self.training or self.token_dropout_p == 0:
                return x

            B, N, D = x.shape
            keep_prob = 1 - self.token_dropout_p
            
            # Create a mask for tokens to keep of shape (B, N, 1) and apply it.
            mask = torch.bernoulli(torch.full((B, N, 1), keep_prob, device=x.device, dtype=x.dtype))
            
            # Scale the output to maintain the same expected sum.
            return x * mask
        return x

    def predict_action(
            self, 
            actions_hidden_states, 
            proprio=None, 
            proprio_projector=None,
            phase="Inference",
            ground_truth_actions: Optional[torch.Tensor] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if self.use_deep_recursion:
            batch_size = actions_hidden_states.shape[0]
            device = actions_hidden_states.device

            # x_query from VLM hidden states. Let's use the last layer and mean-pool tokens.
            # This is a simplification. A more sophisticated approach might be needed.
            actions_hidden_states = actions_hidden_states[:, -1, self.num_task_tokens :, :]
            # Pool action tokens into action chunks
            x_query = self.token_pooler(actions_hidden_states) # (B, NUM_ACTIONS_CHUNK, D)
            # Initialize y and z
            y_init = torch.zeros(batch_size, NUM_ACTIONS_CHUNK, self.action_dim, device=device, dtype=x_query.dtype)
            z_init = torch.zeros(batch_size, NUM_ACTIONS_CHUNK, self.recursive_action_head.latent_dim, device=device, dtype=x_query.dtype)
            
            # Detach x_query if we are only training the policy
            if phase == "PolicyTraining":
                x_query = x_query.detach()

            (y, z), y_hat = self.recursive_action_head.deep_recursion(
                x_query, y_init, z_init, n=self.n_recursion, T=self.T_recursion
            )
            
            # The pseudocode implies y_hat is the prediction for loss.
            # ground_truth_actions should be returned for loss calculation.
            return y_hat, None, ground_truth_actions

        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
        proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        task_hidden_states = actions_hidden_states[:, :, : self.num_task_tokens, :]
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens :, :]

        if self.only_simple_action_head and not self.deep_supervise and not self.deep_supervise_ensemble:
            if self.ensemble_hidden_state:
                # Mean hidden states across all layers
                actions_hidden_states = actions_hidden_states.mean(dim=1)  # (B, NUM_TOKENS, D)
            else:
                # Use the last layer's action hidden states
                actions_hidden_states = actions_hidden_states[:, -1, :, :]  # (B, NUM_TOKENS, D)

            # Pool action tokens into action chunks
            pooled_actions_hidden = self.token_pooler(actions_hidden_states) # (B, NUM_ACTIONS_CHUNK, D)

            # Apply perturbations during training for regularization
            if self.training:
                pooled_actions_hidden = self._apply_perturbations(
                    pooled_actions_hidden, ground_truth_actions, proprio, proprio_projector
                )

            # Get action prediction
            action, _ = self.coarse_action_head(pooled_actions_hidden)
            return action, None, ground_truth_actions

        if self.deep_supervise or self.deep_supervise_ensemble:
            # Deep supervision uses coarse heads on low, mid, and high layers.
            # We assume actions_hidden_states contains multi-layer representations.
            # Layer indices are chosen based on a 24-layer architecture.
            hidden_low = actions_hidden_states[:, 4, :, :]
            hidden_mid = actions_hidden_states[:, 16, :, :]
            hidden_high = actions_hidden_states[:, -1, :, :]

            pooled_low = self.token_pooler_low(hidden_low)
            pooled_mid = self.token_pooler_mid(hidden_mid)
            pooled_high = self.token_pooler_high(hidden_high)

            coarse_action_low, _ = self.coarse_action_head_low(pooled_low)
            coarse_action_mid, _ = self.coarse_action_head_mid(pooled_mid)
            coarse_action_high, _ = self.coarse_action_head(pooled_high)

            coarse_actions = [coarse_action_low, coarse_action_mid, coarse_action_high]

            if self.deep_supervise_ensemble:
                ensembled_action = torch.stack(coarse_actions).mean(dim=0)
                return ensembled_action, coarse_actions, ground_truth_actions
            else: # deep_supervise
                 # For deep_supervise, we don't return a single main action,
                # but the list of coarse actions for loss computation.
                # The "action" is just one of them for compatibility, but not used for loss.
                return coarse_action_high, coarse_actions, ground_truth_actions

        if self.action_probing:
            # Use the last layer's action hidden states
            last_layer_actions_hidden = actions_hidden_states[:, -1, :, :]  # (B, NUM_TOKENS, D)

            # Pool action tokens into action chunks
            pooled_actions_hidden = self.token_pooler(last_layer_actions_hidden) # (B, NUM_ACTIONS_CHUNK, D)

            # Get coarse action prediction and intermediate representation
            coarse_action, representation = self.coarse_action_head(pooled_actions_hidden)
            rearranged_actions_hidden_states = representation

        else:
            cond_actions_hidden_states = torch.zeros(
                (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
                device=device,
                dtype=actions_hidden_states.dtype,
            ).detach()

            rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
                batch_size, NUM_ACTIONS_CHUNK, -1
            )  # (batch, chunk_len, action_dim * hidden_dim)

        if self.training:
            rearranged_actions_hidden_states = self._apply_perturbations(rearranged_actions_hidden_states, ground_truth_actions, proprio, proprio_projector)

            if self.perturbation_type == "feature_mixup":
                # Beta distribution for mixing coefficient
                m = Beta(torch.tensor([self.mixup_alpha]), torch.tensor([self.mixup_alpha]))
                lam = m.sample().to(device)

                # Shuffle batch
                idx = torch.randperm(batch_size)
                
                # Mix features and ground truth actions
                rearranged_actions_hidden_states = lam * rearranged_actions_hidden_states + (1 - lam) * rearranged_actions_hidden_states[idx]
                ground_truth_actions_for_loss = lam * ground_truth_actions + (1 - lam) * ground_truth_actions[idx]
            else:
                ground_truth_actions_for_loss = ground_truth_actions


        action = self.model(rearranged_actions_hidden_states, h_a=actions_hidden_states, p=proprio_features, h_t=task_hidden_states)

        coarse_action_to_return = coarse_action if self.action_probing else None
        return action, coarse_action_to_return, ground_truth_actions_for_loss
    

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(
            self, 
            num_blocks, 
            input_dim, 
            hidden_dim, 
            output_dim,
            use_pro_version=False,
            sim_expert_v2=False,
            add_sink_token=False
            ):
        
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            if sim_expert_v2:
                self.mlp_resnet_blocks.append(MLPResNetBlock_v2(dim=hidden_dim, add_sink_token=add_sink_token))
            elif use_pro_version and not sim_expert_v2:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
                
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, h_a=None, h_t=None, p= None):
 
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x   



def apply_rope(q, k, cos, sin):
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)


    def rotate_half(x):
        # Swap even and odd dimensions and flip the signs
        x1 = x[..., ::2]   # Even subdimension
        x2 = x[..., 1::2]  # odd subdimension

        return torch.stack((-x2, x1), dim=-1).reshape_as(x)


    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)



class MLPResNetBlock(nn.Module):
    """
    One residual MLP block with cross-attention conditioning.

    This block applies multi-head attention over:
      - token features (self-attention),
      - task-related hidden states (h_t),
      - action/proprioception-related hidden states (h_a, p).
    The outputs are combined via a gating mechanism, projected back to the
    hidden dimension, and passed through a small feedforward sub-network with
    residual connection.

    Args:
        dim (int): Dimensionality of the hidden features. Must be divisible by num_heads.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        h_t (torch.Tensor, optional): Task-related hidden states of shape
                                      (batch_size, K, hidden_dim).
        h_a (torch.Tensor, optional): Action-related hidden states of shape
                                      (batch_size, 1, hidden_dim).
        p (torch.Tensor, optional): Additional conditioning features
                                    (e.g., proprioception), shape (batch_size, 1, hidden_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))



    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """

        g = self.gating_factor
        ratio_g = nn.Tanh()(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        h = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)

        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        K_t = h.size(1)
        K = h_t.size(1)

        task_k = h
        task_v = h

        adapter_k = h_t
        adapter_v = h_t

        q_1 = self.q_proj(x) # (B, T, C)
        k_tokens = self.k_proj(x)             # (B, T, C)
        v_tokens = self.v_proj(x)             # (B, T, C)
        k_task = self.k_proj(task_k)    # (B, K, C)
        v_task = self.v_proj(task_v)    # (B, K, C)

        k_adapter = self.k_proj(adapter_k)    # (B, K, C)
        v_adapter = self.v_proj(adapter_v)    # (B, K, C)

        # (B, seq_len, C) -> (B, num_heads, seq_len, head_dim)
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

        k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1)) # (B, H, T, T)
        attn_scores_task = torch.matmul(q_1, k_task.transpose(-2, -1)) * 1 # (B, H, T, K)
        attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g # (B, H, T, K)

        attn_scores = torch.cat([attn_scores_tokens, attn_scores_task, attn_scores_adapter], dim=-1) # (B, H, T, T+K)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, H, T, T+K)

        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2) # (B, H, T+K, head_dim)
        output = torch.matmul(attn_weights, v_combined) # (B, H, T, head_dim)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x) 

        return x



class MLPResNetBlock_v2(nn.Module):
    """
    One MLP ResNet block with a single cross-attention mechanism over a combined context,
    featuring QK normalization, Sink Token and a SwiGLU FFN. This block is designed for simplicity and elegance.
    """

    def __init__(self, dim, num_heads=8, add_sink_token: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.add_sink_token = add_sink_token

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            SwiGLU(dim, dim, dim),
        )

        # Q (from x only)
        self.q_proj = nn.Linear(dim, dim)

        # Cross-attention K, V from combined context
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # QK Norm
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        if self.add_sink_token:
            # Learnable sink token parameters, one for each attention head.
            # These act as a "null" key/value pair that the model can attend to
            # in order to ignore irrelevant context.
            self.sinks = nn.Parameter(torch.empty(self.num_heads))


    def forward(self, x, h_a=None, h_t=None, p=None):
        """
        Args:
            x: input tensor (query)
            h_a: adapter tokens (context)
            h_t: task tokens (context)
            p: proprioceptive conditioning vector (context)
        """
        B, T, C = x.shape

        # === 1. Prepare combined context ===
        context_tensors = []
        if h_t is not None:
            context_tensors.append(h_t)
        if h_a is not None:
            context_tensors.append(h_a)
        if p is not None:
            context_tensors.append(p)

        # If no context, skip attention and just do the FFN path with residual.
        if not context_tensors:
            residual = x
            x = self.ffn(residual) + residual
            return x

        context = torch.cat(context_tensors, dim=1)
        K_ctx = context.size(1)

        # === 2. Project Q, K, V ===
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # === 3. Reshape for Multi-Head Attention ===
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q, B, T)
        k, v = reshape_heads(k, B, K_ctx), reshape_heads(v, B, K_ctx)

        # === 4. Apply RoPE ===
        cos_q, sin_q = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q, _ = apply_rope(q, q, cos_q, sin_q)
        cos_k, sin_k = self.rope(seq_len=K_ctx, device=x.device, dtype=x.dtype)
        k, _ = apply_rope(k, k, cos_k, sin_k)

        # === 5. Apply QK Norm ===
        q = self.q_norm(q)
        k = self.k_norm(k)

        # === 6. Compute Attention ===
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.add_sink_token:
            # Expand sinks for broadcasting: (H) -> (B, H, T, 1)
            sinks = self.sinks.reshape(1, -1, 1, 1).expand(B, -1, T, -1)

            # Concatenate sink logits to attention scores
            attn_scores = torch.cat([attn_scores, sinks], dim=-1)

            # Stabilize for softmax
            attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values

            # Compute probabilities over keys + sink
            attn_probs = F.softmax(attn_scores, dim=-1, dtype=attn_scores.dtype)

            # Drop the sink probability, allowing attention to "leak away" from the context
            attn_weights = attn_probs[..., :-1]
        else:
            attn_weights = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, v)

        # === 7. Reshape and Final Projection ===
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # === 8. Residual Connection and FFN ===
        residual = output + x
        x = self.ffn(residual) + residual
        return x


class MLPResNetBlock_Pro(nn.Module):
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE, now with FiLM modulation."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            )

        # Q (from x only)
        self.q_proj = nn.Linear(dim, dim)

        # Self-Attention: K, V
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention: K, V
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention: K, V
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # gating
        self.gating_factor = nn.Parameter(torch.zeros(1))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # ---- FiLM ----
        # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now.
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim * 2),  # output γ and β
            )


    def apply_film(self, x, gamma, beta):
        """FiLM: per-channel modulation"""
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


    def forward(self, x, h_a=None, h_t=None, p=None):
        """
        h_a: adapter tokens
        h_t: task tokens
        p:   possible conditioning vector (for FiLM)
        """
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        # concat h_a and p
        h_adapter = torch.cat((h_a, p),dim=1)

        h_task = h_t
        B, T, C = x.shape
        K_a = h_adapter.size(1) if h_a is not None else 0
        K_t = h_task.size(1) if h_task is not None else 0

        # Q
        q_1 = self.q_proj(x)

        # self tokens
        k_tokens = self.k_self(x)
        v_tokens = self.v_self(x)

        # adapter tokens
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)

        # task tokens
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)


        # reshape -> multi-head
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)


        q_1 = reshape_heads(q_1, B, T)
        k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)     
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        # attention scores
        attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
        attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)))
        attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g)
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # combine V
        v_list = [v_tokens,v_adapter,v_task]
        v_combined = torch.cat(v_list, dim=2)

        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # # ---- FiLM ---- 
        # gamma_beta = self.film_gen(p)  # [B, 2C]
        # gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, C], [B, C]
        # output = self.apply_film(output, gamma, beta)

        # residual + FFN
        x = self.ffn(output + x)
        return x


class GeneralMixer(nn.Module):
    """A general MLP-Mixer architecture."""

    def __init__(self, num_chunks: int, input_dim: int, hidden_dim: int, depth: int, output_dim: int, expansion_factor: float = 1.0):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        class TokenMixer(nn.Module):
            def __init__(self, num_chunks: int, expansion_factor: float):
                super().__init__()
                self.ff = FeedForward(num_chunks, int(expansion_factor * num_chunks))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.ff(x.transpose(1, 2)).transpose(1, 2)

        chan_ff = FeedForward(hidden_dim, int(expansion_factor * hidden_dim))

        self.mixer_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    PreNormResidual(hidden_dim, TokenMixer(num_chunks, expansion_factor)),
                    PreNormResidual(hidden_dim, chan_ff),
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        return x



class PolicyNetwork(nn.Module):
    """
    A lightweight, stackable FFN head with residual connections.
    Can be used as a full action head or as a latent feature refiner.
    """
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, num_layers: int = 4, ffn_dim_multiplier: int = 1):
        super().__init__()
        
        self.net = nn.ModuleList()
        for _ in range(num_layers):
            self.net.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * ffn_dim_multiplier),
                nn.ReLU(),
                nn.Linear(hidden_dim * ffn_dim_multiplier, hidden_dim),
            ))

        self.connector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, NUM_ACTIONS_CHUNK, D).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - coarse_action (torch.Tensor): Coarse action prediction of shape (B, NUM_ACTIONS_CHUNK, ACTION_DIM).
                - representation (torch.Tensor): Intermediate representation of shape (B, NUM_ACTIONS_CHUNK, D).
        """
        representation = self.connector(x)
        for block in self.net:
            representation = representation + block(representation)

        coarse_action = self.output_head(representation)
        return coarse_action, representation

class ActionRefiner(nn.Module):
    """
    An action refiner module using MLP-Mixer. It performs one step of refinement.
    z_new = mixer(x, y, z)
    y_new = head(z_new)
    """

    def __init__(self, vlm_dim: int, action_dim: int, latent_dim: int, num_chunks: int, mixer_depth: int):
        super().__init__()
        # 
        self.action_refiner = PolicyNetwork(
            input_dim=vlm_dim,
            hidden_dim=latent_dim,  # mixer operates in latent space
            action_dim=action_dim,
            num_layers=mixer_depth,
            ffn_dim_multiplier=1
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one step of recursive refinement.

        Args:
            x (torch.Tensor): Query embedding from VLM.
            y (torch.Tensor): Current action prediction.
            z (torch.Tensor): Current latent representation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - y_new (torch.Tensor): The refined action.
                - z_new (torch.Tensor): The refined latent representation.
        """
        inp             = x + z
        y_new, z_new    = self.action_refiner(inp)
        return y_new, z_new


class RecursiveActionHead(nn.Module):
    """Implements the deep recursion algorithm for action generation."""

    def __init__(self, vlm_dim: int, action_dim: int, latent_dim: int, num_chunks: int, mixer_depth: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_refiner = ActionRefiner(vlm_dim, action_dim, latent_dim, num_chunks, mixer_depth)

    def latent_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, n: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
        for _ in range(n):  # latent recursion
            y, z = self.action_refiner(x, y, z)
        return y, z

    def deep_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, n: int = 6, T: int = 3) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # recursing T-1 times to improve y and z (no gradients needed)
        with torch.no_grad():
            for _ in range(T - 1):
                y, z = self.latent_recursion(x, y, z, n)
        # recursing once to improve y and z
        y, z = self.latent_recursion(x, y, z, n)
        # The final action `y` is the prediction.
        return (y.detach(), z.detach()), y
