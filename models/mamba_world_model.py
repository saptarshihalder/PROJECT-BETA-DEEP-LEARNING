

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class SelectiveSSM(nn.Module):

    def __init__(self, d_model: int = 128, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand  # expanded inner dimension

        # ── Input projection (split into two paths like Mamba) ────
        #    Path 1: main path through SSM
        #    Path 2: gating path (skip connection with SiLU gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # ── Selective parameters: B_t, C_t, Δ_t from input ───────
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        # Outputs: B_t (d_state), C_t (d_state), delta_t (1)

        # ── Learnable log(A) — initialized as negative (stable) ──
        # A is diagonal: (d_inner, d_state)
        log_A = torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32)
        ).unsqueeze(0).expand(self.d_inner, -1)
        self.log_A = nn.Parameter(-log_A)  # negative for stability

        # ── Delta (Δ) bias — controls default discretization step ─
        # Initialize so softplus(bias) ≈ small positive value
        self.dt_bias = nn.Parameter(torch.randn(self.d_inner) * 0.01 - 2.0)

        # ── D parameter (skip connection in SSM output) ──────────
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ── Output projection back to d_model ────────────────────
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # ── Layer norm (pre-norm like Mamba 2) ───────────────────
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, h: torch.Tensor = None):
        B = x.size(0)
        residual = x

        # Pre-norm
        x = self.norm(x)

        # Input projection -> main path + gate
        xz = self.in_proj(x)                              # (B, d_inner*2)
        x_main, z = xz.chunk(2, dim=-1)                   # each (B, d_inner)

        # Selective parameters from input
        x_proj_out = self.x_proj(x_main)                   # (B, d_state*2+1)
        B_t = x_proj_out[:, :self.d_state]                  # (B, d_state)
        C_t = x_proj_out[:, self.d_state:2*self.d_state]    # (B, d_state)
        delta_raw = x_proj_out[:, -1]                        # (B,)

        # Discretization step: Δ = softplus(raw + bias_mean)
        delta = F.softplus(delta_raw.unsqueeze(-1) + self.dt_bias)  # (B, d_inner)

        # Discretize A:  Ā = exp(Δ · A)
        A = self.log_A.exp()                                # (d_inner, d_state)
        A_bar = torch.exp(-delta.unsqueeze(-1) * A)         # (B, d_inner, d_state)

        # Discretize B:  B̄ = Δ · B_t  (zero-order hold)
        B_bar = delta.unsqueeze(-1) * B_t.unsqueeze(1)      # (B, d_inner, d_state)

        # Initialize hidden state if needed
        if h is None:
            h = torch.zeros(B, self.d_inner, self.d_state,
                            device=x.device, dtype=x.dtype)

        # State update:  h_new = Ā · h + B̄ · x
        h_new = A_bar * h + B_bar * x_main.unsqueeze(-1)    # (B, d_inner, d_state)

        # Readout:  y = C_t · h_new  +  D · x  (with skip)
        y_ssm = (h_new * C_t.unsqueeze(1)).sum(dim=-1)      # (B, d_inner)
        y_ssm = y_ssm + self.D * x_main                      # skip connection

        # Gated output (SiLU gate like Mamba)
        y_gated = y_ssm * F.silu(z)                          # (B, d_inner)

        # Project back to d_model
        y = self.out_proj(y_gated)                            # (B, d_model)

        # Residual connection
        y = y + residual

        return y, h_new

    def init_hidden(self, batch_size: int, device: torch.device = None):
        """Create zero-initialized hidden state."""
        if device is None:
            device = self.log_A.device
        return torch.zeros(
            batch_size, self.d_inner, self.d_state,
            device=device, dtype=torch.float32,
        )
class MambaWorldModel(nn.Module):

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 16,
        max_actions: int = 9,
        hidden_dim: int = 128,
        state_dim: int = 16,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # ── Action embedding: integer action -> R^16 ─────────────
        self.action_embed = nn.Embedding(max_actions, action_dim)

        # ── Fusion layer: concat(z_t, e(a_t)) -> hidden_dim ─────
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
        )

        # ── Mamba 2 blocks ───────────────────────────────────────
        self.blocks = nn.ModuleList([
            SelectiveSSM(d_model=hidden_dim, d_state=state_dim)
            for _ in range(num_blocks)
        ])

        # ── Prediction heads ─────────────────────────────────────
        # Next state: predict z_{t+1}
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Reward: predict r_t (scalar)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Done: predict d_t (logit, use BCE loss)
        self.done_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ── Internal hidden states for multi-step rollouts ───────
        self._hidden_states = None

    def reset_hidden(self, batch_size: int, device: torch.device = None):
        """Reset all SSM hidden states — call before starting a new rollout."""
        if device is None:
            device = next(self.parameters()).device
        self._hidden_states = [
            block.init_hidden(batch_size, device)
            for block in self.blocks
        ]

    def forward(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        hidden_states: list = None,
    ):
        # Use provided or internal hidden states
        h_states = hidden_states if hidden_states is not None else self._hidden_states

        # ── 1. Action embedding ──────────────────────────────────
        a_emb = self.action_embed(action)                   # (B, 16)

        # ── 2. Fusion: concat + project ──────────────────────────
        fused = torch.cat([z_t, a_emb], dim=-1)             # (B, 80)
        h = self.fusion(fused)                               # (B, 128)

        # ── 3. Pass through Mamba blocks ─────────────────────────
        new_h_states = []
        for i, block in enumerate(self.blocks):
            h_prev = h_states[i] if h_states is not None else None
            h, h_new = block(h, h_prev)                      # (B, 128)
            new_h_states.append(h_new)

        # Update internal hidden states
        self._hidden_states = new_h_states

        # ── 4. Prediction heads ──────────────────────────────────
        z_next = self.next_state_head(h)                     # (B, 64)
        reward = self.reward_head(h)                          # (B, 1)
        done_logit = self.done_head(h)                        # (B, 1)

        return z_next, reward, done_logit, new_h_states

    def predict(self, z_t: torch.Tensor, action: torch.Tensor):
        z_next, reward, done_logit, _ = self.forward(z_t, action)
        return z_next, reward.squeeze(-1), torch.sigmoid(done_logit).squeeze(-1)

    def multi_step_rollout(
        self,
        z_start: torch.Tensor,
        actions: torch.Tensor,
    ):
        B, H = actions.shape
        device = z_start.device

        self.reset_hidden(B, device)

        z_states = []
        rewards = []
        dones = []

        z_t = z_start
        for t in range(H):
            z_next, r, d_prob = self.predict(z_t, actions[:, t])
            z_states.append(z_next)
            rewards.append(r)
            dones.append(d_prob)
            z_t = z_next  # feed predicted state back in

        return (
            torch.stack(z_states, dim=1),    # (B, H, 64)
            torch.stack(rewards, dim=1),      # (B, H)
            torch.stack(dones, dim=1),        # (B, H)
        )
class WorldModelLoss(nn.Module):
    """
    Combined loss from paper Eq. 2:

        L_world = ||z_{t+1} - ẑ_{t+1}||^2
                + reward_weight * |r_t - r̂_t|^2
                + BCE(d_t, d̂_t)

    reward_weight = 0.5 as specified in the paper.
    """

    def __init__(self, reward_weight: float = 0.5):
        super().__init__()
        self.reward_weight = reward_weight
        self.state_loss_fn = nn.MSELoss()
        self.reward_loss_fn = nn.MSELoss()
        self.done_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        z_next_pred: torch.Tensor,
        z_next_true: torch.Tensor,
        reward_pred: torch.Tensor,
        reward_true: torch.Tensor,
        done_logit: torch.Tensor,
        done_true: torch.Tensor,
    ):
        l_state = self.state_loss_fn(z_next_pred, z_next_true)
        l_reward = self.reward_loss_fn(reward_pred, reward_true)
        l_done = self.done_loss_fn(done_logit, done_true)

        total = l_state + self.reward_weight * l_reward + l_done

        return total, (l_state.item(), l_reward.item(), l_done.item())
