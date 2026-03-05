import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Multi-Kernel Signal Encoder
# =========================
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}
class Scale(nn.Module):
    """
    Scale-aware temporal gating for AMC features

    Input:  [B, C, T]
    Output: [B, C, T]
    """

    def __init__(self, channel, gap_size=1):
        super().__init__()

        # Global temporal pooling
        self.gap = nn.AdaptiveAvgPool1d(gap_size)

        # Channel-wise modulation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.SiLU()
        )

    def forward(self, x):
        """
        x: [B, C, T]
        """

        # Preserve original signal
        x_raw = x

        # Magnitude information (noise-aware)
        x_abs = x.abs()

        # Global channel statistics
        s = self.gap(x)             # [B, C, 1]
        s = s.squeeze(-1)           # [B, C]
        s = self.fc(s)              # [B, C]
        s = s.unsqueeze(-1)         # [B, C, 1]

        # Soft thresholding
        gate = torch.relu(x_abs - s)

        # Phase-preserving modulation
        out = torch.sigmoid(x_raw) * gate

        return out



class MultiKernelEmbedding(nn.Module):
    """
    Vision-style multi-scale convolution for AMC
    Input:  [B, 1, 128, 2]
    Output: [B, T, D]
    """

    def __init__(self, embed_dim=128):
        super().__init__()

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d((0, 0, 1, 1)),
                nn.Conv2d(1, embed_dim // 4, kernel_size=(3, 2))
            ),
            nn.Sequential(
                nn.ReflectionPad2d((0, 0, 2, 2)),
                nn.Conv2d(1, embed_dim // 4, kernel_size=(5, 2))
            ),
            nn.Sequential(
                nn.ReflectionPad2d((0, 0, 4, 4)),
                nn.Conv2d(1, embed_dim // 4, kernel_size=(9, 2))
            ),
            nn.Sequential(
                nn.ReflectionPad2d((0, 0, 5, 5)),
                nn.Conv2d(1, embed_dim // 4, kernel_size=(11, 2))
            ),
        ])


        self.norm = nn.BatchNorm1d(embed_dim)

        
        self.scale = Scale(
            channel=embed_dim,
            gap_size=1
        )

    def forward(self, x):
        # x: [B, 1, 128, 2]

        feats = []
        for conv in self.branches:
            f = conv(x)          # [B, D/4, 128, 1]
            f = f.squeeze(-1)    # [B, D/4, 128]
            feats.append(f)

        x = torch.cat(feats, dim=1)  # [B, D, 128]
        x = self.norm(x)

       
        x = self.scale(x)            # [B, D, 128]

        return x.permute(0, 2, 1)    # [B, 128, D]



# =========================
# Persistent Memory Tokens (Titans-style)
# =========================
class TitansMemory(nn.Module):
    """
    TITANS-style persistent memory with gradient-based updates (training-time)
    """

    def __init__(self, memory_tokens=8, embed_dim=128, lr=0.1):
        super().__init__()

        self.M = nn.Parameter(torch.zeros(memory_tokens, embed_dim))
        nn.init.normal_(self.M, mean=0.0, std=0.02)

        self.lr = lr
        self.norm = nn.LayerNorm(embed_dim)
        self._pending_grad = None

    def read(self, q):
        """
        q: [B, D]
        """
        attn = torch.softmax(
            q @ self.M.T / (q.size(-1) ** 0.5), dim=-1
        )  # [B, M]

        r = attn @ self.M  # [B, D]
        return r, attn

    def update(self, r, target):
        """
        r:      [B, D]
        target: [B, D]
        """
        # Allow gradients to flow to memory, but keep target detached to avoid
        # second-order gradients through the encoder.
        loss = F.mse_loss(r, target.detach(), reduction='mean')
        if not loss.requires_grad or not self.M.requires_grad:
            return loss.detach()

        # Compute gradient w.r.t. memory parameters and store for later update
        grad = torch.autograd.grad(
            loss,
            self.M,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )[0]
        if grad is None:
            return loss.detach()

        self._pending_grad = grad.detach()
        return loss

    def apply_update(self):
        """Apply the pending memory update after backward to avoid in-place ops during autograd."""
        if self._pending_grad is None:
            return
        with torch.no_grad():
            self.M -= self.lr * self._pending_grad
            self.M.copy_(self.norm(self.M))
        self._pending_grad = None



class Mlp(nn.Module):
    def __init__(self,input_dim,  hidden_dim, output_dim):
        super(Mlp, self).__init__()
       

        self.Linear1 = nn.Linear(input_dim // 2, hidden_dim // 2)
        self.Linear2 = nn.Linear(input_dim // 2, hidden_dim // 2)
        self.Linear3 = nn.Linear(hidden_dim // 2, output_dim //2)
        self.Linear4 = nn.Linear(hidden_dim // 2, output_dim //2)
        self.act_fn = ACT2FN["gelu"]
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.Linear1.weight)
        nn.init.xavier_uniform_(self.Linear2.weight)
        nn.init.normal_(self.Linear1.bias, std=1e-6)
        nn.init.normal_(self.Linear2.bias, std=1e-6)
        nn.init.xavier_uniform_(self.Linear3.weight)
        nn.init.xavier_uniform_(self.Linear4.weight)
        nn.init.normal_(self.Linear3.bias, std=1e-6)
        nn.init.normal_(self.Linear4.bias, std=1e-6)

    def forward(self, x):
        m, n = torch.split(x, x.size(-1) // 2, dim=-1)
        m_out = self.Linear2(m)
        m_out = self.act_fn(m_out)
        n_out = self.Linear1(n)
        n_out = self.act_fn(n_out)
        db_mlp = torch.cat((self.Linear4(m_out), self.Linear3(n_out)), dim=-1)
        return db_mlp

# =========================
# Memory-Augmented Transformer Block
# =========================
class MemoryTransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)

        self.mlp =nn.Sequential(Mlp(dim, int(dim * mlp_ratio), dim))
           
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


# =========================
# Dual-Branch Classification Head
# =========================
class DualBranchClassifier(nn.Module):
    """
    Combines expressive + lightweight branches
    """

    def __init__(self, dim, num_classes):
        super().__init__()

        self.branch_a = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_classes)
        )

        self.branch_b = nn.Linear(dim, num_classes)

    def forward(self, x):
        return (self.branch_a(x) + self.branch_b(x)) * 0.5


# =========================
# MT_MCNet with Memory
# =========================
class MT_MCNet(nn.Module):
    def __init__(
        self,
        num_classes,
        embed_dim=64,
        depth=4,
        memory_tokens=32,
        heads=4
    ):
        super().__init__()

        self.encoder = MultiKernelEmbedding(embed_dim)

        self.memory = TitansMemory(
            memory_tokens=memory_tokens,
            embed_dim=embed_dim,
            lr=0.001
        )

        self.blocks = nn.ModuleList([
            MemoryTransformerBlock(embed_dim, heads)
            for _ in range(depth)
        ])

        self.classifier = DualBranchClassifier(embed_dim, num_classes)

    def forward(self, x, update_memory=True, **kwargs):
        """
        Returns:
        logits, memory_loss
        """
        B, C, H, W = x.shape
        #x = x.view(B, W, H) #B, 2, 128
        #print(x.shape)

        x = self.encoder(x)   # [B, T, D]

        # Global query from signal
        q = x.mean(dim=1)     # [B, D]

        # TITANS read
        r, attn = self.memory.read(q)

        # Allow kwargs override for compatibility
        update_memory = kwargs.get("update_memory", update_memory)
        # Memory update target
        if update_memory and self.training and torch.is_grad_enabled():
            memory_loss = self.memory.update(r, q)
        else:
            memory_loss = torch.zeros((), device=x.device)

        # Inject memory token
        mem_token = r.detach().unsqueeze(1)   # [B, 1, D]
        x = torch.cat([mem_token, x], dim=1)

        for blk in self.blocks:
            x = blk(x)

        out = x[:, 0]   # memory token output

        logits = self.classifier(out)

        return logits, memory_loss

    def apply_memory_update(self):
        self.memory.apply_update()



