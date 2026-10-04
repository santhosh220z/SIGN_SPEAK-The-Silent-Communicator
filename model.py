import torch
import torch.nn as nn
import torch.nn.functional as F


class PreprocessLayer(nn.Module):
    """
    PyTorch port of the TF Preprocess layer.
    Input: (B, T, 543, 3) - keypoints with NaN for missing
    Output: dict with:
        - 'appearance': (B, T, D_app) - static x,y features
        - 'motion': (B, T, D_mot) - velocity + acceleration features
    """
    def __init__(self, point_landmarks=None, max_len=64):
        super().__init__()
        self.max_len = max_len
        
        # Default: LIP + LHAND + RHAND + NOSE + REYE + LEYE = 118 landmarks
        if point_landmarks is None:
            LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 
                   314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 
                   88, 178, 87, 14, 317, 402, 318, 324, 308]
            LHAND = list(range(468, 489))
            RHAND = list(range(522, 543))
            NOSE = [1, 2, 98, 327]
            REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
            LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
            point_landmarks = LIP + LHAND + RHAND + NOSE + REYE + LEYE
        
        self.register_buffer('point_landmarks', torch.tensor(point_landmarks, dtype=torch.long))
        self.num_points = len(point_landmarks)
        self.app_dim = self.num_points * 2      # x, y
        self.mot_dim = self.num_points * 4      # dx, dy, dx2, dy2
    
    def nan_mean(self, x, dim, keepdim=False):
        mask = torch.isnan(x)
        x_clean = torch.where(mask, torch.zeros_like(x), x)
        count = (~mask).float().sum(dim=dim, keepdim=keepdim)
        sum_ = x_clean.sum(dim=dim, keepdim=keepdim)
        return torch.where(count > 0, sum_ / count, torch.zeros_like(sum_))
    
    def nan_std(self, x, center=None, dim=None, keepdim=False):
        if center is None:
            center = self.nan_mean(x, dim=dim, keepdim=True)
        d = x - center
        return torch.sqrt(self.nan_mean(d * d, dim=dim, keepdim=keepdim))
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        B, T, N, C = x.shape
        
        # Select landmarks
        x = x[:, :, self.point_landmarks, :]  # (B, T, 118, 3)
        
        # Reference normalization: nose tip landmark (MediaPipe index 1)
        nose_orig = 1
        nose_in_sel = (self.point_landmarks == nose_orig).nonzero()
        nose_idx = nose_in_sel.item() if len(nose_in_sel) > 0 else 0
        
        ref_coords = x[:, :, nose_idx, :]  # (B, T, 3)
        ref_mean = self.nan_mean(ref_coords, dim=[1, 2], keepdim=True).unsqueeze(-1)  # (B, 1, 1, 1)
        ref_mean = torch.where(torch.isnan(ref_mean), torch.tensor(0.5, device=x.device, dtype=x.dtype), ref_mean)
        
        std = self.nan_std(x, center=ref_mean, dim=[1, 2], keepdim=True)
        x = (x - ref_mean) / (std + 1e-6)
        
        if self.max_len is not None and T > self.max_len:
            x = x[:, :self.max_len]
        T = x.shape[1]
        
        # Split appearance (x,y) and motion (velocity, acceleration)
        xy = x[..., :2]  # (B, T, 118, 2)
        
        # Velocity
        dx = torch.zeros_like(xy)
        if T > 1:
            dx[:, :-1] = xy[:, 1:] - xy[:, :-1]
        
        # Acceleration
        dx2 = torch.zeros_like(xy)
        if T > 2:
            dx2[:, :-2] = xy[:, 2:] - xy[:, :-2]
        
        # Flatten
        app = xy.reshape(B, T, -1)              # (B, T, 236)
        mot = torch.cat([dx, dx2], dim=-1).reshape(B, T, -1)  # (B, T, 472)
        
        app = torch.where(torch.isnan(app), torch.zeros_like(app), app)
        mot = torch.where(torch.isnan(mot), torch.zeros_like(mot), mot)
        
        return {'appearance': app, 'motion': mot, 'xy': xy, 'dx': dx, 'dx2': dx2}


class ECA(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (B, T, C)
        y = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y


class CausalDWConv1D(nn.Module):
    def __init__(self, channels, kernel_size=17, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size,
                                  groups=channels, dilation=dilation, padding=0, bias=False)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))
        x = self.dw_conv(x)
        x = x.transpose(1, 2)
        return x


class Conv1DBlock(nn.Module):
    def __init__(self, channels, kernel_size=17, drop_rate=0.2, expand_ratio=2):
        super().__init__()
        expanded = channels * expand_ratio
        self.expand = nn.Linear(channels, expanded)
        self.dwconv = CausalDWConv1D(expanded, kernel_size)
        self.bn = nn.BatchNorm1d(expanded)
        self.eca = ECA(expanded)
        self.project = nn.Linear(expanded, channels)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.act = nn.SiLU()
        self.use_skip = True
    
    def forward(self, x):
        skip = x
        x = self.act(self.expand(x))
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        x = self.eca(x)
        x = self.project(x)
        x = self.dropout(x)
        if self.use_skip:
            x = x + skip
        return x


class MotionGatedAttention(nn.Module):
    """Attention modulated by per-frame motion magnitude"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Motion gate: predicts per-frame attention scaling
        self.motion_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, num_heads),
            nn.Sigmoid()
        )
    
    def forward(self, x, motion_feat=None):
        B, T, C = x.shape
        
        # Standard MHSA
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, D)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        
        # Apply motion gating if provided
        if motion_feat is not None:
            # motion_feat: (B, T, D_mot) -> project to per-frame, per-head gates
            gate = self.motion_gate(motion_feat)  # (B, T, H)
            gate = gate.permute(0, 2, 1).unsqueeze(-1)  # (B, H, T, 1)
            # Modulate attention scores: frames with high motion get higher attention
            attn = attn * (1.0 + gate)  # (B, H, T, T) * (B, H, T, 1) -> broadcast
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, use_motion_gate=False):
        super().__init__()
        self.use_motion_gate = use_motion_gate
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        if use_motion_gate:
            self.attn = MotionGatedAttention(dim, num_heads, attn_dropout)
        else:
            self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.drop1 = nn.Dropout(drop_rate)
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.SiLU(),
            nn.Linear(dim * expand, dim),
            nn.Dropout(drop_rate)
        )
    
    def forward(self, x, motion_feat=None):
        residual = x
        x = self.norm1(x)
        if self.use_motion_gate:
            x = self.attn(x, motion_feat)
        else:
            x, _ = self.attn(x, x, x)
        x = self.drop1(x)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        return x


class LateDropout(nn.Module):
    def __init__(self, rate, start_step=0):
        super().__init__()
        self.rate = rate
        self.start_step = start_step
        self.dropout = nn.Dropout(rate)
        self.register_buffer('step', torch.tensor(0))
    
    def forward(self, x):
        if self.training and self.step >= self.start_step:
            x = self.dropout(x)
        if self.training:
            self.step += 1
        return x


class SignTransformer(nn.Module):
    """
    Two-stream Sign Language Transformer:
    - Appearance stream: static handshape/pose
    - Motion stream: velocity + acceleration
    - Motion-gated attention in 2nd transformer
    - Contrastive head for metric learning
    """
    def __init__(self, num_classes=100, dim=192, max_len=64, dropout_step=0, 
                 use_motion_gate=True, contrastive_dim=128, late_dropout_rate=0.5):
        super().__init__()
        self.preprocess = PreprocessLayer(max_len=max_len)
        self.use_motion_gate = use_motion_gate
        
        # Two-stream stems
        self.app_stem = nn.Linear(self.preprocess.app_dim, dim)
        self.mot_stem = nn.Linear(self.preprocess.mot_dim, dim)
        self.stem_bn = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        
        # Stream fusion
        self.fusion = nn.Linear(dim * 2, dim)
        
        # First 3 Conv1DBlocks (shared)
        self.conv_blocks1 = nn.Sequential(
            Conv1DBlock(dim, drop_rate=0.2),
            Conv1DBlock(dim, drop_rate=0.2),
            Conv1DBlock(dim, drop_rate=0.2),
        )
        
        # First TransformerBlock (no motion gate)
        self.transformer1 = TransformerBlock(dim, num_heads=4, expand=2, drop_rate=0.2)
        
        # Second 3 Conv1DBlocks
        self.conv_blocks2 = nn.Sequential(
            Conv1DBlock(dim, drop_rate=0.2),
            Conv1DBlock(dim, drop_rate=0.2),
            Conv1DBlock(dim, drop_rate=0.2),
        )
        
        # Second TransformerBlock (WITH motion gate)
        self.transformer2 = TransformerBlock(
            dim, num_heads=4, expand=2, drop_rate=0.2, 
            use_motion_gate=use_motion_gate
        )
        
        # Output heads
        self.top_conv = nn.Linear(dim, dim * 2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.late_dropout = LateDropout(late_dropout_rate, start_step=dropout_step)
        self.classifier = nn.Linear(dim * 2, num_classes)
        
        # Contrastive head (for metric learning)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(dim * 2, contrastive_dim),
            nn.LayerNorm(contrastive_dim)
        )
    
    def forward(self, x, return_embedding=False):
        # x: (B, T, 543, 3) or (T, 543, 3)
        feats = self.preprocess(x)  # dict with 'appearance', 'motion'
        app = feats['appearance']   # (B, T, 236)
        mot = feats['motion']       # (B, T, 472)
        
        # Two-stream encoding
        app = self.act(self.app_stem(app))
        mot = self.act(self.mot_stem(mot))
        
        # Fuse
        x = torch.cat([app, mot], dim=-1)  # (B, T, 2*dim)
        x = self.fusion(x)                 # (B, T, dim)
        x = x.transpose(1, 2)
        x = self.stem_bn(x)
        x = x.transpose(1, 2)
        
        # First conv blocks
        x = self.conv_blocks1(x)
        
        # First transformer (no motion gate)
        x = self.transformer1(x)
        
        # Second conv blocks
        x = self.conv_blocks2(x)
        
        # Second transformer (WITH motion gate)
        x = self.transformer2(x, motion_feat=mot)
        
        # Pooling
        x = self.top_conv(x)          # (B, T, 2*dim)
        x = x.transpose(1, 2)         # (B, 2*dim, T)
        x = self.global_pool(x).squeeze(-1)  # (B, 2*dim)
        
        # Contrastive embedding
        emb = self.contrastive_proj(x)  # (B, contrastive_dim)
        
        x = self.late_dropout(x)
        logits = self.classifier(x)
        
        if return_embedding:
            return logits, emb
        return logits
    
    def get_embedding(self, x):
        """Get contrastive embedding for a batch"""
        _, emb = self.forward(x, return_embedding=True)
        return F.normalize(emb, p=2, dim=1)


def load_tf_weights(model, tf_model_path):
    raise NotImplementedError(
        "Direct TF .h5 weight loading is not supported. "
        "Use PyTorch checkpoints (MODEL/checkpoints/*.pt) instead."
    )


if __name__ == '__main__':
    model = SignTransformer(num_classes=100, dim=192, use_motion_gate=True)
    x = torch.randn(2, 64, 543, 3)
    x[0, :, 100:, :] = float('nan')
    logits, emb = model(x, return_embedding=True)
    print(f'Logits: {logits.shape}, Embedding: {emb.shape}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')