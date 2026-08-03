import torch
import torch.nn as nn
import torch.nn.functional as F


class PreprocessLayer(nn.Module):
    """
    PyTorch port of the TF Preprocess layer.
    Input: (B, T, 543, 3) - keypoints with NaN for missing
    Output: (B, T, 543*6) - normalized x,y + velocity + acceleration
    """
    def __init__(self, point_landmarks=None, max_len=64):
        super().__init__()
        self.max_len = max_len
        
        # Default: same 90 landmarks as TF model (LIP + LHAND + RHAND + NOSE + REYE + LEYE)
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
    
    def nan_mean(self, x, dim, keepdim=False):
        """Mean ignoring NaNs"""
        mask = torch.isnan(x)
        x_clean = torch.where(mask, torch.zeros_like(x), x)
        count = (~mask).float().sum(dim=dim, keepdim=keepdim)
        sum_ = x_clean.sum(dim=dim, keepdim=keepdim)
        return torch.where(count > 0, sum_ / count, torch.zeros_like(sum_))
    
    def nan_std(self, x, center=None, dim=None, keepdim=False):
        """Std ignoring NaNs"""
        if center is None:
            center = self.nan_mean(x, dim=dim, keepdim=True)
        d = x - center
        return torch.sqrt(self.nan_mean(d * d, dim=dim, keepdim=keepdim))
    
    def forward(self, x):
        """
        x: (B, T, 543, 3) or (T, 543, 3)
        """
        # Handle missing batch dim
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1, T, 543, 3)
        
        B, T, N, C = x.shape
        
        # Select relevant landmarks first
        x = x[:, :, self.point_landmarks, :]  # (B, T, 90, 3)
        
        # Use nose landmark (index 17 in original) as reference for normalization
        # Find where nose_idx (17) is in our selected landmarks
        nose_original_idx = 17
        nose_in_selected = (self.point_landmarks == nose_original_idx).nonzero()
        if len(nose_in_selected) > 0:
            nose_idx_in_selected = nose_in_selected.item()
        else:
            # If nose not in selected landmarks, use first landmark as reference
            nose_idx_in_selected = 0
        
        # Reference point for normalization (mean of reference landmark across time)
        # Gather reference landmark across all frames
        ref_coords = x[:, :, nose_idx_in_selected, :]  # (B, T, 3)
        ref_mean = self.nan_mean(ref_coords, dim=[1, 2], keepdim=True)  # (B, 1, 1)
        ref_mean = ref_mean.unsqueeze(-1)  # (B, 1, 1, 1)
        ref_mean = torch.where(torch.isnan(ref_mean), 
                               torch.tensor(0.5, device=x.device, dtype=x.dtype), 
                               ref_mean)
        
        # Normalize
        std = self.nan_std(x, center=ref_mean, dim=[1, 2], keepdim=True)  # (B, 1, 1, 3)
        x = (x - ref_mean) / (std + 1e-6)
        
        # Truncate to max_len
        if self.max_len is not None and T > self.max_len:
            x = x[:, :self.max_len]
        
        T = x.shape[1]
        
        # Keep only x, y coordinates (drop z)
        x = x[..., :2]  # (B, T, 90, 2)
        
        # Compute velocity (dx) and acceleration (dx2)
        # dx: frame t+1 - frame t
        dx = torch.zeros_like(x)
        if T > 1:
            dx[:, :-1] = x[:, 1:] - x[:, :-1]
        
        # dx2: frame t+2 - frame t
        dx2 = torch.zeros_like(x)
        if T > 2:
            dx2[:, :-2] = x[:, 2:] - x[:, :-2]
        
        # Reshape and concatenate: (B, T, 90*2 * 3) = (B, T, 540)
        x_flat = x.reshape(B, T, -1)          # (B, T, 180)
        dx_flat = dx.reshape(B, T, -1)        # (B, T, 180)
        dx2_flat = dx2.reshape(B, T, -1)      # (B, T, 180)
        
        x = torch.cat([x_flat, dx_flat, dx2_flat], dim=-1)  # (B, T, 540)
        
        # Replace NaN with 0
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        return x


class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (B, T, C)
        y = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = self.sigmoid(y)  # (B, 1, C)
        return x * y


class CausalDWConv1D(nn.Module):
    """Causal Depthwise Conv1D"""
    def __init__(self, channels, kernel_size=17, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size, 
                                  groups=channels, dilation=dilation, 
                                  padding=0, bias=False)
    
    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))
        x = self.dw_conv(x)
        x = x.transpose(1, 2)
        return x


class Conv1DBlock(nn.Module):
    """MBConv-style block with ECA"""
    def __init__(self, channels, kernel_size=17, drop_rate=0.2, expand_ratio=2):
        super().__init__()
        expanded = channels * expand_ratio
        
        self.expand = nn.Linear(channels, expanded)
        self.dwconv = CausalDWConv1D(expanded, kernel_size)
        self.bn = nn.BatchNorm1d(expanded)
        self.eca = ECA(expanded)
        self.project = nn.Linear(expanded, channels)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.act = nn.SiLU()  # Swish
        
        self.use_skip = True
    
    def forward(self, x):
        # x: (B, T, C)
        skip = x
        
        x = self.expand(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) for BatchNorm1d
        x = self.bn(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.eca(x)
        x = self.project(x)
        x = self.dropout(x)
        
        if self.use_skip:
            x = x + skip
        return x


class TransformerBlock(nn.Module):
    """Transformer block with MHSA + FFN"""
    def __init__(self, dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.drop1 = nn.Dropout(drop_rate)
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.SiLU(),
            nn.Linear(dim * expand, dim),
            nn.Dropout(drop_rate)
        )
    
    def forward(self, x, mask=None):
        # x: (B, T, C)
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.drop1(x)
        x = x + residual
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        return x


class LateDropout(nn.Module):
    """Dropout that starts after certain steps"""
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
    Sign Language Transformer matching the TF/Keras architecture:
    - Preprocess layer (NaN-robust normalization + temporal diffs)
    - Stem: Dense(192) + BN
    - 3x Conv1DBlock
    - TransformerBlock
    - 3x Conv1DBlock
    - TransformerBlock
    - GlobalAvgPool + LateDropout(0.8) + Classifier
    """
    def __init__(self, num_classes=100, dim=192, max_len=64, dropout_step=0):
        super().__init__()
        self.preprocess = PreprocessLayer(max_len=max_len)
        # After preprocess: (B, T, num_points*6)
        num_features = self.preprocess.num_points * 6
        
        self.stem = nn.Linear(num_features, dim)
        self.stem_bn = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        
        # First 3 Conv1DBlocks
        self.conv_blocks1 = nn.Sequential(
            Conv1DBlock(dim, drop_rate=0.2),
            Conv1DBlock(dim, drop_rate=0.2),
            Conv1DBlock(dim, drop_rate=0.2),
        )
        
        # First TransformerBlock
        self.transformer1 = TransformerBlock(dim, num_heads=4, expand=2, drop_rate=0.2)
        
        # Second 3 Conv1DBlocks
        self.conv_blocks2 = nn.Sequential(
            Conv1DBlock(dim, drop_rate=0.2),
            Conv1DBlock(dim, drop_rate=0.2),
            Conv1DBlock(dim, drop_rate=0.2),
        )
        
        # Second TransformerBlock
        self.transformer2 = TransformerBlock(dim, num_heads=4, expand=2, drop_rate=0.2)
        
        # Top conv + pooling
        self.top_conv = nn.Linear(dim, dim * 2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.late_dropout = LateDropout(0.8, start_step=dropout_step)
        self.classifier = nn.Linear(dim * 2, num_classes)
    
    def forward(self, x):
        # x: (B, T, 543, 3) or (T, 543, 3)
        x = self.preprocess(x)  # (B, T, 540)
        
        # Stem
        x = self.stem(x)  # (B, T, dim)
        x = x.transpose(1, 2)  # (B, dim, T) for BatchNorm1d
        x = self.stem_bn(x)
        x = x.transpose(1, 2)  # (B, T, dim)
        x = self.act(x)
        
        # First conv blocks
        x = self.conv_blocks1(x)
        
        # First transformer
        x = self.transformer1(x)
        
        # Second conv blocks
        x = self.conv_blocks2(x)
        
        # Second transformer
        x = self.transformer2(x)
        
        # Top conv
        x = self.top_conv(x)  # (B, T, dim*2)
        x = x.transpose(1, 2)  # (B, dim*2, T)
        x = self.global_pool(x).squeeze(-1)  # (B, dim*2)
        
        x = self.late_dropout(x)
        logits = self.classifier(x)
        return logits


def load_tf_weights(model, tf_model_path):
    """Load weights from TF/Keras model (requires manual mapping)"""
    # This would require h5py to read the Keras model
    # For now, train from scratch
    pass


if __name__ == '__main__':
    # Quick test
    model = SignTransformer(num_classes=100, dim=192)
    x = torch.randn(2, 64, 543, 3)
    x[0, :, 100:, :] = float('nan')  # Simulate missing landmarks
    logits = model(x)
    print(f'Output shape: {logits.shape}')  # Should be (2, 100)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')