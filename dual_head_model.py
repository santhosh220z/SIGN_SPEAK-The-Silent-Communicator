"""
Dual-Head Gesture Recognition Model
- Action Head: Temporal classification (SignTransformer) for dynamic gestures
- Static Head: Single-frame classification for static hand poses
- Shared backbone for efficiency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class StaticGestureHead(nn.Module):
    """Classify static hand pose from single-frame landmarks (21, 3)."""
    
    def __init__(self, input_dim: int = 63, hidden_dim: int = 256, num_classes: int = 50, dropout: float = 0.3):
        super().__init__()
        # Input: (B, 21, 3) -> flatten to (B, 63)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 21, 3) or (B, 63)
        if x.dim() == 3:
            x = x.flatten(1)  # (B, 63)
        x = self.input_proj(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layers(x)
        return self.classifier(x)


class ActionGestureHead(nn.Module):
    """Classify action gestures from temporal landmark sequences using SignTransformer."""
    
    def __init__(
        self, 
        num_classes: int = 100, 
        dim: int = 192, 
        max_len: int = 64,
        use_motion_gate: bool = True,
        contrastive_dim: int = 128,
    ):
        super().__init__()
        # Import SignTransformer components
        from model import SignTransformer
        self.transformer = SignTransformer(
            num_classes=num_classes,
            dim=dim,
            max_len=max_len,
            use_motion_gate=use_motion_gate,
            contrastive_dim=contrastive_dim,
        )
        
    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        # x: (B, T, 543, 3) or (B, T, 21, 3) - full body or hand-only
        return self.transformer(x, return_embedding=return_embedding)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer.get_embedding(x)


class DualHeadGestureModel(nn.Module):
    """
    Dual-head model for gesture recognition:
    - Static head: single-frame hand pose classification
    - Action head: temporal sequence classification
    
    Can be used in two modes:
    1. Static only: classify hand pose from single frame (fast, low latency)
    2. Action only: classify dynamic gesture from sequence (accurate for actions)
    3. Combined: both heads for comprehensive recognition
    """
    
    def __init__(
        self,
        num_static_classes: int = 50,    # Static gestures (ASL alphabet, numbers, etc.)
        num_action_classes: int = 100,   # Dynamic actions (sign language words)
        static_hidden_dim: int = 256,
        action_dim: int = 192,
        action_max_len: int = 64,
        use_motion_gate: bool = True,
        contrastive_dim: int = 128,
        shared_backbone: bool = False,   # Not implemented yet - separate for now
    ):
        super().__init__()
        
        self.num_static_classes = num_static_classes
        self.num_action_classes = num_action_classes
        
        # Static gesture head (single frame, 21 landmarks)
        self.static_head = StaticGestureHead(
            input_dim=63,  # 21 * 3
            hidden_dim=static_hidden_dim,
            num_classes=num_static_classes,
            dropout=0.3,
        )
        
        # Action gesture head (temporal, full body keypoints)
        self.action_head = ActionGestureHead(
            num_classes=num_action_classes,
            dim=action_dim,
            max_len=action_max_len,
            use_motion_gate=use_motion_gate,
            contrastive_dim=contrastive_dim,
        )
        
        # Gate to decide which head to use (learned or heuristic)
        self.use_action_gate = nn.Sequential(
            nn.Linear(63, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self, 
        static_input: Optional[torch.Tensor] = None,  # (B, 21, 3) or (B, 63)
        action_input: Optional[torch.Tensor] = None,  # (B, T, 543, 3)
        return_embedding: bool = False,
        mode: str = "auto",  # "static", "action", "both", "auto"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for dual-head model.
        
        Args:
            static_input: Single-frame hand landmarks (B, 21, 3)
            action_input: Temporal sequence (B, T, 543, 3)
            return_embedding: Whether to return embeddings
            mode: Which head(s) to use
            
        Returns:
            Dict with keys: 'static_logits', 'action_logits', 'static_emb', 'action_emb'
        """
        outputs = {}
        
        # Static head
        if mode in ("static", "both", "auto") and static_input is not None:
            static_logits = self.static_head(static_input)
            outputs['static_logits'] = static_logits
            
            if return_embedding:
                # Get penultimate layer features
                with torch.no_grad():
                    x = static_input.flatten(1) if static_input.dim() == 3 else static_input
                    x = F.relu(self.static_head.bn1(self.static_head.input_proj(x)))
                    x = self.static_head.layers(x)
                    outputs['static_emb'] = x
        
        # Action head
        if mode in ("action", "both", "auto") and action_input is not None:
            if return_embedding:
                action_logits, action_emb = self.action_head(action_input, return_embedding=True)
                outputs['action_emb'] = action_emb
            else:
                action_logits = self.action_head(action_input, return_embedding=False)
            outputs['action_logits'] = action_logits
            
        # Auto mode: if only one input provided, use that head
        if mode == "auto":
            if static_input is not None and action_input is None:
                mode = "static"
            elif action_input is not None and static_input is None:
                mode = "action"
            elif static_input is not None and action_input is not None:
                mode = "both"
            outputs['mode_used'] = mode
                
        return outputs
    
    def predict_static(self, landmarks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict static gesture from landmarks. Returns (logits, probs)."""
        logits = self.static_head(landmarks)
        probs = F.softmax(logits, dim=-1)
        return logits, probs
    
    def predict_action(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict action gesture from sequence. Returns (logits, probs)."""
        logits = self.action_head(sequence)
        probs = F.softmax(logits, dim=-1)
        return logits, probs
    
    def get_action_embedding(self, sequence: torch.Tensor) -> torch.Tensor:
        """Get action embedding for metric learning / retrieval."""
        return self.action_head.get_embedding(sequence)


# Example static gesture classes (ASL alphabet + numbers + common)
STATIC_GESTURE_CLASSES = [
    # ASL Alphabet (26)
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    # Numbers (10)
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    # Common gestures (14)
    'thumbs_up', 'thumbs_down', 'ok', 'peace', 'fist', 'open_hand',
    'pointing', 'call_me', 'rock_on', 'crossed_fingers', 'heart',
    'wave', 'clap', 'snap',
]  # Total: 50


# Example action gesture classes (dynamic)
ACTION_GESTURE_CLASSES = [
    # Add your action classes here - these should match your dataset
    # e.g., 'hello', 'thank_you', 'yes', 'no', 'please', 'sorry', etc.
]


if __name__ == '__main__':
    # Test model
    model = DualHeadGestureModel(
        num_static_classes=50,
        num_action_classes=100,
    )
    
    # Test static head
    static_input = torch.randn(2, 21, 3)
    out = model(static_input=static_input, mode="static")
    print(f"Static logits: {out['static_logits'].shape}")  # (2, 50)
    
    # Test action head
    action_input = torch.randn(2, 64, 543, 3)
    out = model(action_input=action_input, mode="action")
    print(f"Action logits: {out['action_logits'].shape}")  # (2, 100)
    
    # Test both
    out = model(static_input=static_input, action_input=action_input, mode="both")
    print(f"Both - Static: {out['static_logits'].shape}, Action: {out['action_logits'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    static_params = sum(p.numel() for p in model.static_head.parameters())
    action_params = sum(p.numel() for p in model.action_head.parameters())
    print(f"\nTotal params: {total_params:,}")
    print(f"  Static head: {static_params:,}")
    print(f"  Action head: {action_params:,}")