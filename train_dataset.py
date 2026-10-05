import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter


class SignLanguageDataset(Dataset):
    """Dataset for sign language keypoints (64, 543, 3)"""
    
    def __init__(self, split='train', max_len=64, dataset='nslt100', 
                 min_samples_per_class=2, oversample=False):
        self.max_len = max_len
        self.keypoint_dir = Path('dataset/keypoints')
        self.dataset = dataset
        self.split = split

        if dataset == 'nslt100':
            self._load_nslt('dataset/nslt_100.json')
        elif dataset == 'asl':
            self.keypoint_dir = Path('dataset/keypoints_asl')
            self._load_nslt('dataset/asl_labels.json')
        elif dataset == 'nslt300':
            self._load_nslt('dataset/nslt_300.json')
        elif dataset == 'nslt1000':
            self._load_nslt('dataset/nslt_1000.json')
        elif dataset == 'nslt2000':
            self._load_nslt('dataset/nslt_2000.json')
        elif dataset == 'wlasl':
            self._load_wlasl(min_samples_per_class)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Build class weights for imbalanced sampling
        self.class_counts = Counter([label for _, label in self.samples])
        self.num_classes = max(self.class_counts.keys()) + 1 if self.class_counts else 0
        
        if oversample and split == 'train':
            self._setup_oversampling()
    
    def _load_nslt(self, json_path):
        with open(json_path) as f:
            annos = json.load(f)
        
        kp_ids = {f.stem for f in self.keypoint_dir.glob('*.npy')}
        
        self.samples = []
        for vid, info in annos.items():
            if info['subset'] == self.split:
                label = info['action'][0]  # class index
                kp_path = self.keypoint_dir / f'{vid}.npy'
                if kp_path.exists():
                    self.samples.append((kp_path, label))
        
        print(f"[NSLT] {self.split}: {len(self.samples)} samples with keypoints")
    
    def _load_wlasl(self, min_samples_per_class):
        with open('dataset/WLASL_v0.3.json') as f:
            wlasl = json.load(f)
        
        with open('dataset/wlasl_class_list.txt') as f:
            class_list = {line.strip().split('\t')[1]: int(line.strip().split('\t')[0]) 
                          for line in f}
        
        kp_ids = {f.stem for f in self.keypoint_dir.glob('*.npy')}
        
        # Count samples per class first
        class_samples = {}
        for gloss_entry in wlasl:
            gloss = gloss_entry['gloss']
            if gloss not in class_list:
                continue
            label = class_list[gloss]
            if label not in class_samples:
                class_samples[label] = []
            for inst in gloss_entry['instances']:
                if inst['split'] == self.split and inst['video_id'] in kp_ids:
                    class_samples[label].append(self.keypoint_dir / f'{inst["video_id"]}.npy')
        
        # Filter classes with enough samples
        self.samples = []
        for label, paths in class_samples.items():
            if len(paths) >= min_samples_per_class:
                for p in paths:
                    self.samples.append((p, label))
        
        print(f"[WLASL] {self.split}: {len(self.samples)} samples from {len(class_samples)} classes "
              f"(min {min_samples_per_class} samples/class)")
    
    def _setup_oversampling(self):
        """Create weighted sampler for balanced training"""
        class_weights = {cls: 1.0 / count for cls, count in self.class_counts.items()}
        sample_weights = [class_weights[label] for _, label in self.samples]
        self.sample_weights = torch.DoubleTensor(sample_weights)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        kp_path, label = self.samples[idx]
        keypoints = np.load(kp_path).astype(np.float32)  # (T, 543, 3)
        
        # Handle variable length
        T = keypoints.shape[0]
        if T > self.max_len:
            # Random crop for augmentation during training
            if self.split == 'train':
                start = np.random.randint(0, T - self.max_len + 1)
                keypoints = keypoints[start:start + self.max_len]
            else:
                keypoints = keypoints[:self.max_len]
        elif T < self.max_len:
            pad = np.full((self.max_len - T, 543, 3), np.nan, dtype=np.float32)
            keypoints = np.concatenate([keypoints, pad], axis=0)
        
        x = torch.from_numpy(keypoints)  # (64, 543, 3)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def get_dataloaders(dataset='nslt100', batch_size=64, max_len=64, 
                    num_workers=2, persistent_workers=True, oversample=True, min_samples_per_class=2):
    """Create train/val/test dataloaders"""
    
    train_ds = SignLanguageDataset('train', max_len, dataset, 
                                    min_samples_per_class, oversample)
    val_ds = SignLanguageDataset('val', max_len, dataset, 
                                  min_samples_per_class, False)
    test_ds = SignLanguageDataset('test', max_len, dataset, 
                                   min_samples_per_class, False)
    
    # Weighted sampler for training
    train_sampler = None
    shuffle = True
    if oversample and hasattr(train_ds, 'sample_weights'):
        train_sampler = WeightedRandomSampler(
            train_ds.sample_weights, len(train_ds.sample_weights), replacement=True
        )
        shuffle = False  # sampler handles shuffling
    
    # Use persistent_workers only if num_workers > 0
    use_persistent = persistent_workers and num_workers > 0
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, 
                              sampler=train_sampler, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=use_persistent)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=use_persistent)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=use_persistent)
    
    return train_loader, val_loader, test_loader, train_ds.num_classes