import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split

import timm
from typing import Dict, Optional, List, Tuple

from sklearn.manifold import TSNE
from tqdm import tqdm

from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRange,
    ToTensor,
    Compose,
    Resize,
    RandFlip,
    RandRotate,
    RandZoom,
    RandGaussianNoise
)

#############################################################################
# 1) DATASET
#############################################################################

class DualViewTransform:
    """
    Generates two augmented versions of the same image for SimCLR.
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, image_path):
        view1 = self.base_transform(image_path)
        view2 = self.base_transform(image_path)
        return view1, view2


class MultiModalMammoDataset(Dataset):
    """
    Loads images + metadata. Returns two augmented views plus numeric-encoded metadata.
    """
    def __init__(self, metadata_df, transform=None):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.encodings = self._create_encodings()

    def _create_encodings(self):
        """Create numeric encodings for each categorical feature."""
        encodings = {}
        features = {
            'label': 'mass_calc',  
            'pathology': 'pathology',
            'subtlety': 'subtlety',
            'breast_density': 'breast_density',
            'assessment': 'assessment',
            'abnormality_type': 'abnormality_type'
        }
        for col, feature_name in features.items():
            unique_vals = sorted(self.metadata_df[col].unique())
            encodings[feature_name] = {val: idx for idx, val in enumerate(unique_vals)}

        return encodings

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        image_path = row['image_path']

  
        metadata = {
            'mass_calc': torch.tensor(self.encodings['mass_calc'][row['label']]),
            'pathology': torch.tensor(self.encodings['pathology'][row['pathology']]),
            'subtlety': torch.tensor(self.encodings['subtlety'][row['subtlety']]),
            'breast_density': torch.tensor(self.encodings['breast_density'][row['breast_density']]),
            'assessment': torch.tensor(self.encodings['assessment'][row['assessment']]),
            'abnormality_type': torch.tensor(self.encodings['abnormality_type'][row['abnormality_type']])
        }

        if self.transform:
            img1, img2 = self.transform(image_path)
        else:
            base_img = LoadImage(image_only=True)(image_path)
            base_img = torch.tensor(base_img[None, ...])  # [1, H, W]
            img1, img2 = base_img.clone(), base_img.clone()

        return (img1, img2, metadata)


def dual_view_collate(batch):
    """
    Collates a batch of (img1, img2, metadata).
    """
    img1_list = []
    img2_list = []
    metadata_list = []

    for (i1, i2, m) in batch:
        img1_list.append(i1)
        img2_list.append(i2)
        metadata_list.append(m)

    img1_tensor = torch.stack(img1_list)
    img2_tensor = torch.stack(img2_list)

    collated_metadata = {}
    for key in metadata_list[0].keys():
        collated_metadata[key] = torch.stack([m[key] for m in metadata_list])

    return img1_tensor, img2_tensor, collated_metadata


#############################################################################
# 2) DATALOADER CREATION
#############################################################################

def create_clustering_dataloaders(metadata_df, batch_size=4, num_workers=2, seed=42):

    base_transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensityRange(a_min=0, a_max=65535, b_min=0.0, b_max=1.0, clip=True),
        RandFlip(prob=0.5, spatial_axis=0),
        RandRotate(range_x=15, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3),
        RandGaussianNoise(prob=0.2),
        Resize((224, 224)),
        ToTensor()
    ])
    transform = DualViewTransform(base_transform)

    full_dataset = MultiModalMammoDataset(metadata_df, transform=transform)

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size   = int(0.1 * total_size)
    test_size  = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dual_view_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dual_view_collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dual_view_collate
    )

 
    print("\n[INFO] Feature Encodings:")
    for feature, mapping in full_dataset.encodings.items():
        print(f"  {feature}:")
        for val, idx in mapping.items():
            print(f"    {val} -> {idx}")

    return train_loader, val_loader, test_loader

#############################################################################
# 3) UTILS: METADATA PARSING
#############################################################################

def find_full_mammogram(dcm_files, patient_folder):
    """
    Finds 'full mammogram' DICOM file for a given patient folder name.
    """
    matching_files = []
    for dcm_path in dcm_files:
        if patient_folder in dcm_path and "full mammogram images" in dcm_path:
            matching_files.append(dcm_path)
    return matching_files[0] if matching_files else None


def process_dataframe(df, dcm_files, is_mass=True):
    """
    Process each CSV row, matching the relevant DICOM path for 'full mammogram'.
    """
    prefix = "Mass-Training" if is_mass else "Calc-Training"
    density_col = 'breast_density' if is_mass else 'breast density'
    data = []

    for _, row in df.iterrows():
        patient_id = row['patient_id'].split('_')[1]
        breast = row['left or right breast']
        view = row['image view']

        folder_pattern = f"{prefix}_P_{patient_id}_{breast}_{view}"
        full_image_path = find_full_mammogram(dcm_files, folder_pattern)
        if full_image_path:
            item = {
                'image_path': full_image_path,
                'label': 1 if is_mass else 0,
                'pathology': row['pathology'],
                'subtlety': row['subtlety'],
                'breast_density': row[density_col],
                'assessment': row['assessment'],
                'abnormality_type': row['abnormality type'],
                'patient_id': row['patient_id'],
                'breast': breast,
                'view': view
            }

            if is_mass:
                item['mass_shape'] = row['mass shape']
                item['mass_margins'] = row['mass margins']
            else:
                item['calc_type'] = row['calc type']
                item['calc_distribution'] = row['calc distribution']

            data.append(item)
    return data


def parse_ddsm_metadata(root_dir, dicom_list):
    """
    Combines the DDSM CSV metadata with the loaded DICOM paths.
    """
    if not os.path.exists(root_dir):
        raise ValueError(f"[ERROR] root_dir does not exist: {root_dir}")

    mass_train_csv = os.path.join(root_dir, 'mass_case_description_train_set.csv')
    mass_test_csv  = os.path.join(root_dir, 'mass_case_description_test_set.csv')
    calc_train_csv = os.path.join(root_dir, 'calc_case_description_train_set.csv')
    calc_test_csv  = os.path.join(root_dir, 'calc_case_description_test_set.csv')

    if not (os.path.isfile(mass_train_csv) and os.path.isfile(mass_test_csv) and
            os.path.isfile(calc_train_csv) and os.path.isfile(calc_test_csv)):
        raise ValueError("[ERROR] Missing one or more CSV files in root_dir.")

    mass_train = pd.read_csv(mass_train_csv)
    mass_test  = pd.read_csv(mass_test_csv)
    calc_train = pd.read_csv(calc_train_csv)
    calc_test  = pd.read_csv(calc_test_csv)

    all_data = []
    all_data.extend(process_dataframe(mass_train, dicom_list, is_mass=True))
    all_data.extend(process_dataframe(mass_test,  dicom_list, is_mass=True))
    all_data.extend(process_dataframe(calc_train, dicom_list, is_mass=False))
    all_data.extend(process_dataframe(calc_test,  dicom_list, is_mass=False))

    metadata_df = pd.DataFrame(all_data)
    print(f"[INFO] After matching CSV+DICOM, found {len(metadata_df)} records.")
    if len(metadata_df) > 0:
        print(metadata_df.head(3))

    return metadata_df

#############################################################################
# 4) MODEL DEFINITION
#############################################################################

class MultiModalSwinEncoder(nn.Module):

    def __init__(self, feature_dims: Dict[str, int], embed_dim: int = 128):
        super().__init__()
        self.swin = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=True,
            in_chans=1,
            features_only=True
        )
   
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 224, 224)
            dummy_output = self.swin(dummy_input)
            swin_output_dim = dummy_output[-1].shape[1]


        self.meta_embeddings = nn.ModuleDict({
            name: nn.Embedding(dim_size, 128) for name, dim_size in feature_dims.items()
        })

      
        self.fusion_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1)

 
        total_dim = swin_output_dim + 128
        self.projection = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, images: torch.Tensor, metadata: Dict[str, torch.Tensor]):
      
        swin_feats = self.swin(images)
        img_feats = swin_feats[-1]
        img_feats = F.adaptive_avg_pool2d(img_feats, 1).squeeze(-1).squeeze(-1)  # [B, C]

   
        meta_list = []
        for name, embed_layer in self.meta_embeddings.items():
            meta_list.append(embed_layer(metadata[name]))  # [B, 128]
        meta_tensor = torch.stack(meta_list, dim=0)         # [n_features, B, 128]
        fused_meta, _ = self.fusion_attn(meta_tensor, meta_tensor, meta_tensor)
        meta_pooled = fused_meta.mean(dim=0)               # [B, 128]

    
        combined = torch.cat([img_feats, meta_pooled], dim=1)
        embedding = self.projection(combined)
        embedding = F.normalize(embedding, dim=1)
        return embedding

#############################################################################
# 5) LOSS DEFINITION
#############################################################################

class NTXentLoss(nn.Module):
    """
    NT-Xent (InfoNCE) used in SimCLR:
      - Input: two batches z1, z2
      - Positive pair: (z1[i], z2[i])
      - Negatives: all other samples
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B, _ = z1.shape
        # Concat => [2B, D]
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)

        # Similarity matrix => [2B, 2B]
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Positive indices
        pos_idx = torch.arange(B, 2 * B, device=z.device)
        pos_idx = torch.cat([pos_idx, torch.arange(0, B, device=z.device)], dim=0)
        sim_pos = sim_matrix[torch.arange(2 * B), pos_idx]

        # Avoid numerical instability => subtract max from each row
        row_max = torch.max(sim_matrix, dim=1, keepdim=True)[0]
        sim_matrix_exp = torch.exp(sim_matrix - row_max)
        sim_sum = sim_matrix_exp.sum(dim=1) - torch.diagonal(sim_matrix_exp, 0)

        # NT-Xent => -log( exp(sim_pos - row_max) / sum_neg )
        loss = -torch.log(torch.exp(sim_pos - row_max.squeeze()) / sim_sum)
        return loss.mean()

#############################################################################
# 6) TRAIN-EVAL FUNCTION (Trainer)
#############################################################################

class SimCLRTrainer:
    """
    Handles the training loop for a multi-modal SimCLR model.
    """
    def __init__(
        self,
        encoder: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07
    ):
        self.encoder = encoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = NTXentLoss(temperature=temperature).to(device)
        self.optimizer = optim.AdamW(self.encoder.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

    def train_epoch(self, epoch, num_epochs):
        self.encoder.train()
        total_loss = 0.0
        batch_pbar = tqdm(
            self.train_loader,
            desc=f"[Epoch {epoch+1}/{num_epochs}]",
            leave=False,
            file=sys.stdout,
            mininterval=0.1
        )
        for batch_idx, (img1, img2, meta) in enumerate(batch_pbar):
            img1 = img1.cuda(non_blocking=True)
            img2 = img2.cuda(non_blocking=True)
            meta = {k: v.cuda(non_blocking=True) for k, v in meta.items()}

            self.optimizer.zero_grad()
            z1 = self.encoder(img1, meta)
            z2 = self.encoder(img2, meta)
            loss = self.criterion(z1, z2)
            loss.backward()

            nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.optimizer.step()

            # Cosine annealing scheduler
            current_iter = epoch + batch_idx / len(self.train_loader)
            self.scheduler.step(current_iter)

            total_loss += loss.item()
            batch_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def evaluate(self):
        self.encoder.eval()
        all_embeddings = []
        val_pbar = tqdm(
            self.val_loader,
            desc="[Val Evaluation]",
            leave=False,
            file=sys.stdout,
            mininterval=0.1
        )
        for img1, img2, meta in val_pbar:
            img1 = img1.cuda(non_blocking=True)
            meta = {k: v.cuda(non_blocking=True) for k, v in meta.items()}
            z1 = self.encoder(img1, meta)
            all_embeddings.append(z1.cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings

    def train(self, num_epochs=10, output_dir='.', log_mode='console'):
        """
        Full training loop.
        """
        if log_mode == 'file':
            log_path = os.path.join(output_dir, 'simclr_training_log.txt')
            sys.stdout = open(log_path, 'w', buffering=1)
            matplotlib.use('Agg')
            print(f"[INFO] Logging to file: {log_path}")

        print("[INFO] Starting SimCLR Training ...")
        best_loss = float('inf')

        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(epoch, num_epochs)
            val_embeddings = self.evaluate()
            print(f"\n[Epoch {epoch+1}/{num_epochs}] Avg Train Loss: {avg_loss:.4f} | Val Emb: {val_embeddings.shape}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt_path = os.path.join(output_dir, "best_simclr_model.pth")
                torch.save(self.encoder.state_dict(), ckpt_path)
                print(f"  [*] New best train loss: {best_loss:.4f}. Saved at {ckpt_path}.")

        final_path = os.path.join(output_dir, "final_simclr_model.pth")
        torch.save(self.encoder.state_dict(), final_path)
        print(f"[INFO] Training complete. Final model saved at {final_path}.")

        if log_mode == 'file':
            sys.stdout.close()

def visualize_combined_tsne(test_embeddings, metadata_list, output_dir, log_mode='console'):

    if torch.is_tensor(test_embeddings):
        embeddings_np = test_embeddings.numpy()
    else:
        embeddings_np = test_embeddings

   
    print("[INFO] Computing t-SNE projection...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=42
    )
    tsne_embeds = tsne.fit_transform(embeddings_np)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

   
    scatter1 = ax1.scatter(
        tsne_embeds[:, 0],
        tsne_embeds[:, 1],
        c=range(len(tsne_embeds)),  
        cmap='viridis',
        s=50,
        alpha=0.6
    )
    ax1.set_title('Natural Clusters in Latent Space', fontsize=14)
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)

    
    pathology_labels = [meta['pathology'].item() for meta in metadata_list]
    scatter2 = ax2.scatter(
        tsne_embeds[:, 0],
        tsne_embeds[:, 1],
        c=pathology_labels,
        cmap='coolwarm',
        s=50,
        alpha=0.6
    )
    ax2.set_title('Clusters by Pathology', fontsize=14)
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    plt.colorbar(scatter2, ax=ax2, label='Pathology')

    plt.tight_layout()

    tsne_path = os.path.join(output_dir, "combined_tsne_visualization.png")
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Combined t-SNE visualization saved to {tsne_path}")

    if log_mode == 'console':
        plt.show()
    else:
        plt.close()

#############################################################################
# 7) MAIN
#############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True,
                        help="Root directory containing CBIS-DDSM CSVs.")
    parser.add_argument('--dicom_list', type=str, required=True,
                        help="Path to a .txt file listing DICOM file paths.")
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help="Directory to store logs, checkpoints, etc.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size.")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of DataLoader workers.")
    parser.add_argument('--epochs', type=int, default=1,
                        help="Number of epochs for SimCLR training.")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="Weight decay.")
    parser.add_argument('--temperature', type=float, default=0.07,
                        help="NT-Xent temperature.")
    parser.add_argument('--log_mode', type=str, choices=['console','file'], default='console',
                        help="Log to console or file.")
    parser.add_argument('--do_tsne', action='store_true',
                        help="If set, perform a t-SNE on test embeddings.")
    parser.add_argument('--train_cluster', action='store_true',
                        help="If set, do full cluster (SimCLR) training from scratch.")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="Path to a checkpoint to load. If provided and --train_cluster is False, loads from checkpoint for inference.")
    # -----------------------------------------------------------------

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)


    if args.log_mode == 'file':
        log_path = os.path.join(args.output_dir, 'simclr_log.txt')
        sys.stdout = open(log_path, 'w', buffering=1)
        matplotlib.use('Agg')
        print(f"[INFO] Logging to file: {log_path}")

  
    if not os.path.isfile(args.dicom_list):
        print(f"[ERROR] Missing DICOM list file: {args.dicom_list}")
        sys.exit(1)
    with open(args.dicom_list, 'r') as f:
        dcm_files = [line.strip() for line in f]
    print(f"[INFO] Loaded {len(dcm_files)} DICOM file paths.")

    
    metadata_df = parse_ddsm_metadata(args.root_dir, dcm_files)
    if len(metadata_df) == 0:
        print("[ERROR] No matched records in metadata. Exiting.")
        sys.exit(1)

   
    train_loader, val_loader, test_loader = create_clustering_dataloaders(
        metadata_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42
    )


    full_dataset = train_loader.dataset.dataset
    feature_dims = {name: len(mapping) for name, mapping in full_dataset.encodings.items()}
    encoder = MultiModalSwinEncoder(feature_dims=feature_dims, embed_dim=128)

    trainer = SimCLRTrainer(
        encoder=encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature
    )

    if args.train_cluster:
        print("[INFO] Starting cluster (SimCLR) training from scratch...")
        trainer.train(num_epochs=args.epochs, output_dir=args.output_dir, log_mode=args.log_mode)
    else:

        if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
            trainer.encoder.load_state_dict(torch.load(args.checkpoint_path))
            print(f"[INFO] Loaded checkpoint from {args.checkpoint_path}")
        else:
            print("[WARNING] No checkpoint loaded (either not provided or not found). Using model's random weights.")
   
    print("\n[INFO] Performing final evaluation on test set ...")
    test_embeddings = []
    trainer.encoder.eval()

    test_pbar = tqdm(
        test_loader,
        desc="[Test Evaluation]",
        leave=False,
        file=sys.stdout,
        mininterval=0.1
    )

    for img1, img2, meta in test_pbar:
        img1 = img1.cuda(non_blocking=True)
        meta = {k: v.cuda(non_blocking=True) for k, v in meta.items()}
        with torch.no_grad():
            emb = trainer.encoder(img1, meta)
        test_embeddings.append(emb.cpu())

    test_embeddings = torch.cat(test_embeddings, dim=0)
    print(f"[INFO] Test Embeddings shape: {test_embeddings.shape}")


    if args.do_tsne:
        print("[INFO] Running combined t-SNE visualization...")
        metadata_list = []
        test_pbar = tqdm(test_loader, desc="[Collecting Data]")
        
        for _, _, meta in test_pbar:
            metadata_list.extend([{k: v[i] for k, v in meta.items()} 
                                for i in range(len(meta['pathology']))])
        
        visualize_combined_tsne(test_embeddings, metadata_list, args.output_dir, args.log_mode)

    print("[INFO] Done.")

    if args.log_mode == 'file':
        sys.stdout.close()

if __name__ == "__main__":
    main()

