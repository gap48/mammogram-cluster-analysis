# Multi-Modal Clustering Framework for Self-Supervised Mammography Analysis with Clinical Metadata Fusion


The work implements a novel multi-modal contrastive learning architecture for mammography analysis that integrates image-based self-supervised learning with metadata-driven attention mechanisms. The approach leverages the SimCLR paradigm with significant extensions for medical imaging applications, establishing a robust representation learning pipeline for mammograms that unifies visual and clinical metadata features.

The architecture employs a dual-input contrastive learning strategy where each mammogram undergoes distinct augmentations via parameterized transformations (horizontal flips, rotations, scaling variations, and intensity perturbations). Simultaneously, categorical clinical metadata (BI-RADS classifications, breast density, pathology labels) is encoded through embedding layers and processed via a multi-head attention mechanism (4 heads, embedding dimension 128) that effectively models inter-feature dependencies.

Feature extraction utilizes a modified Swin Transformer (swin_base_patch4_window7_224) backbone with single-channel input adaptation for grayscale mammograms. The multi-modal fusion process concatenates the globalized image features with attention-pooled metadata embeddings before projection through a non-linear MLP (512 → 128 dimensions) with GELU activation and L2 normalization.

Training employs the NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss function with a temperature parameter of 0.07, optimizing similarity between positive pairs while maximizing distance to negative samples. The NT-Xent implementation addresses numerical stability issues through proper normalization and row-based maximum subtraction. Optimization utilizes AdamW with cosine annealing warm restarts for robust convergence.

The learned embedding space demonstrates emergent clustering properties visualized through t-SNE dimensionality reduction, revealing distinct grouping patterns that correspond to pathological classifications without explicit supervision. This behavior enables downstream applications including pathology assessment, radiologist interpretation standardization, patient cohort identification, and detection of acquisition protocol biases.

## Implementation Details

- **Image Augmentation Pipeline**: Implements comprehensive medical-specific transformations including controlled intensity perturbations suitable for preserving diagnostic features
- **Metadata Processing**: Creates numerically encoded representations for categorical features with explicit embedding layers
- **Encoder Architecture**: Utilizes hierarchical Swin Transformer features with adaptive average pooling for fixed-dimension representation
- **Multi-head Attention**: Implements scaled dot-product attention with 4 heads on metadata features to model cross-feature relationships
- **Training Protocol**: Employs gradient clipping (norm 1.0), warm restart scheduling, and configurable batch sizes with three-way (80/10/10) dataset splitting
- **Evaluation Methodology**: Implements clean embedding extraction for t-SNE visualization with pathology-based colorization

The implementation includes comprehensive data handling for DICOM files from the CBIS-DDSM dataset, synchronized metadata parsing, and optimized data loading through custom collation functions.

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--root_dir` | Required | Path to directory containing CBIS-DDSM CSV files |
| `--dicom_list` | Required | Path to a text file listing DICOM file paths |
| `--output_dir` | ./outputs | Directory to save checkpoints, results, and visualizations |
| `--batch_size` | 4 | Batch size for data loaders |
| `--num_workers` | 2 | Number of worker processes for data loading |
| `--epochs` | 1 | Number of epochs for SimCLR training |
| `--lr` | 1e-4 | Learning rate |
| `--weight_decay` | 1e-5 | Weight decay |
| `--temperature` | 0.07 | NT-Xent temperature for contrastive loss |
| `--log_mode` | console | Logging mode: 'console' for interactive display, 'file' for saving to output directory |
| `--do_tsne` | False | Flag to perform a t-SNE visualization on test embeddings |
| `--train_cluster` | False | Flag to perform full cluster (SimCLR) training from scratch |
| `--checkpoint_path` | None | Path to a checkpoint to load |

## Example Usage

```bash
# SimCLR Training
python mammogram_cluster.py \
  --root_dir /path/to/manifest-<ID> \
  --dicom_list /path/to/dcm_files.txt \
  --output_dir ./outputs \
  --epochs 10 \
  --batch_size 4 \
  --num_workers 2 \
  --train_cluster \
  --do_tsne

# Inference with a pre-trained model
python mammogram_cluster.py \
  --root_dir /path/to/manifest-<ID> \
  --dicom_list /path/to/dcm_files.txt \
  --output_dir ./outputs \
  --checkpoint_path /path/to/best_simclr_model.pth \
  --do_tsne
