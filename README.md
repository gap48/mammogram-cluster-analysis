# Multi-Modal Clustering Framework for Self-Supervised Mammography Analysis with Clinical Metadata Fusion

## Overview

This framework introduces a novel multi-modal contrastive learning approach for mammography analysis that synergistically combines image-based self-supervised learning with metadata-driven attention. Leveraging the SimCLR paradigm, the system employs rigorous data augmentation to learn robust representations from mammogram images while integrating clinical metadata through a multi-head attention mechanism to enhance diagnostic interpretability.

The framework enables meaningful cluster formation in the learned embedding space, clearly delineated via t-SNE visualizations. These clusters facilitate various downstream applications including precise pathology assessment, adaptive refinement of interpretability techniques, customized patient assignment, and identification of biases related to imaging devices and hospital protocols.

## Key Features

- **Self-Supervised Contrastive Learning**: Uses SimCLR with dual-view augmentation and NT-Xent loss
- **Metadata Fusion**: Incorporates clinical metadata (BI-RADS, breast density, pathology) through multi-head attention
- **Swin Transformer Backbone**: Utilizes pre-trained hierarchical vision transformer for feature extraction
- **t-SNE Visualization**: Provides clear delineation of embedding clusters for analysis
- **Clustering Applications**: Supports pathology, radiologist, patient, and device/hospital clustering

## Technical Architecture

The framework consists of several key components:
1. **Dual-View Augmentation**: Creates two augmented views of each mammogram as positive pairs
2. **Multi-Modal Data Processing**: Combines image features with categorical metadata
3. **Swin Transformer Encoder**: Extracts visual features from mammograms
4. **Metadata Attention Mechanism**: Processes clinical metadata through multi-head attention
5. **Projection Head**: Maps combined features to the final embedding space
6. **NT-Xent Loss**: Optimizes for similarity between positive pairs

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
python script.py \
  --root_dir /path/to/manifest-<ID> \
  --dicom_list /path/to/dcm_files.txt \
  --output_dir ./outputs \
  --epochs 10 \
  --batch_size 4 \
  --num_workers 2 \
  --train_cluster \
  --do_tsne

# Inference with a pre-trained model
python script.py \
  --root_dir /path/to/manifest-<ID> \
  --dicom_list /path/to/dcm_files.txt \
  --output_dir ./outputs \
  --checkpoint_path /path/to/best_simclr_model.pth \
  --do_tsne
