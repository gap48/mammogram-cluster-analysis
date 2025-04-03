# Setup and Usage Instructions

## Required Directory Structure

Before running the code, ensure the following directory structure:
```markdown
project_root/
├── mammogram_cluster.py                                # Main script
├── manifest-ZkhPvrLo5216730872708713142/
│   └── CBIS-DDSM/                          # Dataset directory
│       ├── mass_case_description_train_set.csv
│       ├── mass_case_description_test_set.csv
│       ├── calc_case_description_train_set.csv
│       ├── calc_case_description_test_set.csv
│       └── full mammogram images/
│           ├── Mass-Training_P_00001_LEFT_CC/
│           ├── Mass-Training_P_00001_LEFT_MLO/
│           └── ...
├── dcm_files.txt                           # List of DICOM paths
└── outputs/                                # Created automatically
├── checkpoints/
├── plots/
├── visualizations/
└── training_log.txt
```

## Training Modes

### Cluster (SimCLR) Training from Scratch

To train the model using full SimCLR contrastive learning:
```bash
python mammogram_cluster.py \
  --root_dir /path/to/manifest-<ID> \
  --dicom_list /path/to/dcm_files.txt \
  --output_dir ./outputs \
  --epochs 10 \
  --batch_size 4 \
  --num_workers 2 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --temperature 0.07 \
  --train_cluster \
  --do_tsne \
  --log_mode console
```

### Checkpoint-based Inference

To load a pretrained checkpoint for inference (without training from scratch):
```bash
python mammogram_cluster.py \
  --root_dir /path/to/manifest-<ID> \
  --dicom_list /path/to/dcm_files.txt \
  --output_dir ./outputs \
  --epochs 1 \
  --batch_size 4 \
  --num_workers 2 \
  --checkpoint_path /path/to/best_simclr_model.pth \
  --do_tsne \
  --log_mode console
```

 ### Visualization Only

 To generate a t-SNE visualization on test embeddings using a previously trained model:
```bash
 python mammogram_cluster.py \
  --root_dir /path/to/manifest-<ID> \
  --dicom_list /path/to/dcm_files.txt \
  --output_dir ./outputs \
  --do_tsne \
  --log_mode console
```

## Output Structure
```markdown
outputs/
├── checkpoints/
│   ├── best_simclr_model.pth     # Best model checkpoint based on training loss
│   └── final_simclr_model.pth    # Final model after training completion
├── plots/
│   └── loss_curve.png            # Training loss curves
├── visualizations/
│   └── combined_tsne_visualization.png    # t-SNE projection of test embeddings
└── simclr_log.txt                # Detailed logs of training progress (if log_mode=file)
```
