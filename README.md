
# GRPO Trainer for CNNs on ImageNet-1k (ResNet-50 & YOLOv8-CLS)

This is a **complete PyTorch GRPO trainer** for *CNN-based* image classification on **ImageNet-2012 (1k classes)**.
It supports:
- **ResNet-50** (torchvision)
- **YOLOv8 classification** variants (via `ultralytics`, optional)

It implements **Group Relative Policy Optimization (GRPO)** style fine-tuning for classification as a **contextual bandit**:
- Sample *K* actions per image from the policy (softmax over 1000 classes)
- Compute **group-relative advantages**: normalize rewards within each group
- Update with policy gradient + **KL penalty** to a frozen reference policy
- Optional **auxiliary cross-entropy** stabilization
- **Head-only** or **full-model** updates

> **Note on IoU rewards**: IoU is meaningful for *detection/segmentation*, not ImageNet classification.  
> The code ships a placeholder IoU reward function so you can later adapt this to a detection model/dataset (e.g., YOLOv8-detect on COCO). For ImageNet, use `--reward bandit|topk|margin`.

---

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # pick the right CUDA/CPU build
# Optional for YOLOv8 classification:
pip install ultralytics
```

## Data (ImageNet-1k)

**ImageNet cannot be auto-downloaded.** Place the dataset locally with the standard layout:
```
/path/to/imagenet/
  train/
    n01440764/
      *.JPEG
    ...
  val/
    n01440764/
      *.JPEG
    ...
```

Then run with `--data_root /path/to/imagenet`.

---

## Quickstart

### ResNet-50 (head-only GRPO, 300 steps)
```bash
python grpo_trainer.py --model resnet50 --data_root /datasets/imagenet \
  --batch_size 128 --group_size 4 --grpo_steps 300 --head_only \
  --reward bandit --aux_ce_coef 0.1 --kl_coef 0.02
```

### ResNet-50 (full-model GRPO, margin reward)
```bash
python grpo_trainer.py --model resnet50 --data_root /datasets/imagenet \
  --batch_size 128 --group_size 4 --grpo_steps 300 \
  --reward margin --margin_alpha 1.0 --aux_ce_coef 0.05 --kl_coef 0.02
```

### YOLOv8n classification (requires `ultralytics`)
```bash
python grpo_trainer.py --model yolov8n-cls --data_root /datasets/imagenet \
  --batch_size 128 --group_size 4 --grpo_steps 300 --head_only \
  --reward topk --topk 5
```

> If the YOLOv8 checkpoint's classification head doesn't match 1000 classes, the script attaches an extra linear head.

---

## Options

- `--model`: `resnet50` | `yolov8n-cls` | `yolov8s-cls` … (for YOLOv8 classification you need `ultralytics` installed)
- `--data_root`: path to ImageNet-1k root with `train/` and `val/` subfolders
- `--batch_size`, `--num_workers`, `--img_size`
- **GRPO config**:
  - `--group_size` (K samples per image; default 4)
  - `--kl_coef` (KL penalty to frozen reference; default 0.02)
  - `--aux_ce_coef` (aux cross-entropy weight; default 0.1)
  - `--temperature` (sampling temperature; default 1.0)
  - `--reward` (`bandit` | `topk` | `margin` | `iou`*)
  - `--topk` (for `topk` reward; default 5)
  - `--margin_alpha` (for `margin` reward; default 1.0)
- `--grpo_steps` (number of GRPO updates; default 300)
- `--lr`, `--weight_decay`
- `--head_only` (freeze everything except classifier head)
- `--eval_interval` (validate every N steps)
- `--seed`

\* IoU reward raises on ImageNet classification. Use it when you adapt the script to a detector + detection dataset.

---

## How it works (high level)

For each batch:
1. **Group sampling**: replicate each image **K** times and forward → logits. Sample an action (class) for each replicate.
2. **Reward**: compute per-sample reward (e.g., 1 if sampled class == ground-truth).
3. **Group-relative advantage**: standardize rewards within each group (per image).
4. **Loss**: 
   - Policy gradient: `-E[ advantage * log π(a|x) ]`  
   - KL penalty: `β * KL(π || π_ref)` where π_ref is a frozen copy of the pretrained model  
   - (Optional) auxiliary CE: `α * CE(logits, target)` on the repeated batch for stability
5. **Step** optimizer; periodically evaluate on the validation set (Top-1/Top-5).

---

## Extending to Detection (IoU Reward)

To support **YOLOv8-detect + COCO** (or your dataset):
- Replace the model builder to load a detection model and return predicted boxes.
- Implement `compute_rewards(..., reward_type="iou")` that:
  - For each image, compares predicted boxes to GT boxes and uses max IoU (or average IoU for matched pairs) as reward.
  - Optionally add format/consistency rewards (e.g., valid box count, NMS quality).
- Keep the same group-relative update and KL-to-reference scheme.

This file already has an IoU path placeholder that intentionally raises on classification, so you won't silently get nonsense rewards.

---

## Checkpoints

On each validation improvement, the trainer saves `best_<model>_grpo.pt` with:
- `model.state_dict()`
- CLI config
- Best Top-1 accuracy

---

## Notes & Tips

- Start with **head-only** GRPO for stability, then unfreeze more layers if improvements stall.
- Increase `group_size` to reduce variance (with more compute).
- Use `margin` reward if you need a denser signal than 0/1 correctness.
- Tune `kl_coef` to prevent policy drift; too high will freeze learning, too low may destabilize.
- `aux_ce_coef` can be annealed from 0.2 → 0.0 over steps if you want a purer RL objective later.

---

## License

This trainer is provided as-is under the MIT license.
