
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO Trainer for CNN-based Image Classification (ImageNet-1k)
Supports:
  - ResNet-50 (torchvision)
  - YOLOv8-classification head (via ultralytics, optional)

Implements a GRPO-style reinforcement fine-tuning loop for classification:
  - Bandit reward (0/1 correctness) with optional shaped variants (top-k, margin)
  - Group-relative baseline: advantages computed against group mean/std
  - KL penalty to a frozen reference (pretrained) policy for stability
  - Optional auxiliary cross-entropy (alpha * CE) to stabilize early updates
  - Choice to update entire model or only the final classification head

NOTE on IoU reward:
  - IoU reward is meaningful for detection, not for ImageNet classification.
  - The reward interface includes an IoU option for extensibility. On ImageNet-1k, use bandit/top-k/margin.
  - If you later adapt this script to a detection dataset (e.g., COCO) with a detector model,
    you can plug in the IoU reward function provided below.

DATA:
  - ImageNet-1k cannot be downloaded programmatically due to licensing. Provide a local path.
  - Set --data_root to the directory containing 'train' and 'val' subfolders in standard layout.
    e.g., /path/to/imagenet/ (with /train and /val inside).

USAGE (examples):
  # ResNet-50, GRPO head-only for 300 steps
  python grpo_trainer.py --model resnet50 --data_root /datasets/imagenet \
      --batch_size 128 --group_size 4 --grpo_steps 300 --head_only \
      --reward bandit --aux_ce_coef 0.1 --kl_coef 0.02

  # YOLOv8n classification (requires: pip install ultralytics)
  python grpo_trainer.py --model yolov8n-cls --data_root /datasets/imagenet \
      --batch_size 128 --group_size 4 --grpo_steps 300 --head_only \
      --reward bandit

"""
import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

# Optional: ultralytics for YOLOv8 classification
_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO  # pip install ultralytics
    _YOLO_AVAILABLE = True
except Exception:
    pass


# ---------------------------- Utilities ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> list:
    """Compute the top-k accuracies for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return [r.item() for r in res]


# ---------------------------- Models ----------------------------

class ResNet50Classifier(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False, num_classes: int = 1000):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.net = torchvision.models.resnet50(weights=weights)
        # ensure classifier has correct num_classes
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, p in self.net.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad = False

    def forward(self, x):
        return self.net(x)


class YOLOv8ClsWrapper(nn.Module):
    """
    Wraps ultralytics YOLOv8 classification model to present a simple logits interface.
    Requires 'ultralytics' package and a YOLOv8-cls checkpoint name (e.g., 'yolov8n-cls.pt').
    """
    def __init__(self, ckpt: str = "yolov8n-cls.pt", freeze_backbone: bool = False, num_classes: int = 1000):
        super().__init__()
        if not _YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed. pip install ultralytics")
        self.yolo = YOLO(ckpt)
        # Attempt to adapt classifier head to desired classes if needed
        # Ultralytics classification models usually have a model.model[-1] as Linear head.
        # We will try to find a linear head and reset it if num_classes differs.
        # If mismatch occurs, we will raise with a helpful message.
        head = None
        for m in self.yolo.model.modules():
            if isinstance(m, nn.Linear):
                head = m
        if head is None:
            raise RuntimeError("Could not locate a Linear classification head in YOLOv8 model.")
        if head.out_features != num_classes:
            # replace last linear layer (best-effort; might require retraining head)
            in_feats = head.in_features
            new_head = nn.Linear(in_feats, num_classes, bias=True)
            # Attempt to replace
            replaced = False
            for name, module in self.yolo.model.named_modules():
                if module is head:
                    # We cannot directly assign into named_modules; we'll do a param-wise replace:
                    # Heuristic: search for attribute pointing to 'head' and swap.
                    for pname, pmodule in self.yolo.model.named_children():
                        # recursively try attributes
                        pass
            # Fallback: register as extra head applied after forward features.
            self.extra_head = new_head
            self.use_extra_head = True
        else:
            self.extra_head = None
            self.use_extra_head = False

        if freeze_backbone:
            for p in self.yolo.model.parameters():
                p.requires_grad = False
            # unfreeze last linear head if present
            if self.extra_head is not None:
                for p in self.extra_head.parameters():
                    p.requires_grad = True
            else:
                # try to unfreeze final linear layers
                for m in self.yolo.model.modules():
                    if isinstance(m, nn.Linear):
                        for p in m.parameters():
                            p.requires_grad = True

    def forward(self, x):
        # Ultralytics YOLO returns a Results object if you call self.yolo(x).
        # To get logits, we use the underlying model forward.
        # Most YOLOv8-cls models: forward(x) -> logits or tuple
        feats = self.yolo.model(x)
        if isinstance(feats, (list, tuple)):
            logits = feats[-1]
        else:
            logits = feats
        if self.use_extra_head:
            logits = self.extra_head(logits)
        return logits


def build_model(name: str, head_only: bool, num_classes: int = 1000):
    name = name.lower()
    if name == "resnet50":
        model = ResNet50Classifier(pretrained=True, freeze_backbone=head_only, num_classes=num_classes)
        ref_model = ResNet50Classifier(pretrained=True, freeze_backbone=True, num_classes=num_classes)  # frozen reference
        for p in ref_model.parameters():
            p.requires_grad = False
    elif name.startswith("yolov8") and name.endswith("-cls"):
        if not _YOLO_AVAILABLE:
            raise RuntimeError("Model requested is YOLOv8-cls but 'ultralytics' is not installed.")
        model = YOLOv8ClsWrapper(ckpt=name + ".pt", freeze_backbone=head_only, num_classes=num_classes)
        ref_model = YOLOv8ClsWrapper(ckpt=name + ".pt", freeze_backbone=True, num_classes=num_classes)
        for p in ref_model.parameters():
            p.requires_grad = False
    else:
        raise ValueError(f"Unsupported model '{name}'. Use 'resnet50' or 'yolov8n-cls'/'yolov8s-cls' etc.")
    return model, ref_model


# ---------------------------- GRPO Core ----------------------------

@dataclass
class GRPOConfig:
    group_size: int = 4
    kl_coef: float = 0.02
    aux_ce_coef: float = 0.1
    temperature: float = 1.0
    clip_ratio: float = 0.2   # PPO-style clip on logit changes approximated via KL (optional use)
    reward_type: str = "bandit"  # bandit | topk | margin | iou
    topk: int = 5  # for reward_type == "topk"
    margin_alpha: float = 1.0  # for reward_type == "margin"
    eps: float = 1e-8


def compute_rewards(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reward_type: str = "bandit",
    topk: int = 5,
    margin_alpha: float = 1.0
) -> torch.Tensor:
    """
    Compute reward per sample given logits and targets for classification.
    Args:
      logits: [B, C]
      targets: [B]
    Returns: rewards [B] in [0, 1] typically
    """
    with torch.no_grad():
        if reward_type == "bandit":
            preds = logits.argmax(dim=-1)
            return (preds == targets).float()

        elif reward_type == "topk":
            _, top = logits.topk(k=min(topk, logits.size(-1)), dim=-1)
            ok = (top == targets.view(-1, 1)).any(dim=-1)
            return ok.float()

        elif reward_type == "margin":
            # reward is sigmoid of margin between true logit and best wrong logit
            B, C = logits.shape
            true_logits = logits.gather(1, targets.view(-1, 1)).squeeze(1)
            # mask out the true class to find best wrong
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, targets.view(-1, 1), True)
            wrong_logits = logits.masked_fill(mask, float("-inf"))
            best_wrong = wrong_logits.max(dim=-1).values
            margin = true_logits - best_wrong
            return torch.sigmoid(margin_alpha * margin)

        elif reward_type == "iou":
            # IoU reward is not defined for classification logits; this is a placeholder.
            # If using a detection dataset/model, replace this with IoU computation
            # between predicted boxes and ground-truth boxes for each sample.
            raise RuntimeError("IoU reward requested but current task is classification. "
                               "Use 'bandit'/'topk'/'margin' for ImageNet-1k or adapt code for detection.")
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")


def kl_divergence_with_ref(logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """Compute mean KL divergence KL(pi || pi_ref) over batch."""
    p = F.log_softmax(logits, dim=-1).exp()
    log_p = torch.log(p + 1e-12)
    q = F.log_softmax(ref_logits, dim=-1).exp()
    log_q = torch.log(q + 1e-12)
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return kl.mean()


def sample_actions(logits: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample actions and return (actions, log_probs) given logits and temperature."""
    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    actions = dist.sample()
    logp = dist.log_prob(actions)
    return actions, logp


def grpo_step(
    model: nn.Module,
    ref_model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    cfg: GRPOConfig,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Performs one GRPO update with group sampling.
    Returns logging metrics.
    """
    model.train()
    ref_model.eval()

    B = images.size(0)
    G = cfg.group_size

    # Repeat images/targets G times to get group samples
    images_rep = images.repeat_interleave(G, dim=0)  # [B*G, ...]
    targets_rep = targets.repeat_interleave(G, dim=0)

    logits = model(images_rep)  # [B*G, C]
    with torch.no_grad():
        ref_logits = ref_model(images_rep)

    # Sample actions and get their log probs under current policy
    actions, logp = sample_actions(logits, temperature=cfg.temperature)  # [B*G]
    # Compute rewards per sample (for classification; IoU not supported here)
    # For reward computation we want per-sample logits; reward compares action to targets
    # Option A: Reward based on sampled action correctness
    # Option B: Reward based on greedy prediction (more stable). We'll do based on sampled action.
    rewards = (actions == targets_rep).float() if cfg.reward_type == "bandit" else \
              compute_rewards(logits, targets_rep, cfg.reward_type, cfg.topk, cfg.margin_alpha)

    # Group-wise advantage: reshape to [B, G]
    rewards_group = rewards.view(B, G)
    # Standardize within group
    group_mean = rewards_group.mean(dim=1, keepdim=True)
    group_std = rewards_group.std(dim=1, keepdim=True) + cfg.eps
    advantages = (rewards_group - group_mean) / group_std
    advantages = advantages.view(B * G)

    # Policy gradient loss (REINFORCE with group-relative baseline)
    # L_pg = - E[ advantage * log_prob(a) ]
    pg_loss = -(advantages.detach() * logp).mean()

    # KL penalty to reference policy (stabilize updates)
    kl = kl_divergence_with_ref(logits, ref_logits)
    kl_loss = cfg.kl_coef * kl

    # Optional auxiliary CE to true label using current logits (stabilization)
    if cfg.aux_ce_coef > 0.0:
        ce_loss = F.cross_entropy(logits, targets_rep)
        aux_loss = cfg.aux_ce_coef * ce_loss
    else:
        aux_loss = torch.tensor(0.0, device=logits.device)

    loss = pg_loss + kl_loss + aux_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    with torch.no_grad():
        # Report greedy accuracy on the original batch (not repeated)
        logits_eval = model(images)
        top1, top5 = accuracy(logits_eval, targets, topk=(1, 5))

    return {
        "loss": float(loss.item()),
        "pg_loss": float(pg_loss.item()),
        "kl": float(kl.item()),
        "aux_ce": float(aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss),
        "top1": float(top1),
        "top5": float(top5),
    }


# ---------------------------- Data ----------------------------

def build_imagenet_loaders(data_root: str, batch_size: int, num_workers: int = 8, img_size: int = 224):
    """
    Build standard ImageNet train/val dataloaders.
    NOTE: torchvision does NOT download ImageNet. You must provide local path.
    Expected structure:
      data_root/train/<class>/<images>
      data_root/val/<class>/<images>   (or ImageFolder-compatible layout)
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"--data_root '{data_root}' does not exist. "
                                "Please point to your ImageNet-1k directory containing 'train' and 'val'.")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    train_set = torchvision.datasets.ImageFolder(train_dir, transform=train_tf)
    val_set = torchvision.datasets.ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, device: str = "cuda") -> Dict[str, float]:
    model.eval()
    total, correct1, correct5 = 0, 0, 0
    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        maxk = 5
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct1 += correct[:1].reshape(-1).float().sum().item()
        correct5 += correct[:5].reshape(-1).float().sum().item()
        total += targets.size(0)
    top1 = 100.0 * correct1 / total
    top5 = 100.0 * correct5 / total
    return {"val_top1": top1, "val_top5": top5}


# ---------------------------- Main ----------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO Trainer for CNNs on ImageNet-1k")
    parser.add_argument("--model", type=str, default="resnet50",
                        help="resnet50 | yolov8n-cls | yolov8s-cls | ... (requires ultralytics)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to ImageNet-1k root containing 'train' and 'val' folders")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)

    # GRPO config
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--aux_ce_coef", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reward", type=str, default="bandit", choices=["bandit", "topk", "margin", "iou"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--margin_alpha", type=float, default=1.0)
    parser.add_argument("--grpo_steps", type=int, default=300, help="Number of GRPO update steps")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--head_only", action="store_true", help="Update only classification head")
    parser.add_argument("--eval_interval", type=int, default=50, help="Validate every N GRPO steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = args.device
    train_loader, val_loader = build_imagenet_loaders(args.data_root, batch_size=args.batch_size,
                                                      num_workers=args.num_workers, img_size=args.img_size)

    # Build model and frozen reference
    model, ref_model = build_model(args.model, head_only=args.head_only, num_classes=1000)
    model.to(device)
    ref_model.to(device)

    # Optimizer (only params that require grad)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=args.weight_decay)

    cfg = GRPOConfig(
        group_size=args.group_size,
        kl_coef=args.kl_coef,
        aux_ce_coef=args.aux_ce_coef,
        temperature=args.temperature,
        reward_type=args.reward,
        topk=args.topk,
        margin_alpha=args.margin_alpha
    )

    step = 0
    best_top1 = -1.0

    # Simple infinite iterator over train loader
    train_iter = iter(train_loader)

    while step < args.grpo_steps:
        try:
            images, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, targets = next(train_iter)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logs = grpo_step(model, ref_model, images, targets, optim, cfg, device=device)

        step += 1
        if step % 10 == 0:
            print(f"[Step {step:04d}] loss={logs['loss']:.4f} pg={logs['pg_loss']:.4f} "
                  f"kl={logs['kl']:.4f} aux={logs['aux_ce']:.4f} "
                  f"top1={logs['top1']:.2f} top5={logs['top5']:.2f}", flush=True)

        if step % args.eval_interval == 0 or step == args.grpo_steps:
            val_logs = evaluate(model, val_loader, device=device)
            print(f"  >> VAL @ step {step}: top1={val_logs['val_top1']:.2f} top5={val_logs['val_top5']:.2f}", flush=True)
            if val_logs["val_top1"] > best_top1:
                best_top1 = val_logs["val_top1"]
                save_path = f"best_{args.model}_grpo.pt"
                torch.save({"model": model.state_dict(),
                            "config": vars(args),
                            "val_top1": best_top1}, save_path)
                print(f"  >> Saved checkpoint: {save_path} (best top1={best_top1:.2f})", flush=True)


if __name__ == "__main__":
    main()
