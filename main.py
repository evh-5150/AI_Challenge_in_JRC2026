
import os
import sys
import glob
import csv
import time
import json
import copy
import random
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


DEFAULT_DIRS = {
    'images': './segmentation/org',
    'masks': './segmentation/label',
    'outputs': './outputs/main',
}



AUGMENTATION_PARAMS = {
    'resize': {'height': 256, 'width': 256},
    'affine': {
        'translate_percent': {'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
        'scale': (0.9, 1.1),
        'rotate': (-10, 10),
        'shear': 0.0,
        'p': 0.7,
    },
    'random_brightness_contrast': {'p': 0.2},
    'grid_distortion': {'num_steps': 5, 'distort_limit': 0.1, 'p': 0.3},
    'normalize': {'mean': (0.5,), 'std': (0.5,)},
}



LUNG_LABEL_VALUE = 85

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_transform(img_size, aug_mode, is_train):
    ops = [A.Resize(img_size, img_size)]
    if is_train:
        if aug_mode in ('geometric', 'full'):
            ops.append(A.Affine(
                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                scale=(0.9, 1.1), rotate=(-10, 10), shear=0.0, p=0.7,
            ))
            ops.append(A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3))
        if aug_mode in ('intensity', 'full'):
            ops.append(A.RandomBrightnessContrast(p=0.2))
    ops.append(A.Normalize(mean=(0.5,), std=(0.5,)))
    ops.append(ToTensorV2())
    return A.Compose(ops)


class JSRTDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        if len(image_paths) != len(mask_paths):
            raise ValueError('The number of image and mask paths must match.')
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f'Cannot read image: {image_path}')
        if mask is None:
            raise FileNotFoundError(f'Cannot read mask: {mask_path}')
        label = (mask == LUNG_LABEL_VALUE).astype(np.float32)
        if not np.any(label):
            raise ValueError(f'Mask does not contain label value {LUNG_LABEL_VALUE}: {mask_path}')
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        if not torch.is_tensor(image):
            image = torch.from_numpy(image).unsqueeze(0)
        if not torch.is_tensor(label):
            label = torch.from_numpy(label)
        return image.float(), label.unsqueeze(0).float(), image_path


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.main(x) + self.skip(x))


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64, block=ConvBlock):
        super().__init__()
        c = base_ch
        self.pool = nn.MaxPool2d(2)
        self.enc1 = block(in_ch, c)
        self.enc2 = block(c, c * 2)
        self.enc3 = block(c * 2, c * 4)
        self.enc4 = block(c * 4, c * 8)
        self.bridge = block(c * 8, c * 16)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = block(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = block(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = block(c * 2, c)
        self.mask_head = nn.Conv2d(c, out_ch, 1)

    @staticmethod
    def up_and_cat(up_layer, current, skip):
        y = up_layer(current)
        if y.shape[2:] != skip.shape[2:]:
            y = torch.nn.functional.interpolate(y, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([skip, y], dim=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bridge(self.pool(s4))
        d4 = self.dec4(self.up_and_cat(self.up4, b, s4))
        d3 = self.dec3(self.up_and_cat(self.up3, d4, s3))
        d2 = self.dec2(self.up_and_cat(self.up2, d3, s2))
        d1 = self.dec1(self.up_and_cat(self.up1, d2, s1))
        return self.mask_head(d1)


class ResUNet(UNet):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch, block=ResBlock)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, intermediate_channels):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(gate_channels, intermediate_channels, 1, bias=True), nn.BatchNorm2d(intermediate_channels))
        self.w_x = nn.Sequential(nn.Conv2d(skip_channels, intermediate_channels, 1, bias=True), nn.BatchNorm2d(intermediate_channels))
        self.psi = nn.Sequential(nn.Conv2d(intermediate_channels, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = torch.nn.functional.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x * self.psi(self.relu(self.w_g(g) + self.w_x(x)))


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        c = base_ch
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_ch, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.bridge = ConvBlock(c * 8, c * 16)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.att4 = AttentionGate(c * 8, c * 8, c * 4)
        self.dec4 = ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.att3 = AttentionGate(c * 4, c * 4, c * 2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.att2 = AttentionGate(c * 2, c * 2, c)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.att1 = AttentionGate(c, c, max(1, c // 2))
        self.dec1 = ConvBlock(c * 2, c)
        self.mask_head = nn.Conv2d(c, out_ch, 1)

    @staticmethod
    def cat_with_shape_check(a, b):
        if a.shape[2:] != b.shape[2:]:
            a = torch.nn.functional.interpolate(a, size=b.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([b, a], dim=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bridge(self.pool(s4))
        u4 = self.up4(b)
        d4 = self.dec4(self.cat_with_shape_check(u4, self.att4(u4, s4)))
        u3 = self.up3(d4)
        d3 = self.dec3(self.cat_with_shape_check(u3, self.att3(u3, s3)))
        u2 = self.up2(d3)
        d2 = self.dec2(self.cat_with_shape_check(u2, self.att2(u2, s2)))
        u1 = self.up1(d2)
        d1 = self.dec1(self.cat_with_shape_check(u1, self.att1(u1, s1)))
        return self.mask_head(d1)


def build_model(model_name, base_ch):
    if model_name == 'unet':
        return UNet(base_ch=base_ch)
    if model_name == 'attention_unet':
        return AttentionUNet(base_ch=base_ch)
    if model_name == 'resunet':
        return ResUNet(base_ch=base_ch)
    raise ValueError(f'Unknown model: {model_name}')


def dice_loss_from_logits(logits, target, smooth=1e-6):
    prob = torch.sigmoid(logits)
    prob = prob.view(prob.shape[0], -1)
    target = target.view(target.shape[0], -1)
    dice = (2.0 * (prob * target).sum(dim=1) + smooth) / (prob.sum(dim=1) + target.sum(dim=1) + smooth)
    return 1.0 - dice.mean()


def make_loss(loss_name):
    bce = nn.BCEWithLogitsLoss()

    def loss_fn(logits, target):
        if loss_name == 'bce':
            return bce(logits, target)
        if loss_name == 'dice':
            return dice_loss_from_logits(logits, target)
        if loss_name == 'bce_dice':
            return bce(logits, target) + dice_loss_from_logits(logits, target)
        raise ValueError(f'Unknown loss: {loss_name}')

    return loss_fn


def binary_metrics(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    dice = (2.0 * tp / denom) if denom > 0 else 1.0
    union = np.logical_or(pred, target).sum()
    iou = (tp / union) if union > 0 else 1.0
    return float(dice), float(iou)


def surface_distances(mask_a, mask_b):
    if not SCIPY_AVAILABLE:
        return np.array([np.nan], dtype=np.float32)
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if a.sum() == 0 or b.sum() == 0:
        return np.array([np.nan], dtype=np.float32)
    kernel = np.ones((3, 3), dtype=np.uint8)
    a_border = a ^ cv2.erode(a.astype(np.uint8), kernel, iterations=1).astype(bool)
    b_border = b ^ cv2.erode(b.astype(np.uint8), kernel, iterations=1).astype(bool)
    if a_border.sum() == 0 or b_border.sum() == 0:
        return np.array([np.nan], dtype=np.float32)
    dt_b = distance_transform_edt(~b_border)
    dt_a = distance_transform_edt(~a_border)
    return np.concatenate([dt_b[a_border], dt_a[b_border]]).astype(np.float32)


def hd95_assd(pred, target):
    d = surface_distances(pred, target)
    if np.isnan(d).all():
        return float('nan'), float('nan')
    return float(np.nanpercentile(d, 95)), float(np.nanmean(d))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def summarize(values):
    arr = np.array(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return {'mean': float('nan'), 'sd': float('nan'), 'median': float('nan'), 'q1': float('nan'), 'q3': float('nan')}
    return {
        'mean': float(np.mean(valid)),
        'sd': float(np.std(valid, ddof=1)) if valid.size > 1 else float('nan'),
        'median': float(np.median(valid)),
        'q1': float(np.percentile(valid, 25)),
        'q3': float(np.percentile(valid, 75)),
    }


def evaluate_model(model, loader, device, threshold=0.5, save_masks_dir=None, fold=0, exp_name=''):
    model.eval()
    rows = []
    times = []
    if save_masks_dir is not None:
        os.makedirs(save_masks_dir, exist_ok=True)
    with torch.no_grad():
        for imgs, masks, paths in loader:
            imgs = imgs.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(imgs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / max(1, imgs.shape[0])
            times.append(elapsed)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            gt = masks.detach().cpu().numpy()
            for i in range(imgs.shape[0]):
                pred = probs[i, 0] > threshold
                target = gt[i, 0] > 0.5
                dice, iou = binary_metrics(pred, target)
                hd95, assd = hd95_assd(pred, target)
                filename = os.path.basename(paths[i])
                rows.append({
                    'filename': filename,
                    'dice': dice,
                    'iou': iou,
                    'hd95': hd95,
                    'assd': assd,
                    'inference_time_sec': elapsed,
                })
                if save_masks_dir is not None:
                    name_only = os.path.splitext(filename)[0]
                    out_path = os.path.join(save_masks_dir, f'{exp_name}_fold{fold}_{name_only}_pred.png')
                    cv2.imwrite(out_path, pred.astype(np.uint8) * 255)
    return rows, float(np.nanmean(times)) if len(times) > 0 else float('nan')


def save_example_figures(model, loader, device, out_dir, exp_name, fold, max_cases=6):
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    examples = []
    model.eval()
    with torch.no_grad():
        for imgs, masks, paths in loader:
            logits = model(imgs.to(device))
            probs = torch.sigmoid(logits).detach().cpu()
            preds = (probs > 0.5).float()
            for i in range(imgs.shape[0]):
                pred_np = preds[i, 0].numpy() > 0.5
                gt_np = masks[i, 0].numpy() > 0.5
                dice, _ = binary_metrics(pred_np, gt_np)
                examples.append((dice, imgs[i, 0].cpu().numpy(), gt_np, pred_np, os.path.basename(paths[i])))
    if not examples:
        return 0
    examples = sorted(examples, key=lambda x: x[0])
    n_low = max_cases // 2
    n_high = max_cases - n_low
    selected = examples[:n_low] + examples[-n_high:]
    for dice, img, gt, pred, filename in selected:
        name_only = os.path.splitext(filename)[0]
        img_u8 = ((img * 0.5 + 0.5) * 255.0).clip(0, 255).astype('uint8')
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_u8, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(gt.astype(np.uint8) * 255, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('Ground truth')
        axes[1].axis('off')
        axes[2].imshow(pred.astype(np.uint8) * 255, cmap='gray', vmin=0, vmax=255)
        axes[2].set_title(f'Prediction Dice={dice:.3f}')
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'{exp_name}_fold{fold}_{name_only}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    return len(selected)


def append_rows(csv_path, rows, fieldnames):
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if need_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def export_onnx_and_compare(model, loader, device, onnx_path, img_size):
    result = {
        'onnx_exported': False,
        'onnxruntime_available': False,
        'mean_abs_error': '',
        'max_abs_error': '',
        'dice_delta_mean': '',
        'onnx_error': '',
    }
    try:
        model_cpu = copy.deepcopy(model).cpu().eval()
        dummy = torch.randn(1, 1, img_size, img_size, device='cpu')
        torch.onnx.export(
            model_cpu,
            dummy,
            onnx_path,
            input_names=['input'],
            output_names=['logits'],
            opset_version=17,
            dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
        )
        result['onnx_exported'] = True
    except Exception as e:
        result['onnx_error'] = str(e)
        return result

    try:
        import onnxruntime as ort
        result['onnxruntime_available'] = True
    except Exception as e:
        result['onnx_error'] = str(e)
        return result

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    abs_errors = []
    max_errors = []
    dice_deltas = []
    model_cpu.eval()
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs_cpu = imgs.cpu().float()
            torch_logits = model_cpu(imgs_cpu).detach().numpy()
            ort_logits = sess.run(None, {'input': imgs_cpu.numpy().astype(np.float32)})[0]
            abs_errors.append(float(np.mean(np.abs(torch_logits - ort_logits))))
            max_errors.append(float(np.max(np.abs(torch_logits - ort_logits))))
            torch_pred = 1.0 / (1.0 + np.exp(-torch_logits)) > 0.5
            ort_pred = 1.0 / (1.0 + np.exp(-ort_logits)) > 0.5
            gt = masks.numpy() > 0.5
            for i in range(imgs.shape[0]):
                d_t, _ = binary_metrics(torch_pred[i, 0], gt[i, 0])
                d_o, _ = binary_metrics(ort_pred[i, 0], gt[i, 0])
                dice_deltas.append(abs(d_t - d_o))
    result['mean_abs_error'] = float(np.mean(abs_errors)) if abs_errors else ''
    result['max_abs_error'] = float(np.max(max_errors)) if max_errors else ''
    result['dice_delta_mean'] = float(np.mean(dice_deltas)) if dice_deltas else ''
    return result


def save_learning_curves(histories, out_dir, exp_name):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for hist in histories:
        epochs = np.arange(1, len(hist['train_loss']) + 1)
        label = f"Fold {hist['fold']} Ep.{hist['best_epoch']} Dice {hist['best_val_dice']:.4f}"
        axes[0].plot(epochs, hist['train_loss'], linewidth=1.5, label=label)
        axes[1].plot(epochs, hist['val_dice'], linewidth=1.5, label=label)
        if 1 <= hist['best_epoch'] <= len(epochs):
            axes[1].scatter(hist['best_epoch'], hist['val_dice'][hist['best_epoch'] - 1], s=25)
    axes[0].set_title(f'Training loss: {exp_name}')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle=':', linewidth=0.5)
    axes[0].legend(fontsize=8)
    axes[1].set_title(f'Validation Dice: {exp_name}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle=':', linewidth=0.5)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f'{exp_name}_learning_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def save_boxplot(all_case_rows, out_dir):
    if not all_case_rows:
        return None
    exp_names = sorted({r['experiment'] for r in all_case_rows})
    metrics = ['dice', 'iou', 'hd95']
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        data = []
        labels = []
        for exp in exp_names:
            vals = []
            for r in all_case_rows:
                if r['experiment'] == exp:
                    try:
                        v = float(r[metric])
                    except Exception:
                        continue
                    if not np.isnan(v):
                        vals.append(v)
            if vals:
                data.append(vals)
                labels.append(exp)
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(metric.upper())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(out_dir, 'experiment_comparison_boxplots.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def save_summary_tables(all_case_rows, out_dir):
    if not all_case_rows:
        return None
    path = os.path.join(out_dir, 'summary_by_experiment.csv')
    exp_names = sorted({r['experiment'] for r in all_case_rows})
    fields = [
        'experiment', 'model', 'loss', 'augmentation', 'n_cases',
        'dice_mean', 'dice_sd', 'dice_median', 'dice_q1', 'dice_q3',
        'iou_mean', 'iou_sd', 'hd95_mean', 'hd95_sd', 'assd_mean', 'assd_sd',
        'inference_time_sec_mean', 'parameter_count'
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for exp in exp_names:
            rows = [r for r in all_case_rows if r['experiment'] == exp]
            if not rows:
                continue
            dice_s = summarize([r['dice'] for r in rows])
            iou_s = summarize([r['iou'] for r in rows])
            hd95_s = summarize([r['hd95'] for r in rows])
            assd_s = summarize([r['assd'] for r in rows])
            time_s = summarize([r['inference_time_sec'] for r in rows])
            writer.writerow({
                'experiment': exp,
                'model': rows[0].get('model', ''),
                'loss': rows[0].get('loss', ''),
                'augmentation': rows[0].get('augmentation', ''),
                'n_cases': len(rows),
                'dice_mean': dice_s['mean'],
                'dice_sd': dice_s['sd'],
                'dice_median': dice_s['median'],
                'dice_q1': dice_s['q1'],
                'dice_q3': dice_s['q3'],
                'iou_mean': iou_s['mean'],
                'iou_sd': iou_s['sd'],
                'hd95_mean': hd95_s['mean'],
                'hd95_sd': hd95_s['sd'],
                'assd_mean': assd_s['mean'],
                'assd_sd': assd_s['sd'],
                'inference_time_sec_mean': time_s['mean'],
                'parameter_count': rows[0].get('parameter_count', ''),
            })
    return path



def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def train_one_experiment(args, img_paths, msk_paths, model_name, loss_name, aug_mode, device, results_csv):
    exp_name = f'{model_name}_{loss_name}_{aug_mode}'
    print(f'\n=== Experiment: {exp_name} ===')
    exp_out_dir = os.path.join(args.outputs, exp_name)
    exp_model_dir = os.path.join(args.models_dir, exp_name)
    os.makedirs(exp_out_dir, exist_ok=True)
    os.makedirs(exp_model_dir, exist_ok=True)

    train_transform = build_transform(args.img_size, aug_mode, is_train=True)
    val_transform = build_transform(args.img_size, aug_mode, is_train=False)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    histories = []
    all_case_rows = []
    param_count = None

    fieldnames = [
        'record_type', 'experiment', 'model', 'loss', 'augmentation', 'fold', 'epoch', 'filename',
        'train_loss', 'val_dice', 'best_epoch', 'best_val_dice',
        'dice', 'iou', 'hd95', 'assd', 'inference_time_sec', 'parameter_count',
        'onnx_exported', 'onnxruntime_available', 'mean_abs_error', 'max_abs_error', 'dice_delta_mean', 'onnx_error'
    ]

    for fold, (idx_t, idx_v) in enumerate(kf.split(img_paths), start=1):
        print(f'>>> {exp_name} Fold {fold}/{args.n_splits}')
        set_random_seed(args.seed + fold)
        train_ds = JSRTDataset([img_paths[i] for i in idx_t], [msk_paths[i] for i in idx_t], train_transform)
        val_ds = JSRTDataset([img_paths[i] for i in idx_v], [msk_paths[i] for i in idx_v], val_transform)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
        viz_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

        model = build_model(model_name, args.base_ch).to(device)
        if param_count is None:
            param_count = count_parameters(model)
            with open(os.path.join(exp_out_dir, 'model_summary.txt'), 'w', encoding='utf-8') as f:
                f.write(str(model))
                f.write(f'\n\nTrainable parameters: {param_count}\n')
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        loss_fn = make_loss(loss_name)
        best_score = -1.0
        best_epoch = 0
        best_path = os.path.join(exp_model_dir, f'best_fold{fold}.pth')
        history = {'fold': fold, 'train_loss': [], 'val_dice': [], 'best_epoch': 0, 'best_val_dice': 0.0}
        is_tty = sys.stdout.isatty()
        epoch_bar = tqdm(range(1, args.epochs + 1), desc=f'{exp_name} Fold {fold}', disable=not is_tty, dynamic_ncols=True)

        for epoch in epoch_bar:
            model.train()
            losses = []
            for x, y, _ in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
            avg_loss = float(np.mean(losses)) if losses else float('nan')
            val_rows, _ = evaluate_model(model, val_loader, device)
            avg_val_dice = float(np.nanmean([r['dice'] for r in val_rows])) if val_rows else 0.0
            history['train_loss'].append(avg_loss)
            history['val_dice'].append(avg_val_dice)
            append_rows(results_csv, [{
                'record_type': 'epoch', 'experiment': exp_name, 'model': model_name, 'loss': loss_name,
                'augmentation': aug_mode, 'fold': fold, 'epoch': epoch, 'train_loss': avg_loss,
                'val_dice': avg_val_dice, 'parameter_count': param_count,
            }], fieldnames)
            if avg_val_dice > best_score:
                best_score = avg_val_dice
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
            scheduler.step()
            epoch_bar.set_postfix(val=f'{avg_val_dice:.4f}', best=f'{best_score:.4f}', ep=best_epoch)

        history['best_epoch'] = best_epoch
        history['best_val_dice'] = best_score
        histories.append(history)

        model.load_state_dict(load_state_dict(best_path, device))
        model.to(device).eval()
        mask_dir = os.path.join(exp_out_dir, 'raw_masks') if args.save_masks else None
        case_rows, mean_infer = evaluate_model(model, val_loader, device, save_masks_dir=mask_dir, fold=fold, exp_name=exp_name)
        for r in case_rows:
            r.update({
                'record_type': 'case', 'experiment': exp_name, 'model': model_name,
                'loss': loss_name, 'augmentation': aug_mode, 'fold': fold,
                'best_epoch': best_epoch, 'best_val_dice': best_score,
                'parameter_count': param_count,
            })
        append_rows(results_csv, case_rows, fieldnames)
        all_case_rows.extend(case_rows)

        if args.save_figures:
            save_example_figures(model, viz_loader, device, exp_out_dir, exp_name, fold, max_cases=6)

        dice_s = summarize([r['dice'] for r in case_rows])
        iou_s = summarize([r['iou'] for r in case_rows])
        hd95_s = summarize([r['hd95'] for r in case_rows])
        append_rows(results_csv, [{
            'record_type': 'fold_summary', 'experiment': exp_name, 'model': model_name,
            'loss': loss_name, 'augmentation': aug_mode, 'fold': fold,
            'best_epoch': best_epoch, 'best_val_dice': best_score,
            'dice': dice_s['mean'], 'iou': iou_s['mean'], 'hd95': hd95_s['mean'],
            'inference_time_sec': mean_infer, 'parameter_count': param_count,
        }], fieldnames)

        if args.export_onnx and fold == 1:
            onnx_path = os.path.join(exp_model_dir, f'{exp_name}_fold1.onnx')
            onnx_result = export_onnx_and_compare(model, val_loader, device, onnx_path, args.img_size)
            row = {'record_type': 'onnx', 'experiment': exp_name, 'model': model_name, 'loss': loss_name, 'augmentation': aug_mode, 'fold': fold}
            row.update(onnx_result)
            append_rows(results_csv, [row], fieldnames)

        print(f'[Fold {fold}] best validation Dice={best_score:.4f} at epoch {best_epoch}')

    save_learning_curves(histories, exp_out_dir, exp_name)
    dice_all = summarize([r['dice'] for r in all_case_rows])
    iou_all = summarize([r['iou'] for r in all_case_rows])
    hd95_all = summarize([r['hd95'] for r in all_case_rows])
    assd_all = summarize([r['assd'] for r in all_case_rows])
    append_rows(results_csv, [{
        'record_type': 'experiment_summary', 'experiment': exp_name, 'model': model_name,
        'loss': loss_name, 'augmentation': aug_mode,
        'dice': dice_all['mean'], 'iou': iou_all['mean'], 'hd95': hd95_all['mean'],
        'assd': assd_all['mean'], 'parameter_count': param_count,
    }], fieldnames)
    return all_case_rows


def find_pairs(image_dir, mask_dir):
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.bmp')))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    if not image_paths:
        raise FileNotFoundError(f'No BMP images found in: {image_dir}')
    if not mask_paths:
        raise FileNotFoundError(f'No PNG masks found in: {mask_dir}')

    image_ids_all = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]
    mask_ids_all = [os.path.splitext(os.path.basename(path))[0] for path in mask_paths]
    if len(set(image_ids_all)) != len(image_ids_all):
        raise RuntimeError('Duplicate image stems were found.')
    if len(set(mask_ids_all)) != len(mask_ids_all):
        raise RuntimeError('Duplicate mask stems were found.')
    image_by_id = dict(zip(image_ids_all, image_paths))
    mask_by_id = dict(zip(mask_ids_all, mask_paths))
    image_ids = set(image_by_id)
    mask_ids = set(mask_by_id)
    missing_masks = sorted(image_ids - mask_ids)
    missing_images = sorted(mask_ids - image_ids)
    if missing_masks or missing_images:
        details = []
        if missing_masks:
            details.append(f'missing masks for {missing_masks[:10]}')
        if missing_images:
            details.append(f'missing images for {missing_images[:10]}')
        raise RuntimeError('Image-mask stem mismatch: ' + '; '.join(details))

    case_ids = sorted(image_ids)
    return [image_by_id[case_id] for case_id in case_ids], [mask_by_id[case_id] for case_id in case_ids]

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate JSRT cardiac silhouette segmentation models.')
    parser.add_argument('--images', default=DEFAULT_DIRS['images'])
    parser.add_argument('--masks', default=DEFAULT_DIRS['masks'])
    parser.add_argument('--outputs', default=DEFAULT_DIRS['outputs'])
    parser.add_argument('--models-dir', dest='models_dir', default=None)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base-ch', type=int, default=64)
    parser.add_argument('--models', nargs='+', default=['resunet'], choices=['unet', 'attention_unet', 'resunet'])
    parser.add_argument('--losses', nargs='+', default=['bce_dice'], choices=['bce', 'dice', 'bce_dice'])
    parser.add_argument('--augs', nargs='+', default=['full'], choices=['none', 'geometric', 'intensity', 'full'])
    parser.add_argument('--save-masks', action='store_true')
    parser.add_argument('--save-figures', action='store_true')
    parser.add_argument('--export-onnx', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.img_size < 16 or args.img_size % 16 != 0:
        raise ValueError('--img-size must be at least 16 and divisible by 16.')
    if args.n_splits < 2:
        raise ValueError('--n-splits must be at least 2.')
    if args.epochs < 1:
        raise ValueError('--epochs must be at least 1.')
    set_random_seed(args.seed)
    device = select_device()
    if args.models_dir is None:
        args.models_dir = os.path.join(args.outputs, 'models')
    os.makedirs(args.outputs, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    img_paths, msk_paths = find_pairs(args.images, args.masks)
    print(f'Device: {device}')
    if len(img_paths) < args.n_splits:
        raise ValueError(f'--n-splits ({args.n_splits}) exceeds the number of cases ({len(img_paths)}).')
    print(f'Images: {len(img_paths)}')
    print(f'Model names: {args.models}')
    print(f'Model save directory: {args.models_dir}')

    results_csv = os.path.join(args.outputs, 'all_results.csv')
    if os.path.exists(results_csv):
        os.remove(results_csv)

    with open(os.path.join(args.outputs, 'experiment_config.json'), 'w', encoding='utf-8') as f:
        config = vars(args).copy()
        config['device'] = str(device)
        config['augmentation_params'] = AUGMENTATION_PARAMS
        config['scipy_available'] = SCIPY_AVAILABLE
        json.dump(config, f, ensure_ascii=True, indent=2)

    all_case_rows = []
    for model_name in args.models:
        for loss_name in args.losses:
            for aug_mode in args.augs:
                rows = train_one_experiment(args, img_paths, msk_paths, model_name, loss_name, aug_mode, device, results_csv)
                all_case_rows.extend(rows)

    box_path = save_boxplot(all_case_rows, args.outputs)
    summary_path = save_summary_tables(all_case_rows, args.outputs)
    print('\n=== Finished ===')
    print(f'Results CSV: {results_csv}')
    if summary_path:
        print(f'Summary CSV: {summary_path}')
    if box_path:
        print(f'Comparison boxplot: {box_path}')


if __name__ == '__main__':
    main()
