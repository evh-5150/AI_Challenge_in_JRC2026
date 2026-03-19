import os
import sys
import glob
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

def _set_japanese_font():
    candidates = [
        'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'AppleGothic',
        'Yu Gothic', 'YuGothic', 'MS Gothic',
        'Noto Sans CJK JP', 'IPAexGothic', 'IPAPGothic', 'TakaoPGothic'
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42
            return name
    return None

_USED_FONT = _set_japanese_font()
if _USED_FONT is None:
    print('Warning: 日本語フォントが見つかりません．Noto Sans CJK JP などをインストールしてください．')

DIR_CONFIG = {
    'images': './segmentation/org',
    'masks': './segmentation/label',
    'outputs': './segmentation/main7.3_results_resunet',
    'models': './segmentation/main7.3_models_resunet',
}

TRAIN_PARAMS = {
    'img_size': 256,
    'batch_size': 8,
    'epochs': 100,
    'lr': 1e-4,
    'n_splits': 5,
    'num_workers': 0,
}

train_transform = A.Compose([
    A.Resize(TRAIN_PARAMS['img_size'], TRAIN_PARAMS['img_size']),
    A.Affine(
        translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
        scale=(0.9, 1.1),
        rotate=(-10, 10),
        shear=0.0,
        p=0.7,
    ),
    A.RandomBrightnessContrast(p=0.2),
    A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(TRAIN_PARAMS['img_size'], TRAIN_PARAMS['img_size']),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

def select_device():
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.skip = nn.Conv2d(ch_in, ch_out, 1, bias=False) if ch_in != ch_out else nn.Identity()
        self.main_path = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(True),
            nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
        )
        self.activation = nn.ReLU(True)

    def forward(self, x):
        return self.activation(self.main_path(x) + self.skip(x))

class ResUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ResBlock(in_ch, 64)
        self.enc2 = ResBlock(64, 128)
        self.enc3 = ResBlock(128, 256)
        self.enc4 = ResBlock(256, 512)
        self.bridge = ResBlock(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ResBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ResBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ResBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ResBlock(128, 64)
        self.mask_head = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bridge(self.pool(s4))
        def _up_and_cat(up_layer, current, skip):
            y = up_layer(current)
            if y.shape != skip.shape:
                y = torch.nn.functional.interpolate(y, size=skip.shape[2:])
            return torch.cat([skip, y], dim=1)
        d4 = self.dec4(_up_and_cat(self.up4, b, s4))
        d3 = self.dec3(_up_and_cat(self.up3, d4, s3))
        d2 = self.dec2(_up_and_cat(self.up2, d3, s2))
        d1 = self.dec1(_up_and_cat(self.up1, d2, s1))
        return self.mask_head(d1)

class JSRTDataset(Dataset):
    def __init__(self, img_list, msk_list, aug=None):
        self.img_list = img_list
        self.msk_list = msk_list
        self.aug = aug

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx], 0)
        msk = cv2.imread(self.msk_list[idx], 0)
        if img is None:
            raise FileNotFoundError(f'画像を読めません: {self.img_list[idx]}')
        if msk is None:
            raise FileNotFoundError(f'マスクを読めません: {self.msk_list[idx]}')
        label = np.where(msk == 85, 1.0, 0.0).astype(np.float32)
        if self.aug:
            res = self.aug(image=img, mask=label)
            img, label = res['image'], res['mask']
        return img, label.unsqueeze(0), self.img_list[idx]

def calc_dice(logits, target, smooth=1e-8):
    pred = (torch.sigmoid(logits) > 0.5).float().view(-1)
    target = target.view(-1)
    return (2.0 * (pred * target).sum()) / (pred.sum() + target.sum() + smooth)

def _prepare_case_results_csv(save_dir):
    csv_path = os.path.join(save_dir, 'case_results.csv')
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    f = open(csv_path, 'a', newline='')
    writer = csv.writer(f)
    if need_header:
        writer.writerow(['Fold', 'Filename', 'Dice'])
    return f, writer

def visualize_and_save(model, loader, fold_idx, save_dir, device, num_samples=float('inf')):
    model.eval()
    cnt = 0
    raw_mask_dir = os.path.join(save_dir, 'raw_masks')
    paper_fig_dir = os.path.join(save_dir, 'paper_figures')
    os.makedirs(raw_mask_dir, exist_ok=True)
    os.makedirs(paper_fig_dir, exist_ok=True)
    case_csv_f, case_csv_writer = _prepare_case_results_csv(save_dir)
    try:
        with torch.no_grad():
            for imgs, masks, paths in loader:
                probs = torch.sigmoid(model(imgs.to(device)))
                preds = (probs > 0.5).float().cpu()
                for i in range(imgs.shape[0]):
                    if cnt >= num_samples:
                        return
                    filename = os.path.basename(paths[i])
                    name_only, _ = os.path.splitext(filename)
                    img_np = (imgs[i, 0].cpu().numpy() * 0.5 + 0.5)
                    img_u8 = (img_np * 255.0).clip(0, 255).astype('uint8')
                    gt_u8 = (masks[i, 0].cpu().numpy() * 255.0).astype('uint8')
                    pr_u8 = (preds[i, 0].numpy() * 255.0).astype('uint8')
                    inter = (preds[i, 0] * masks[i, 0]).sum().item()
                    union = preds[i, 0].sum().item() + masks[i, 0].sum().item()
                    dice = (2.0 * inter) / (union + 1e-8)
                    mask_path = os.path.join(raw_mask_dir, f'fold{fold_idx}_{name_only}_pred.png')
                    cv2.imwrite(mask_path, pr_u8)
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(img_u8, cmap='gray', vmin=0, vmax=255)
                    axes[0].set_title('Original Image')
                    axes[0].axis('off')
                    axes[1].imshow(gt_u8, cmap='gray', vmin=0, vmax=255)
                    axes[1].set_title('Ground Truth')
                    axes[1].axis('off')
                    axes[2].imshow(pr_u8, cmap='gray', vmin=0, vmax=255)
                    axes[2].set_title(f'Prediction (Dice: {dice:.3f})')
                    axes[2].axis('off')
                    plt.tight_layout()
                    fig_path = os.path.join(paper_fig_dir, f'fold{fold_idx}_{name_only}_comparison.png')
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    case_csv_writer.writerow([fold_idx, filename, f'{dice:.12f}'])
                    cnt += 1
    finally:
        try:
            case_csv_f.close()
        except Exception:
            pass

def save_learning_curves(fold_histories, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, 'learning_curves_5fold_cv.png')
    grayscale_colors = ['black', 'dimgray', 'gray', 'darkgray', 'silver']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for i, hist in enumerate(fold_histories):
        epochs = np.arange(1, len(hist['loss']) + 1)
        c = grayscale_colors[i % len(grayscale_colors)]
        ls = linestyles[i % len(linestyles)]
        best_ep = hist['best_ep']
        best_dice = hist['best_dice']
        label = f'Fold {i+1} (Ep. {best_ep}, Dice {best_dice:.4f})'
        axes[0].plot(epochs, hist['loss'], color=c, linestyle=ls, linewidth=1.8, label=label)
        axes[1].plot(epochs, hist['val_dice'], color=c, linestyle=ls, linewidth=1.8, label=label)
        if 1 <= best_ep <= len(epochs):
            axes[1].scatter(best_ep, hist['val_dice'][best_ep - 1], color=c, s=30, marker='o', edgecolors='black', zorder=3)
    axes[0].set_title('Learning Curves (5-Fold Cross Validation)')
    axes[0].set_ylabel('Training Loss')
    axes[0].grid(True, linestyle=':', linewidth=0.5, color='lightgray')
    axes[0].legend(loc='best', fontsize=9)
    axes[1].set_title('Validation Dice')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle=':', linewidth=0.5, color='lightgray')
    axes[1].legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig_path

def run_training():
    device = select_device()
    for d in [DIR_CONFIG['outputs'], DIR_CONFIG['models']]:
        os.makedirs(d, exist_ok=True)
    img_paths = sorted(glob.glob(os.path.join(DIR_CONFIG['images'], '*.bmp')))
    msk_paths = sorted(glob.glob(os.path.join(DIR_CONFIG['masks'], '*.png')))
    if not img_paths:
        print('Error: No images found.')
        return
    if len(img_paths) != len(msk_paths):
        raise RuntimeError(f'画像とマスクの枚数が一致しません：{len(img_paths)} vs {len(msk_paths)}')
    log_file = os.path.join(DIR_CONFIG['outputs'], 'training_log.csv')
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['fold', 'epoch', 'loss', 'val_dice'])
    fold_histories = []
    kf = KFold(n_splits=TRAIN_PARAMS['n_splits'], shuffle=True, random_state=42)
    for fold, (idx_t, idx_v) in enumerate(kf.split(img_paths), start=1):
        print(f'\n>>> Fold {fold} processing')
        loader_t = DataLoader(
            JSRTDataset([img_paths[i] for i in idx_t], [msk_paths[i] for i in idx_t], train_transform),
            batch_size=TRAIN_PARAMS['batch_size'],
            shuffle=True,
            num_workers=TRAIN_PARAMS['num_workers'],
            pin_memory=False,
        )
        loader_v = DataLoader(
            JSRTDataset([img_paths[i] for i in idx_v], [msk_paths[i] for i in idx_v], val_transform),
            batch_size=TRAIN_PARAMS['batch_size'],
            shuffle=False,
            num_workers=TRAIN_PARAMS['num_workers'],
            pin_memory=False,
        )
        model = ResUNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=TRAIN_PARAMS['lr'], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_PARAMS['epochs'])
        bce_loss = nn.BCEWithLogitsLoss()
        best_score = -1.0
        best_epoch = 0
        best_path = os.path.join(DIR_CONFIG['models'], f'best_f{fold}.pth')
        history = {'loss': [], 'val_dice': [], 'best_ep': 0, 'best_dice': 0.0}
        is_tty = sys.stdout.isatty()
        epoch_bar = tqdm(
            range(1, TRAIN_PARAMS['epochs'] + 1),
            desc=f'Fold {fold}/{TRAIN_PARAMS["n_splits"]}',
            unit='ep',
            dynamic_ncols=True,
            leave=True,
            position=0,
            disable=not is_tty,
            mininterval=0.5,
        )
        for epoch in epoch_bar:
            model.train()
            sum_loss = 0.0
            n_batches = len(loader_t)
            batch_bar = tqdm(
                loader_t,
                desc=f'Fold {fold} | Ep {epoch}/{TRAIN_PARAMS["epochs"]}',
                total=n_batches,
                leave=False,
                dynamic_ncols=True,
                position=1,
                disable=not is_tty,
                mininterval=0.2,
            )
            for x, y, _ in batch_bar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                p_flat = torch.sigmoid(pred).view(-1)
                y_flat = y.view(-1)
                dice_l = 1.0 - (2.0 * (p_flat * y_flat).sum() + 1e-6) / (p_flat.sum() + y_flat.sum() + 1e-6)
                loss = bce_loss(pred, y) + dice_l
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                batch_bar.set_postfix(loss=f'{loss.item():.4f}')
            avg_loss = sum_loss / max(1, n_batches)
            model.eval()
            scores = []
            with torch.no_grad():
                for x, y, _ in loader_v:
                    scores.append(calc_dice(model(x.to(device)), y.to(device)))
            avg_dice = float(np.mean([s.detach().cpu().item() for s in scores])) if scores else 0.0
            history['loss'].append(avg_loss)
            history['val_dice'].append(avg_dice)
            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([fold, epoch, avg_loss, avg_dice])
            if avg_dice > best_score:
                best_score = avg_dice
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
                history['best_ep'] = best_epoch
                history['best_dice'] = best_score
            current_lr = optimizer.param_groups[0]['lr']
            epoch_bar.set_postfix(best=f'{best_score:.4f}', best_ep=best_epoch, val=f'{avg_dice:.4f}', lr=f'{current_lr:.2e}')
            scheduler.step()
        fold_histories.append(history)
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
            model.to(device).eval()
            viz_loader = DataLoader(
                JSRTDataset([img_paths[i] for i in idx_v], [msk_paths[i] for i in idx_v], val_transform),
                batch_size=1,
                shuffle=False,
                num_workers=TRAIN_PARAMS['num_workers'],
            )
            visualize_and_save(model, viz_loader, fold, DIR_CONFIG['outputs'], device, num_samples=float('inf'))
            print(f'[Fold {fold}] Best Dice: {best_score:.4f} (Ep. {best_epoch})')
            print(f'[Fold {fold}] Visualization saved to: {os.path.join(DIR_CONFIG["outputs"], "raw_masks")} and {os.path.join(DIR_CONFIG["outputs"], "paper_figures")}')
        else:
            print(f'[Fold {fold}] Best model not found at {best_path}')
    curves_path = save_learning_curves(fold_histories, DIR_CONFIG['outputs'])
    print(f'Learning curves saved to: {curves_path}')
    print('\n=== Cross-validation summary ===')
    for i, hist in enumerate(fold_histories, start=1):
        print(f'Fold {i}: Best Dice = {hist["best_dice"]:.4f} (Ep. {hist["best_ep"]})')
    mean_best = float(np.mean([h['best_dice'] for h in fold_histories])) if fold_histories else 0.0
    print(f'Mean Best Dice: {mean_best:.4f}')

if __name__ == '__main__':
    run_training()
