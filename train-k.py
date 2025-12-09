import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import kDataset
from model import HATIQCMix

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("[WARN] torch.cuda.amp not found. Slower FP32 training will be used.")


def train(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    PATCH_SIZE = args.patch_size
    LR_RATE = 2e-4

    CHECKPOINT_PATH = "checkpoint.pth"
    SAVE_INTERVAL = 10
    PRINT_INTERVAL = 1

    HR_DIR = 'data/train_hr'
    LR_DIR = 'data/train_lr'

    if not os.path.exists(HR_DIR): HR_DIR = 'data/train_HR'
    if not os.path.exists(LR_DIR): LR_DIR = 'data/train_LR'

    print(f"[INFO] Starting training on {DEVICE}")
    print(f"Data - HR: {HR_DIR}, LR: {LR_DIR}")
    print(f"Config - Patch: {PATCH_SIZE}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")

    train_dataset = kDataset(hr_dir=HR_DIR, lr_dir=LR_DIR, debug_mode=False, patch_size=PATCH_SIZE, upscale_factor=args.upscale)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    model = HATIQCMix(
            img_size=PATCH_SIZE,
            patch_size=1,
            in_chans=3,
            embed_dim=48,
            depths=(6, 6, 6, 6),
            num_heads=(6, 6, 6, 6),
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            mlp_ratio=2.,
            qkv_bias=True,
            upscale=args.upscale
        )
    model = model.to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"[INFO] Detected {torch.cuda.device_count()} GPUs. Enabling Multi-GPU")
        model = nn.DataParallel(model)

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LR_RATE, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    scaler = GradScaler() if AMP_AVAILABLE else None

    start_epoch = 0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"[INFO] Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"[INFO] Resuming from epoch {start_epoch}")
    else:
        print("[INFO] No checkpoint found. Starting from scratch.")

    if hasattr(torch, 'compile'):
        print("[INFO] PyTorch 2.0+ detected. Compiling model for faster execution...")
        model = torch.compile(model)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    model.train()
    for epoch in range(start_epoch, EPOCHS):
        epoch_start_time = time.time()
        epoch_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            lr_imgs = data['LR'].to(DEVICE, non_blocking=True)
            hr_imgs = data['HR'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True) # efficient zero_grad

            if AMP_AVAILABLE and DEVICE.type == 'cuda':
                with autocast():
                    outputs = model(lr_imgs)
                    loss = criterion(outputs, hr_imgs)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(lr_imgs)
                loss = criterion(outputs, hr_imgs)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        epoch_duration = time.time() - epoch_start_time

        if (epoch + 1) % PRINT_INTERVAL == 0:
            print(f"USING {gpu_name} <=> Epoch //= [{epoch+1}/{EPOCHS}] && Loss //= {avg_loss:.6f} && Time //= {epoch_duration:.2f}s")

        if (epoch + 1) % SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, CHECKPOINT_PATH)

            torch.save(model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(), "ltsmodel.pth")

    print("[INFO] Training finished.")

    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, "final_model_checkpoint.pth")

    torch.save(model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(), "final_model_weights.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=800, help='Total epochs (default: 800)')
    parser.add_argument('--upscale', type=int, default=4, help='Upscale factor (default: 4)')
    args = parser.parse_args()

    train(args)
