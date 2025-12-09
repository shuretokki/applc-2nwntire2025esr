import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

class kDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, debug_mode=False, patch_size=None, upscale_factor=2):
        """
        Args:
            hr_dir (str): Directory containing High-Resolution images.
            lr_dir (str): Directory containing Low-Resolution images.
            debug_mode (bool): If True, loads only a subset of images for debugging.
            patch_size (int): Size of the cropped patch for training (e.g., 48 or 64).
            upscale_factor (int): The super-resolution scale factor (e.g., 2 or 4).
        """
        super(kDataset, self).__init__()
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.debug_mode = debug_mode
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor

        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, '*')))
        self.lr_files = sorted(glob.glob(os.path.join(lr_dir, '*')))


        self.hr_files = [f for f in self.hr_files if os.path.isfile(f)]
        self.lr_files = [f for f in self.lr_files if os.path.isfile(f)]

        if self.debug_mode:
            print("[INFO] Debug mode enabled: Loading first 10 images only.")
            self.hr_files = self.hr_files[:10]
            self.lr_files = self.lr_files[:10]

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset (HR/LR pair).
        Applies random cropping and augmentations.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: {'LR': lr_tensor, 'HR': hr_tensor}
        """
        hr_path = self.hr_files[idx]
        lr_path = self.lr_files[idx]

        hr_image = Image.open(hr_path).convert('RGB')
        lr_image = Image.open(lr_path).convert('RGB')

        if self.patch_size is not None:
            w, h = lr_image.size
            tp = self.patch_size

            # ensure img is big enough
            if w >= tp and h >= tp:
                i = random.randint(0, h - tp)
                j = random.randint(0, w - tp)

                lr_image = lr_image.crop((j, i, j + tp, i + tp))

                # crop hr relative to lr
                hr_image = hr_image.crop((j * self.upscale_factor, i * self.upscale_factor,
                                          (j + tp) * self.upscale_factor, (i + tp) * self.upscale_factor))
            else:
                pass

        if random.random() < 0.5:
            hr_image = TF.hflip(hr_image)
            lr_image = TF.hflip(lr_image)

        if random.random() < 0.5:
            hr_image = TF.vflip(hr_image)
            lr_image = TF.vflip(lr_image)

        if random.random() < 0.5:
            hr_image = TF.rotate(hr_image, 90)
            lr_image = TF.rotate(lr_image, 90)

        hr_tensor = self.to_tensor(hr_image)
        lr_tensor = self.to_tensor(lr_image)

        return {'LR': lr_tensor, 'HR': hr_tensor}

if __name__ == "__main__":
    hr_path = 'data/train_hr'
    lr_path = 'data/train_lr'


    if not os.path.exists(hr_path): hr_path = 'data/train_HR'
    if not os.path.exists(lr_path): lr_path = 'data/train_LR'

    print(f"[INFO] Initializing dataset...\nHR: {hr_path}\nLR: {lr_path}")

    train_ds = kDataset(hr_dir=hr_path, lr_dir=lr_path, debug_mode=True, patch_size=48)
    print(f"[INFO] Dataset size: {len(train_ds)}")

    if len(train_ds) > 0:

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
        batch = next(iter(train_loader))

        print("First batch shapes:")
        print(f"  LR: {batch['LR'].shape}")
        print(f"  HR: {batch['HR'].shape}")
