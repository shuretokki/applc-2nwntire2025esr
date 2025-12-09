import os
import glob
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
def prepare_dataset(hr_sourcedir='data/train', limit=300):
    """
    Args:
        hr_sourcedir (str): Path to the source directory containing high-resolution images.
        limit (int): Maximum number of source images to process (to prevent disk overflow).
    """
    # prepares dataset, fixes hr to mult of 4, generates 4x lr
    if not os.path.exists(hr_sourcedir) and os.path.exists('data/train'):
        hr_sourcedir = 'data/train'

    hr_fixed_dir = 'data/train_hr'
    lr_dir = 'data/train_lr'

    os.makedirs(hr_fixed_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    extensions = ['png', 'jpg', 'jpeg', 'BMP', 'tif', 'tiff']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(hr_sourcedir, '**', f'*.{ext}'), recursive=True))
        files.extend(glob.glob(os.path.join(hr_sourcedir, '**', f'*.{ext.upper()}'), recursive=True))

    files = sorted(list(set(files)))

    if limit is not None and len(files) > limit:
        print(f"[WARN] Limiting dataset to first {limit} images (disk safety)")
        files = files[:limit]

    print(f"[INFO] Found {len(files)} images in {hr_sourcedir}")

    count = 0
    last_hr_shape = (0, 0)
    last_lr_shape = (0, 0)

    for file_path in files:
        filename = os.path.basename(file_path)

        try:
            with Image.open(file_path) as img:
                img = img.convert('RGB')
                w, h = img.size

                patch_size = 512
                stride = 512

                w, h = img.size

                # if image is too small, just keep it (unless tiny)
                if w < patch_size or h < patch_size:
                    w_new = w - (w % 4)
                    h_new = h - (h % 4)
                    if w_new < 16 or h_new < 16: continue # skip garbage
                    img_fixed = img.crop((0,0,w_new,h_new))
                    save_name = f"{os.path.splitext(filename)[0]}.png"

                    img_fixed.save(os.path.join(hr_fixed_dir, save_name))
                    img_fixed.resize((w_new//4, h_new//4), Image.BICUBIC).save(os.path.join(lr_dir, save_name))
                    count += 1
                    continue

                for x in range(0, w - patch_size + 1, stride):
                    for y in range(0, h - patch_size + 1, stride):
                        img_patch = img.crop((x, y, x + patch_size, y + patch_size))

                        # save HR (as PNG)
                        save_name = f"{os.path.splitext(filename)[0]}_{x}_{y}.png"
                        img_patch.save(os.path.join(hr_fixed_dir, save_name))

                        # generate & save LR
                        img_lr = img_patch.resize((patch_size//4, patch_size//4), Image.BICUBIC)
                        img_lr.save(os.path.join(lr_dir, save_name))

                        count += 1
                        last_hr_shape = img_patch.size
                        last_lr_shape = img_lr.size


        except Exception as e:
            print(f"[ERROR] processing {filename}: {e}")

    print(f"[INFO] Processed {count} images.")
    if count > 0:
        print(f"Sample verification - HR shape: {last_hr_shape}, LR shape: {last_lr_shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/train', help='Path to source HR images')
    parser.add_argument('--limit', type=int, default=500, help='Limit number of images')
    args = parser.parse_args()

    prepare_dataset(hr_sourcedir=args.source, limit=args.limit)
