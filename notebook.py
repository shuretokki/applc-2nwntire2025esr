import os

if not os.path.exists("applc-2nwntire2025esr"):
    !git clone https://github.com/shuretokki/applc-2nwntire2025esr.git

!pip install applc-2nwntire2025esr/basicsr-1.4.2
!pip install -r applc-2nwntire2025esr/requirements.txt

ds = "/kaggle/input/visdrone-dataset/VisDrone2019-DET-train/images"
repo_data = "applc-2nwntire2025esr/data"

if not os.path.exists(repo_data):
    os.makedirs(repo_data)

TARGET_HR = "applc-2nwntire2025esr/data/train_hr"

if not os.path.exists(TARGET_HR):
    if os.path.exists(ds):
        print(f"Linking {ds} to {TARGET_HR}")
        os.symlink(ds, TARGET_HR)
    else:
        print(f"WARNING: Dataset path {ds} not found.")

%cd applc-2nwntire2025esr

!python train-k.py --epochs 1
# !python train-k.py

%cd ..
