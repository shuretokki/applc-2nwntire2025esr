import os

repo_name = "applc-2nwntire2025esr"
repo_url = "https://github.com/shuretokki/applc-2nwntire2025esr.git"

if os.path.exists(repo_name):
    %cd {repo_name}
    !git reset --hard
    !git pull
    %cd ..
else:
    !git clone {repo_url}

!pip install -q {repo_name}/basicsr-1.4.2
!pip install -q -r {repo_name}/requirements.txt

ds = "/kaggle/input/visdrone-dataset/VisDrone_Dataset/VisDrone2019-DET-train/images"
%cd {repo_name}

if os.path.exists(ds):
    if not os.path.exists("data/train_hr"):
        print("Prepping dataset...")
        !python dataprep.py --source "$ds" --limit 1000

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
!python train-k.py --epochs 1 --batch_size 32 --patch_size 96
# !python train-k.py --batch_size 32 --patch_size 96

%cd ..
