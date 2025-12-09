import os

repo_name = "applc-2nwntire2025esr"
repo_url = "https://github.com/shuretokki/applc-2nwntire2025esr.git"

if os.path.exists(repo_name):
    %cd {repo_name}
    !git pull
    %cd ..
else:
    !git clone {repo_url}

!pip install -q {repo_name}/basicsr-1.4.2
!pip install -q -r {repo_name}/requirements.txt

ds = "/kaggle/input/visdrone-dataset/VisDrone_Dataset/VisDrone2019-DET-train/images"
%cd applc-2nwntire2025esr

if os.path.exists(ds):
    !python dataprep.py --source "$ds" --limit 5
    # !python dataprep.py --source "$ds" --limit 2000
else:
    print(f"WARNING: Source dataset {ds} not found.")

!python train-k.py --epochs 1
# !python train-k.py

%cd ..
