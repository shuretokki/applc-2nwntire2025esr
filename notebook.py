import os

repo_name = "applc-2nwntire2025esr"
repo_url = "https://github.com/shuretokki/applc-2nwntire2025esr.git"

if os.path.exists(repo_name):
    %cd {repo_name}
    !git pull
    %cd ..
else:
    !git clone {repo_url}

!sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' {repo_name}/basicsr-1.4.2/basicsr/data/degradations.py
!pip install -q {repo_name}/basicsr-1.4.2
!pip install -q -r {repo_name}/requirements.txt

ds = "/kaggle/input/visdrone-dataset/VisDrone2019-DET-train/images"
%cd applc-2nwntire2025esr

local_ds = "data/temp"
if os.path.exists(ds):
    if not os.path.exists(local_ds):
        os.makedirs(local_ds)
        !cp -r "{ds}/." "{local_ds}/"

    print(f"Preparing // {local_ds}...")
    !python dataprep.py --source "{local_ds}" --limit 5
    # !python dataprep.py --source "{local_ds}" --limit 2000
else:
    print(f"WARNING: Source dataset {ds} not found.")

!python train-k.py --epochs 1
# !python train-k.py

%cd ..
