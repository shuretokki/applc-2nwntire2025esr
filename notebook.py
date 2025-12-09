import os

if not os.path.exists("applc-2nwntire2025esr"):
    !git clone https://github.com/shuretokki/applc-2nwntire2025esr.git

!pip install applc-2nwntire2025esr/basicsr-1.4.2
!pip install -r applc-2nwntire2025esr/requirements.txt

ds = "/kaggle/input/visdrone-dataset/VisDrone2019-DET-train/images"
%cd applc-2nwntire2025esr

if os.path.exists(ds):
    print(f"Generating dataset from {ds}...")
    !python dataprep.py --source "$ds" --limit 2000
else:
    print(f"WARNING: Source dataset {ds} not found.")

!python train-k.py --epochs 1
# !python train-k.py

%cd ..
