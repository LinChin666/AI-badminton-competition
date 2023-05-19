# [教電腦看羽球 - 台灣第一個結合AI與運動的競賽 - TEAM_3031](https://aidea-web.tw/topic/cbea66cc-a993-4be8-933d-1aa9779001f8)

## 簡介
近年統計全球有約22億羽球人口，台灣則超過3百萬人，單項運動全國普及度排名第二，且近年羽球選手在國際賽場上有十分突出的表現，國民關注度逐漸提升。
針對羽球技戰術分析，本團隊已提出比賽拍拍記錄格式並開發了電腦視覺輔助的快速拍拍標記程式，啟動羽球大數據的研究，雖然已使用許多電腦輔助的技巧，但人工進行拍拍標記依然需耗費人力及時間，尤其技術資料識別仍需具有羽球專業的人員來執行。透過本競賽期望能邀集具機器學習、影像處理及運動科學專長的專家與高手，開發高辨識率的自動拍拍標記模型，讓巨量羽球情蒐成為可能，普及羽球技戰術分析的科研與應用。

## 硬體與軟體
- 顯示卡: NVIDIA GeForce RTX2060 (MSI GP72 7REX), T4 (Google Colaboratory)
- 語言: Python
- 套件: git, matplotlib, numpy, opencv-python, Pillow, psutil, PyYAML, scipy, thop, torch, torchvision, tqdm, tensorboard, pandas, seaborn, pyqt5, PyMySQL, imutils, piexif, scikit-learn, keras


## 測試指令

### TrackNetv2
```
conda create --name tf python=3.9 -y
conda deactivate
conda activate tf
nvidia-smi
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip install --upgrade pip
pip install tensorflow==2.12.*
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

git clone https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2.git
cd TrackNetv2/3_in_3_out/
python3 predict.py --load_weights=model906_30
```

### YOLOv7
```
git clone https://github.com/WongKinYiu/yolov7.git
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt

python answer.py
```



## 名次
### Public: 0.061 - 14th

### Private: 0.0558 - 12th

## 參考資料
- [AI CUP 2023 春季賽【教電腦看羽球競賽－台灣第一個結合AI與運動的競賽】](https://www.youtube.com/playlist?list=PLk_m5EiRQRF2fuGNoLep5TCqcPy1Aac3e)
- [Baseline Code](https://drive.google.com/drive/folders/18Yr3Y630aMGvlUfxQArv7rjh5jo2diUA)
- [TrackNetV2: Efficient TrackNet (GitLab)](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2)
- [Shuttlecock Trajectory Detection and Semantic Analysis](https://hdl.handle.net/11296/85425g)
- [Official YOLOv7](https://github.com/WongKinYiu/yolov7)
