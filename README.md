# XFeat-Sigma 🚀  
Improving XFeat’s precision with enhanced feature correspondence and noise filtering.

## 📌 Overview  
XFeat++ is an improved version of **XFeat: Accelerated Features for Lightweight Image Matching**, designed to enhance image feature extraction and matching. We introduce **outlier removal using DBSCAN clustering** and **homography transformations** to improve robustness against perspective and geometric variations.  

## 🔥 Features  
✅ **Lightweight & Efficient** – Optimized for fast feature extraction and matching  
✅ **Homography-Guided Matching** – Improves resilience to viewpoint and geometric distortions  
✅ **Outlier Removal using DBSCAN** – Filters incorrect feature matches for higher accuracy  
✅ **Semi-Dense Matching Enhancement** – Better feature correspondences for more precise alignment  
✅ **Trained on Megadepth & COCO** – Ensuring generalizability across diverse datasets  

## 📂 Project Structure  ( still under progress )
```
XFeat++/
│── data/              # Dataset & preprocessing scripts  
│── models/            # XFeat++ model and modifications  
│── experiments/       # Ablation studies and evaluation scripts  
│── results/           # Logs and output results  
│── utils/             # Helper functions and utilities  
│── train.py           # Model training script  
│── test.py            # Model evaluation script  
│── README.md          # Project documentation  
```

## 🛠️ Installation  
### 1️⃣ Clone the Repository  
```bash  
git clone https://github.com/tanalpha-aditya/XFeat-Enhanced.git  
cd XFeat-Enhanced  
```

### 2️⃣ Install Dependencies  
```bash  
pip install -r requirements.txt  
```

### 3️⃣ Download Datasets  
- **[MegaDepth](https://github.com/mihaidusmanu/megadepth)**  
- **[COCO](https://cocodataset.org/#download)**  

Modify `config.yaml` to set dataset paths.

## 🚀 Training  
To train the model from scratch:  
```bash  
python train.py --epochs 100 --batch_size 10  
```

For evaluation:  
```bash  
python test.py --dataset HPatches  
```

## 📊 Results & Benchmarks  
- Achieves **competitive accuracy** while being up to **5x faster** than deep learning-based methods.  
- Improves **feature correspondence reliability** by removing outlier matches.  
- Outperforms baseline XFeat under perspective and geometric variations.  

## 📝 Future Work  
- Further optimization of feature matching under extreme transformations.  
- Experimentation with different clustering methods for outlier detection.  
- Integration with real-time SLAM and AR applications.  

## 👥 Contributors  
- **Aditya Raghuvanshi** ([@tanalpha-aditya](https://github.com/tanalpha-aditya))  
- **Yash Bhaskar** ([@yash9439](https://github.com/yash9439))  

## 📜 References  
[1] XFeat: Accelerated Features for Lightweight Image Matching  
[2] HPatches Dataset for Homography Estimation  
[3] MegaDepth Dataset for Feature Learning  
[4] COCO Dataset for Synthetic Image Warping  

## 🌟 Show Your Support  
If you find this project useful, please ⭐ the repo and contribute!  

--- 