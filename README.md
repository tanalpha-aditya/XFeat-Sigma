# XFeat: Analysis, Replication, and Enhancement 🚀

**Course:** CS7.505 - Computer Vision 2024-25, IIIT Hyderabad
**Authors:** Aditya Raghuvanshi ([@tanalpha-aditya](https://github.com/tanalpha-aditya)), Yash Bhaskar ([@yash9439](https://github.com/yash9439))

## 📌 Overview

This project provides a comprehensive analysis, replication, and enhancement of the **XFeat: Accelerated Features for Lightweight Image Matching** paper. XFeat proposes a lightweight CNN for efficient detection, description, and matching of local image features, offering both sparse (XFeat) and semi-dense (XFeat*) modes.

Our work focuses on:
1.  **Replicating** the baseline results reported in the XFeat paper across standard benchmarks (Megadepth-1500, ScanNet-1500, HPatches).
2.  **Analyzing** the impact of different training strategies and architectural variations (Ablation Studies).
3.  **Exploring** the performance of XFeat in various downstream tasks (Out-of-Domain Use Cases).
4.  **Proposing and Evaluating** several post-training improvements to enhance feature quality and matching robustness.

**Key Enhancements Explored:**
*   **Homography Transformation Augmentation:** Using multiple homographic projections during feature detection to potentially improve keypoint quality and robustness.
*   **RANSAC-Based Refinement:** Filtering keypoint correspondences using geometric constraints (Homography/Fundamental Matrix) to remove outliers.
*   **DBSCAN Clustering for Semi-Dense Matching:** Applying clustering to XFeat* semi-dense points to identify and remove spatial outliers before matching refinement.

For a detailed overview, methodology, and results, please see our presentation: [CV_Project_Presentation.pdf](./CV_Project_Presentation.pdf).

## 📂 Project Structure

The repository is organized as follows:

```
.
├── CV_Project_Presentation.pdf  # Project presentation slides
├── Data_Download_Scripts/       # Scripts for downloading COCO & Megadepth
├── Out_of_Domain_Usecase_Testing/ # Code for Frame Generation, Image-to-3D, Stitching, Video Tracking
│   ├── FrameGeneration/
│   ├── Image_to_3d_Usecase/
│   ├── Multi-Image_Stitching/
│   └── Video_Tracking_Usecase/
├── Post_Training_Improvements/  # Code and results for proposed enhancements (Homography, RANSAC, DBSCAN)
│   ├── accelerated_features/     # Base XFeat code used for improvements
│   ├── eval_megadepth.py        # Evaluation script on Megadepth
│   ├── qualitative.py           # Script for qualitative visualization
│   ├── Results*.txt             # Quantitative results for improvements
│   └── stdout*.txt              # Log files from improvement experiments
├── Project Related Documents/   # Proposal, Original Paper, Timeline
├── README.md                    # This file
├── Replicating_Evaluations/     # Scripts and results for replicating paper benchmarks
│   ├── HPatch_Table_3/          # HPatches evaluation code and results
│   ├── Megadepth1500_Table_1/   # Megadepth-1500 evaluation code and results
│   └── Scannet_Table_2/         # ScanNet-1500 evaluation code and results
├── requirements.txt             # Python dependencies
├── tree.txt                     # Full file tree listing
├── Videos/                      # Output videos from use-case testing
├── Xfeat_Architecture_Ablation_TrainingCode/ # Code for training/evaluating architectural/strategy variations
│   ├── Idea1_improvement_CBAM/  # Adding CBAM Attention
│   ├── Idea2_imporvement_DL/    # Adding Dropout/GroupNorm
│   ├── Idea3_improvement_CombiningAbove2Idea/ # Combining Idea 1 & 2
│   ├── Strategy_1_Default-Xfeat/ # Default training strategy code
│   ├── Strategy_2_No_Synthetic_Data/ # Training without COCO warped data
│   ├── Strategy_3_Smaller_model/   # Training a smaller capacity model
│   └── Strategy_4_Joint_keypoint_extraction/ # (Less developed) Exploration of joint extraction
└── XFeat_parent_repo/           # A copy of the original XFeat repository used as a base
```

## 🔬 Core Contributions & Findings

### 1. Replication Efforts
We successfully replicated the evaluation results presented in the original XFeat paper on multiple benchmarks:
*   **Megadepth-1500:** Relative pose estimation (Table 1 in paper).
*   **ScanNet-1500:** Indoor relative pose estimation (Table 2 / Table 6 in paper).
*   **HPatches:** Homography estimation and feature matching (Table 3 in paper).
We compared XFeat against baselines like ORB, SIFT, ALIKE, SuperPoint, DISK, etc., largely confirming the performance reported in the paper. See `Replicating_Evaluations/` for scripts and detailed results.

### 2. Architecture & Training Ablation
We investigated variations in the XFeat architecture and training process:
*   **Training Strategies:** Evaluated the impact of removing synthetic COCO data and using a smaller model architecture (halved channels). Results showed performance trade-offs as expected.
*   **Architectural Modifications:** Explored adding CBAM attention modules, replacing BatchNorm with GroupNorm + Dropout, and combining these ideas. These modifications generally did not lead to significant improvements over the baseline XFeat on the Megadepth benchmark, sometimes causing a drop in performance. See `Xfeat_Architecture_Ablation_TrainingCode/` for code and `Post_Training_Improvements/Results*.txt` / Presentation for results.

### 3. Post-Training Improvements
We implemented and evaluated several techniques applied *after* standard XFeat feature extraction/matching:
*   **Homography Transformation:** Detecting features on multiple warped versions of an image and unifying/intersecting them. Intersection (`intersect`) maintained performance while significantly reducing keypoint count; unification (`unify`) drastically increased keypoints but decreased matching accuracy.
*   **RANSAC Refinement:** Using `findHomography` or `findFundamentalMat` to filter matches. This consistently provided a small boost in pose estimation accuracy (AUC scores) on Megadepth with minimal overhead.
*   **DBSCAN Clustering (XFeat*):** Applied to the semi-dense matches from XFeat* before refinement. This slightly improved AUC scores by removing spatial outliers, but incurred a notable computational cost.
See `Post_Training_Improvements/` for scripts and detailed results tables in the Presentation.

### 4. Out-of-Domain Use Cases
We tested XFeat's applicability in diverse computer vision tasks:
*   **Frame Generation:** Used XFeat matches to estimate motion vectors for video frame interpolation.
*   **Image-to-3D (Structure from Motion):** Built a basic SfM pipeline using XFeat features.
*   **Multi-Image Stitching:** Created panoramas by finding matches and transformations with XFeat.
*   **Visual Odometry Tracking:** Compared XFeat's tracking performance against ORB and SIFT in real-time video.
These tests demonstrated XFeat's versatility as a general-purpose feature matcher. See `Out_of_Domain_Usecase_Testing/` for code and `Videos/` for demos.

## 🛠️ Setup & Usage

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: Ensure you have compatible PyTorch and CUDA versions installed if using GPU.)*

### 3. Download Datasets
*   **Megadepth:** Follow instructions from the official [MegaDepth repository](https://github.com/mihaidusmanu/megadepth). Download scripts are also provided in `Data_Download_Scripts/Megadepth_parallel_adaDownload.sh`. (~150GB+)
*   **COCO:** Download using `Data_Download_Scripts/coco_parallel_adaDownload.sh` or from the [COCO dataset website](https://cocodataset.org/#download). Needed for training with synthetic data.
*   **ScanNet:** Use `Replicating_Evaluations/Scannet_Table_2/download_scannet1500.sh`.
*   **HPatches:** Use `Replicating_Evaluations/HPatch_Table_3/download_hpatch.sh`.

Update relevant paths in configuration files or script arguments as needed.

### 4. Running Evaluations & Experiments (Examples)

*   **Megadepth Evaluation (Baseline):**
    ```bash
    # Navigate to the appropriate directory if needed
    python Post_Training_Improvements/eval_megadepth.py --weights_path XFeat_parent_repo/weights/xfeat.pt --megadepth_path /path/to/megadepth/ --megadepth_pairs Post_Training_Improvements/accelerated_features/assets/megadepth_1500.json
    ```
*   **HPatches Evaluation (XFeat):**
    ```bash
    cd Replicating_Evaluations/HPatch_Table_3/Xfeat_inference/
    python hpatch_xfeat.py --weights weights/xfeat.pt --data_dir /path/to/hpatches-sequences-release/
    ```
*   **Running Training (e.g., Default Strategy):**
    ```bash
    cd Xfeat_Architecture_Ablation_TrainingCode/Strategy_1_Default-Xfeat/
    bash train_megadepth.sh # Modify script with correct data paths first
    ```
*   **Running Post-Training Improvement Evaluation (e.g., RANSAC Refined):**
    ```bash
    # Modify eval_megadepth.py or similar scripts to include the refinement step.
    # Example modification within eval_megadepth.py (conceptual):
    # matches = xfeat.match(...)
    # mkpts0, mkpts1 = filter_matches_ransac(kpts0, kpts1, matches, method='homography') # Add this step
    # proceed with evaluation using mkpts0, mkpts1
    python Post_Training_Improvements/eval_megadepth.py ... # Run the modified script
    ```

Refer to the specific scripts within each directory for detailed arguments and usage.

## 챌 Challenges Faced

*   **Reproducibility:** Code for some baseline methods and ablation studies mentioned in the original paper was not available, requiring reimplementation or reliance on third-party versions.
*   **Architectural Details:** Specifics for certain ablations (like the "Smaller model") were limited in the original paper.
*   **Data Handling:** The large size of the Megadepth dataset presented data management and processing challenges.

## 📜 References

*   **Main Paper:** Pánek, V., Zdimal, D., & Cech, J. (2024). *XFeat: Accelerated Features for Lightweight Image Matching*. arXiv preprint arXiv:2401.01901. [Link to original repo: XFeat_parent_repo/ or official repo if different]
*   **Megadepth:** Li, Z., & Snavely, N. (2018). *Megadepth: Learning single-view depth prediction from internet photos*. In CVPR.
*   **ScanNet:** Dai, A., Chang, A. X., Savva, M., Halber, M., Funkhouser, T., & Nießner, M. (2017). *Scannet: Richly-annotated 3d reconstructions of indoor scenes*. In CVPR.
*   **HPatches:** Balntas, V., Lenc, K., Vedaldi, A., & Mikolajczyk, K. (2017). *HPatches: A benchmark and evaluation of handcrafted and learned local descriptors*. In CVPR.
*   **COCO:** Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., ... & Zitnick, C. L. (2014). *Microsoft coco: Common objects in context*. In ECCV.

(Additional references can be found in the presentation)

## 🌟 Acknowledgements

This project was undertaken as part of the Computer Vision course (CS7.505) at IIIT Hyderabad. We thank the instructors and TAs for their guidance.
