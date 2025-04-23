import cv2
import numpy as np
import os
from glob import glob
import argparse
import torch
import torchvision.transforms as transforms
from modules.xfeat import XFeat

# Globals set at runtime
device = None
xfeat_model = None
scale_factors = [1.0]
use_ratio_test = False
ransac_thresh = 5.0
clahe_enabled = False

def compute_homography_accuracy(H_gt, H_est, corners):
    """
    Computes the Mean Homography Accuracy (MHA) as described in the paper.
    Returns:
      corner_errors, acc3, acc5, acc7
    """
    warped_gt = cv2.perspectiveTransform(
        corners.reshape(-1, 1, 2).astype(np.float32), H_gt)
    warped_est = cv2.perspectiveTransform(
        corners.reshape(-1, 1, 2).astype(np.float32), H_est)
    corner_errors = np.sqrt(
        np.sum((warped_gt - warped_est) ** 2, axis=2)
    ).flatten()
    mean_error = np.mean(corner_errors)
    acc3 = mean_error < 3
    acc5 = mean_error < 5
    acc7 = mean_error < 7
    return (corner_errors, acc3, acc5, acc7)


def apply_clahe(img):
    """Apply CLAHE to the L channel in LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def preprocess_for_xfeat(img):
    """Preprocess OpenCV image into tensor for XFeat."""
    if img.ndim == 2:
        t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3 and img.shape[2] == 3:
        t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")
    t = t / 255.0
    H, W = t.shape[-2:]
    H32, W32 = (H // 32) * 32, (W // 32) * 32
    if H32 != H or W32 != W:
        t = transforms.functional.resize(
            t, (H32, W32), interpolation=transforms.InterpolationMode.BILINEAR
        )
    return t.to(device)


def extract_features(image):
    """Extract features via XFeat at multiple scales, optional CLAHE."""
    img = apply_clahe(image) if clahe_enabled else image
    all_kpts, all_descs = [], []
    for s in scale_factors:
        img_s = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR) if s != 1.0 else img
        proc = preprocess_for_xfeat(img_s)
        res = xfeat_model.detectAndCompute(
            proc,
            top_k=xfeat_model.top_k,
            detection_threshold=xfeat_model.detection_threshold
        )[0]
        pts = res['keypoints'].cpu().numpy()
        desc = res['descriptors'].cpu().numpy()
        # rescale pts to original image
        pts /= s
        kpts = [cv2.KeyPoint(float(x), float(y), 1) for x, y in pts]
        all_kpts.extend(kpts)
        all_descs.append(desc)
    if len(all_descs) == 0:
        return [], None
    descriptors = np.vstack(all_descs)
    return all_kpts, descriptors


def match_features(desc1, desc2):
    """Match descriptors with optional ratio test."""
    if desc1 is None or desc2 is None or desc1.size == 0 or desc2.size == 0:
        return []
    d1 = desc1.astype(np.float32)
    d2 = desc2.astype(np.float32)
    if use_ratio_test:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = matcher.knnMatch(d1, d2, k=2)
        matches = [m for m, n in knn if m.distance < 0.75 * n.distance]
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(d1, d2)
    return matches


def run_hpatches_experiment(hpatches_path, output_file=None):
    """Runs the homography estimation experiment with XFeat."""
    if output_file:
        out_f = open(output_file, 'w')
    illum = sorted(glob(os.path.join(hpatches_path, 'i_*')))
    view = sorted(glob(os.path.join(hpatches_path, 'v_*')))
    results = {
        'illumination': {'MHA@3': [], 'MHA@5': [], 'MHA@7': []},
        'viewpoint':    {'MHA@3': [], 'MHA@5': [], 'MHA@7': []}
    }
    for seq_path in illum + view:
        seq = os.path.basename(seq_path)
        typ = 'illumination' if seq.startswith('i') else 'viewpoint'
        print(f"Processing {seq}")
        # load reference image
        image1 = None
        for ext in ('ppm', 'png', 'jpg', 'jpeg'):
            p = os.path.join(seq_path, f'1.{ext}')
            if os.path.exists(p):
                image1 = cv2.imread(p)
                break
        if image1 is None:
            continue
        h, w = image1.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        k1, d1 = extract_features(image1)
        for i in range(2, 7):
            image2 = None
            for ext in ('ppm', 'png', 'jpg', 'jpeg'):
                p = os.path.join(seq_path, f'{i}.{ext}')
                if os.path.exists(p):
                    image2 = cv2.imread(p)
                    break
            if image2 is None:
                continue
            k2, d2 = extract_features(image2)
            matches = match_features(d1, d2)
            if len(matches) < 4:
                continue
            src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H_est, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
            if H_est is None:
                continue
            inliers = int(mask.sum())
            H_gt = np.loadtxt(os.path.join(seq_path, f'H_1_{i}'))
            errs, a3, a5, a7 = compute_homography_accuracy(H_gt, H_est, corners)
            results[typ]['MHA@3'].append(a3)
            results[typ]['MHA@5'].append(a5)
            results[typ]['MHA@7'].append(a7)
            if output_file:
                out_f.write(f"{seq},1,{i},{inliers},{a3},{a5},{a7}\n")
    # summarize
    print("\nResults:")
    for t in results:
        for m in results[t]:
            results[t][m] = np.mean(results[t][m]) * 100
    if output_file:
        out_f.write("\nResults:\n")
    for t in ('illumination', 'viewpoint'):
        print(f" {t}:")
        if output_file:
            out_f.write(f" {t}:\n")
        for m in ('MHA@3', 'MHA@5', 'MHA@7'):
            print(f"   {m}: {results[t][m]:.2f}%")
            if output_file:
                out_f.write(f"   {m}: {results[t][m]:.2f}\n")
    if output_file:
        out_f.close()
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HPatches with XFeat & configurable params')
    parser.add_argument('hpatches_path', type=str,
                        help='Path to hpatches-sequences-release')
    parser.add_argument('--model_path', type=str, required=True,
                        help='XFeat checkpoint file')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    parser.add_argument('--top_k', type=int, default=4096,
                        help='Max keypoints for XFeat')
    parser.add_argument('--det_threshold', type=float, default=0.05,
                        help='Detection threshold for XFeat NMS')
    parser.add_argument('--ransac_thresh', type=float, default=5.0,
                        help='Reprojection threshold for RANSAC')
    parser.add_argument('--scale_factors', type=str, default='1.0',
                        help='Comma-separated scales, e.g. "0.5,1.0,1.5"')
    parser.add_argument('--use_ratio_test', action='store_true',
                        help="Enable Lowe's ratio test for matching")
    parser.add_argument('--clahe', action='store_true',
                        help='Apply CLAHE preprocessing')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file (optional)')
    args = parser.parse_args()

    # set globals from args
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    scale_factors = [float(s) for s in args.scale_factors.split(',')]
    use_ratio_test = args.use_ratio_test
    ransac_thresh = args.ransac_thresh
    clahe_enabled = args.clahe

    # load XFeat model
    xfeat_model = XFeat(
        weights=None,
        top_k=args.top_k,
        detection_threshold=args.det_threshold
    )
    ckpt = torch.load(args.model_path, map_location=device)
    sd = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    xfeat_model.net.load_state_dict(sd)
    xfeat_model = xfeat_model.to(device).eval()

    run_hpatches_experiment(
        args.hpatches_path,
        args.output
    )

