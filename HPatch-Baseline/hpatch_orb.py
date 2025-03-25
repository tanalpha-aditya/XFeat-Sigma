import cv2
import numpy as np
import os
from glob import glob
import argparse

def compute_homography_accuracy(H_gt, H_est, corners):
    """
    Computes the Mean Homography Accuracy (MHA) as described in the paper.
    """
    warped_corners_gt = cv2.perspectiveTransform(corners.reshape(-1, 1, 2).astype(np.float32), H_gt)
    warped_corners_est = cv2.perspectiveTransform(corners.reshape(-1, 1, 2).astype(np.float32), H_est)
    corner_errors = np.sqrt(np.sum((warped_corners_gt - warped_corners_est) ** 2, axis=2)).flatten()
    mean_corner_error = np.mean(corner_errors)
    accuracy_at_3 = mean_corner_error < 3
    accuracy_at_5 = mean_corner_error < 5
    accuracy_at_7 = mean_corner_error < 7
    return corner_errors, accuracy_at_3, accuracy_at_5, accuracy_at_7


def extract_features(image, method_name):
    """Extracts features using ORB."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=4096)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def match_features(desc1, desc2, method_name):
    """Matches features using the appropriate matcher."""
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    matches = matcher.match(desc1, desc2)
    return matches


def run_hpatches_experiment(hpatches_path, method_name, output_file=None):
    """Runs the homography estimation experiment."""
    if output_file:
        out_f = open(output_file, 'w')

    illumination_sequences = sorted(glob(os.path.join(hpatches_path, 'i_*')))
    viewpoint_sequences = sorted(glob(os.path.join(hpatches_path, 'v_*')))
    results = {
        'illumination': {'MHA@3': [], 'MHA@5': [], 'MHA@7': []},
        'viewpoint': {'MHA@3': [], 'MHA@5': [], 'MHA@7': []}
    }
    all_sequences = illumination_sequences + viewpoint_sequences

    for seq_path in all_sequences:
        seq_name = os.path.basename(seq_path)
        print(f"Processing sequence: {seq_name}")
        image1_path = os.path.join(seq_path, '1.ppm')
        image1 = cv2.imread(image1_path)
        if image1 is None:
            image1_path = os.path.join(seq_path, '1.png')
            image1 = cv2.imread(image1_path)
        if image1 is None:
            print(f"  Could not read {image1_path}. Skipping.")
            continue

        h, w = image1.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])

        keypoints1, descriptors1 = extract_features(image1, method_name)
        if keypoints1 is None:
            continue

        seq_type = 'illumination' if seq_name.startswith('i') else 'viewpoint'

        for i in range(2, 7):
            image2_path = os.path.join(seq_path, f'{i}.ppm')
            image2 = cv2.imread(image2_path)
            if image2 is None:
                image2_path = os.path.join(seq_path, f'{i}.png')
                image2 = cv2.imread(image2_path)
            if image2 is None:
                print(f"  Could not read {image2_path}. Skipping.")
                continue

            keypoints2, descriptors2 = extract_features(image2, method_name)
            if keypoints2 is None:
                continue

            matches = match_features(descriptors1, descriptors2, method_name)
            if not matches:
                print(f"  No matches found for {seq_name} image pair 1-{i}.")
                continue

            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            H_est, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H_est is None:
                print(f"  Homography estimation failed for {seq_name} image pair 1-{i}.")
                continue
            inlier_matches = [m for i, m in enumerate(matches) if mask[i] == 1]
            num_inliers = np.sum(mask)

            H_gt_file = os.path.join(seq_path, f'H_1_{i}')
            try:
                H_gt = np.loadtxt(H_gt_file)
            except IOError:
                print(f"  Ground truth homography not found: {H_gt_file}")
                continue

            corner_errors, acc_3, acc_5, acc_7 = compute_homography_accuracy(H_gt, H_est, corners)
            results[seq_type]['MHA@3'].append(acc_3)
            results[seq_type]['MHA@5'].append(acc_5)
            results[seq_type]['MHA@7'].append(acc_7)

            if output_file:
                 out_f.write(f"{seq_name},1,{i},{num_inliers},{acc_3},{acc_5},{acc_7}\n")

    for seq_type in ['illumination', 'viewpoint']:
        for threshold in ['MHA@3', 'MHA@5', 'MHA@7']:
            results[seq_type][threshold] = np.mean(results[seq_type][threshold]) * 100

    print("\nResults:")
    if output_file:
        out_f.write("\nResults:\n")
    for seq_type in ['illumination', 'viewpoint']:
        print(f"  {seq_type}:")
        if output_file:
            out_f.write(f"  {seq_type}:\n")
        for threshold in ['MHA@3', 'MHA@5', 'MHA@7']:
            print(f"    {threshold}: {results[seq_type][threshold]:.2f}")
            if output_file:
                out_f.write(f"    {threshold}: {results[seq_type][threshold]:.2f}\n")
    if output_file:
        out_f.close()

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run HPatches homography estimation experiment with ORB.')
    parser.add_argument('hpatches_path', type=str, help='Path to the hpatches-sequences-release directory')
    parser.add_argument('--output', type=str, default=None, help='Path to the output file (optional)')
    args = parser.parse_args()
    run_hpatches_experiment(args.hpatches_path, 'orb', args.output)
