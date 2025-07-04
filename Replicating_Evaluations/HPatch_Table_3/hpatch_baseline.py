import cv2
import numpy as np
import os
from glob import glob
import argparse

from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch
from PIL import Image

def compute_homography_accuracy(H_gt, H_est, corners):
    """
    Computes the Mean Homography Accuracy (MHA) as described in the paper.

    Args:
        H_gt: Ground truth homography (3x3 numpy array).
        H_est: Estimated homography (3x3 numpy array).
        corners:  A numpy array of shape (4, 2) representing the four
                  corners of the reference image (in (x, y) format.

    Returns:
        A tuple: (corner_errors, accuracy_at_3, accuracy_at_5, accuracy_at_7)
                 where corner_errors is a list of per-corner errors in pixels,
                 and accuracy_at_X is a boolean indicating if the average
                 corner error is less than X pixels.
    """
    # Warp the corners of the reference image using the ground truth homography
    warped_corners_gt = cv2.perspectiveTransform(corners.reshape(-1, 1, 2).astype(np.float32), H_gt)

    # Warp the corners of the reference image using the estimated homography
    warped_corners_est = cv2.perspectiveTransform(corners.reshape(-1, 1, 2).astype(np.float32), H_est)

    # Calculate the Euclidean distance between the warped corners
    corner_errors = np.sqrt(np.sum((warped_corners_gt - warped_corners_est) ** 2, axis=2)).flatten()
    mean_corner_error = np.mean(corner_errors)

    accuracy_at_3 = mean_corner_error < 3
    accuracy_at_5 = mean_corner_error < 5
    accuracy_at_7 = mean_corner_error < 7

    return corner_errors, accuracy_at_3, accuracy_at_5, accuracy_at_7

def extract_features(image, method_name):
    """
    Extracts features and descriptors from an image using the specified method.

    Args:
        image: Input image (numpy array).
        method_name: Name of the feature extraction method ('orb', 'sift', 'superpoint', 'disk', 'alike', 'zippypoint').
                     Case-insensitive.

    Returns:
        A tuple: (keypoints, descriptors).  keypoints is a list of OpenCV KeyPoint objects.
                 descriptors is a numpy array of shape (num_keypoints, descriptor_dim).
                 Returns (None, None) if the method is not recognized or if an error occurs.
    """
    if method_name.lower() == 'orb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Ensure grayscale
        orb = cv2.ORB_create(nfeatures=4096) #Limit Keypoints
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors
    elif method_name.lower() == 'sift':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Ensure grayscale
        sift = cv2.SIFT_create(nfeatures=4096)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors

    elif method_name.lower() == 'superpoint':
        # Load SuperPoint model and processor
        processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
        model.eval()  # Set the model to evaluation mode

        # Convert image to RGB and then to PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        inputs = processor(pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            # Correct original_size to (width, height)
            original_size = (image.shape[1], image.shape[0])
            post_processed_outputs = processor.post_process_keypoint_detection(outputs, [original_size])

        if not post_processed_outputs[0]['keypoints'].numel():
            return [], None  # No keypoints detected

        keypoints_np = post_processed_outputs[0]['keypoints'].cpu().numpy()
        scores_np = post_processed_outputs[0]['scores'].cpu().numpy()
        descriptors_np = post_processed_outputs[0]['descriptors'].cpu().numpy()

        # Convert SuperPoint output to OpenCV KeyPoint objects
        keypoints = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1.0, response=float(score))
                     for kp, score in zip(keypoints_np, scores_np)]

        return keypoints, descriptors_np.T.astype(np.float32)  # Transpose and ensure float32

    elif method_name.lower() == 'disk':
        print("DISK implementation not provided in this example.")
        return [], None
    elif method_name.lower() == 'alike':
        print("ALIKE implementation not provided in this example.")
        return [], None
    elif method_name.lower() == 'zippypoint':
        print("ZippyPoint implementation not provided in this example.")
        return [], None
    else:
        print(f"Error: Unknown method '{method_name}'")
        return [], None

def match_features(desc1, desc2, method_name):
    """Matches features using the appropriate matcher for the given method."""
    if method_name.lower() in ('orb', 'zippypoint'):  # Binary descriptors
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:  # Float descriptors (SIFT, SuperPoint, DISK, ALIKE)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []

    matches = matcher.match(desc1, desc2)
    return matches

def run_hpatches_experiment(hpatches_path, method_name, output_file=None):
    """
    Runs the homography estimation experiment on the HPatches dataset.

    Args:
        hpatches_path: Path to the hpatches-sequences-release directory.
        method_name:  Name of the feature extraction method.
        output_file: (Optional) Path to a file to save the results.

    Returns:
        A dictionary containing the results (MHA, accuracy at different thresholds).
    """
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
        if image1 is None:  # Handle .png case
            image1_path = os.path.join(seq_path, '1.png')
            image1 = cv2.imread(image1_path)
        if image1 is None:
            print(f"  Could not read {image1_path}. Skipping.")
            continue

        h, w = image1.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])

        keypoints1, descriptors1 = extract_features(image1, method_name)
        if keypoints1 is None or len(keypoints1) == 0:
            print(f"  No keypoints detected for {seq_name} image 1. Skipping.")
            continue

        seq_type = 'illumination' if seq_name.startswith('i') else 'viewpoint'

        for i in range(2, 7):
            image2_path = os.path.join(seq_path, f'{i}.ppm')
            image2 = cv2.imread(image2_path)
            if image2 is None:  # Handle .png
                image2_path = os.path.join(seq_path, f'{i}.png')
                image2 = cv2.imread(image2_path)
            if image2 is None:
                print(f"  Could not read {image2_path}. Skipping.")
                continue

            keypoints2, descriptors2 = extract_features(image2, method_name)
            if keypoints2 is None or len(keypoints2) == 0:
                print(f"  No keypoints detected for {seq_name} image {i}. Skipping.")
                continue
            if descriptors1 is None or descriptors2 is None:
                print(f"Descriptors could not be extracted from image pair 1-{i}")
                continue

            matches = match_features(descriptors1, descriptors2, method_name)
            if not matches:
                print(f"  No matches found for {seq_name} image pair 1-{i}.")
                continue

            # Convert keypoints to numpy arrays for RANSAC
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate homography using RANSAC
            H_est, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H_est is None:
                print(f"  Homography estimation failed for {seq_name} image pair 1-{i}.")
                continue
            inlier_matches = [m for i, m in enumerate(matches) if mask[i] == 1]
            num_inliers = np.sum(mask)

            # Load ground truth homography
            H_gt_file = os.path.join(seq_path, f'H_1_{i}')
            try:
                H_gt = np.loadtxt(H_gt_file)
            except IOError:
                print(f"  Ground truth homography not found: {H_gt_file}")
                continue

            # Compute homography accuracy
            corner_errors, acc_3, acc_5, acc_7 = compute_homography_accuracy(H_gt, H_est, corners)

            results[seq_type]['MHA@3'].append(acc_3)
            results[seq_type]['MHA@5'].append(acc_5)
            results[seq_type]['MHA@7'].append(acc_7)

            if output_file:
                out_f.write(f"{seq_name},1,{i},{num_inliers},{int(acc_3)},{int(acc_5)},{int(acc_7)}\n")

    # Calculate overall MHA
    for seq_type in ['illumination', 'viewpoint']:
        for threshold in ['MHA@3', 'MHA@5', 'MHA@7']:
            if results[seq_type][threshold]:  # Check if there are entries to avoid division by zero
                results[seq_type][threshold] = np.mean(results[seq_type][threshold]) * 100
            else:
                results[seq_type][threshold] = 0.0

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
    parser = argparse.ArgumentParser(description='Run HPatches homography estimation experiment.')
    parser.add_argument('hpatches_path', type=str, help='Path to the hpatches-sequences-release directory')
    parser.add_argument('--method', type=str, default='orb',
                        help='Feature extraction method (orb, sift, superpoint, disk, alike, zippypoint)')
    parser.add_argument('--output', type=str, default=None, help='Path to the output file (optional)')

    args = parser.parse_args()
    run_hpatches_experiment(args.hpatches_path, args.method, args.output)
