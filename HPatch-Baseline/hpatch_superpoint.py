import cv2
import numpy as np
import os
from glob import glob
import argparse
from PIL import Image
import torch
from transformers import AutoImageProcessor, SuperPointForKeypointDetection

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

def extract_features_superpoint(image, processor, model):
    """
    Extract features using SuperPoint. The input is a cv2 image (BGR).
    The image is converted to a PIL image (RGB) before processing.
    Returns a list of cv2.KeyPoint objects and a numpy array of descriptors.
    """
    # Convert from BGR (OpenCV) to RGB and then to PIL Image.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Process the image with SuperPoint.
    inputs = processor(pil_image, return_tensors="pt")
    outputs = model(**inputs)
    image_size = (pil_image.height, pil_image.width)
    processed = processor.post_process_keypoint_detection(outputs, [image_size])[0]
    
    # Create cv2.KeyPoint objects. Each keypoint is defined by its (x, y) coordinate.
    keypoints = []
    for kp in processed["keypoints"]:
        # Here we assume kp is in (x, y) order.
        keypoints.append(cv2.KeyPoint(float(kp[0]), float(kp[1]), 1))
    
    # Get descriptors as a numpy array.
    descriptors = processed["descriptors"]
    if torch.is_tensor(descriptors):
        descriptors = descriptors.detach().cpu().numpy()
        
    return keypoints, descriptors

def match_features(desc1, desc2):
    """
    Matches features between two sets of descriptors using BFMatcher with L2 norm.
    Returns a list of matches.
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    return matches

def run_hpatches_experiment(hpatches_path, processor, model, output_file=None):
    """
    Runs the HPatches homography estimation experiment using SuperPoint.
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
        if image1 is None:
            image1_path = os.path.join(seq_path, '1.png')
            image1 = cv2.imread(image1_path)
        if image1 is None:
            print(f"  Could not read {image1_path}. Skipping.")
            continue

        h, w = image1.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        
        keypoints1, descriptors1 = extract_features_superpoint(image1, processor, model)
        if keypoints1 is None or descriptors1 is None:
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

            keypoints2, descriptors2 = extract_features_superpoint(image2, processor, model)
            if keypoints2 is None or descriptors2 is None:
                continue

            matches = match_features(descriptors1, descriptors2)
            if not matches:
                print(f"  No matches found for {seq_name} image pair 1-{i}.")
                continue

            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            H_est, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H_est is None or mask is None:
                print(f"  Homography estimation failed for {seq_name} image pair 1-{i}.")
                continue
            num_inliers = int(np.sum(mask))

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
            if results[seq_type][threshold]:
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
    parser = argparse.ArgumentParser(description='Run HPatches homography estimation experiment with SuperPoint.')
    parser.add_argument('hpatches_path', type=str, help='Path to the hpatches-sequences-release directory')
    parser.add_argument('--output', type=str, default=None, help='Path to the output file (optional)')
    args = parser.parse_args()
    
    # Load the SuperPoint processor and model from Hugging Face.
    processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
    model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
    
    # Optionally move the model to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    run_hpatches_experiment(args.hpatches_path, processor, model, args.output)

