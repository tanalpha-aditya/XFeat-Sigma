# sfm.py
import numpy as np
import cv2
import torch
from modules.xfeat import XFeat

print("Loading XFeat model...")
try:
    xfeat = XFeat(top_k=8192)
    print(f"XFeat model loaded. Using device: {xfeat.dev}")
except FileNotFoundError:
    print("Error: XFeat weights not found. Check 'weights/xfeat.pt'")
    exit()
except Exception as e:
    print(f"Unexpected error loading XFeat: {e}")
    exit()

def detectFeatures_xfeat(image):
    frame_data = xfeat.detectAndCompute(image)[0]
    keypoints = frame_data['keypoints'].cpu().numpy().astype(np.float32)
    descriptors = frame_data['descriptors'].cpu().numpy()
    return keypoints, descriptors

def matchDescriptors(desc1, desc2, ratioThreshold):
    desc1 = np.asarray(desc1, dtype=np.float32)
    desc2 = np.asarray(desc2, dtype=np.float32)

    if len(desc1) == 0 or len(desc2) == 0:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []

    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratioThreshold * n.distance:
            good_matches.append((m.queryIdx, m.trainIdx))

    return good_matches

def estimatePose(keypointsOne, keypointsTwo, matchedPairs, cameraMatrix):
    if len(matchedPairs) < 5:
        raise ValueError("Not enough matches to compute Essential matrix.")

    pointsOne = np.float32([keypointsOne[m[0]] for m in matchedPairs])
    pointsTwo = np.float32([keypointsTwo[m[1]] for m in matchedPairs])

    E, mask_essential = cv2.findEssentialMat(pointsOne, pointsTwo, cameraMatrix, method=cv2.RANSAC, prob=0.8, threshold=5.0)
    if E is None or mask_essential.sum() < 5:
        raise ValueError("Essential matrix computation failed.")

    # Recover pose with inliers from essential matrix
    _, R, t, mask_recover = cv2.recoverPose(E, pointsOne, pointsTwo, cameraMatrix, mask=mask_essential)
    inlier_mask = (mask_essential.ravel() == 1) & (mask_recover.ravel() == 1)

    pts0 = pointsOne[inlier_mask]
    pts1 = pointsTwo[inlier_mask]

    return R, t, pts0, pts1

def triangulatePoints(R1, t1, R2, t2, points1, points2, cameraMatrix):
    if len(points1) == 0 or len(points2) == 0:
        return np.empty((0, 3))

    P1 = cameraMatrix @ np.hstack((R1, t1))
    P2 = cameraMatrix @ np.hstack((R2, t2))
    points4D = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points3D = (points4D[:3] / points4D[3]).T

    # Check cheirality (positive depth)
    points_hom = np.hstack((points3D, np.ones((len(points3D), 1))))
    proj1 = (P1 @ points_hom.T).T
    proj2 = (P2 @ points_hom.T).T
    z1 = proj1[:, 2]
    z2 = proj2[:, 2]
    mask = (z1 > 0) & (z2 > 0)

    # Filter points behind cameras
    points3D = points3D[mask]

    # Calculate reprojection errors
    projected1 = (proj1[:, :2].T / proj1[:, 2]).T
    error1 = np.linalg.norm(projected1 - points1[mask], axis=1)
    projected2 = (proj2[:, :2].T / proj2[:, 2]).T
    error2 = np.linalg.norm(projected2 - points2[mask], axis=1)

    # Filter points with high reprojection error
    max_error = 8.0
    error_mask = (error1 < max_error) & (error2 < max_error)
    points3D = points3D[error_mask]

    return points3D

def solvePNP(objectPoints, imagePoints, cameraMatrix):
    objPoints = np.asarray(objectPoints, dtype=np.float32)
    imgPoints = np.asarray(imagePoints, dtype=np.float32)

    if len(objPoints) < 4:
        return None, None

    success, rvec, tvec, inliers = cv2.solvePnPRansac(objPoints, imgPoints, cameraMatrix, None, flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.5, reprojectionError=5.0)
    if not success or inliers is None or len(inliers) < 6:
        return None, None

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec