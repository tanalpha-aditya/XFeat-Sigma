# main.py
import numpy as np
import cv2
from utils import loadImages, readCalibrationMatrix, writePLY
from sfm import detectFeatures_xfeat, matchDescriptors, estimatePose, triangulatePoints, solvePNP

def main():
    numberOfImages = 30
    images = loadImages("images", numberOfImages)
    cameraMatrix = readCalibrationMatrix("K.txt")

    features = []
    for image in images:
        keypoints, descriptors = detectFeatures_xfeat(image)
        features.append((keypoints, descriptors))

    keypoints0, descriptors0 = features[0]
    keypoints1, descriptors1 = features[1]
    matches01 = matchDescriptors(descriptors0, descriptors1, ratioThreshold=0.65)

    print(f"# Matches between image 0 and 1: {len(matches01)}")
    if len(matches01) < 5:
        print("[ERROR] Not enough matches between first two images.")
        return

    R_init, t_init, pts0, pts1 = estimatePose(keypoints0, keypoints1, matches01, cameraMatrix)

    cameraPoses = [
        (np.eye(3), np.zeros((3, 1))),
        (R_init, t_init)
    ]

    pointCloud = triangulatePoints(cameraPoses[0][0], cameraPoses[0][1], R_init, t_init, pts0, pts1, cameraMatrix).tolist()

    associations = {}
    for (i0, i1), pt in zip(matches01, pointCloud):
        associations[(0, i0)] = pt
        associations[(1, i1)] = pt

    for i in range(2, numberOfImages):
        keypointsPrev, descriptorsPrev = features[i-1]
        keypointsCurr, descriptorsCurr = features[i]

        matches = matchDescriptors(descriptorsPrev, descriptorsCurr, ratioThreshold=0.85)
        if len(matches) < 6:
            print(f"[WARNING] Not enough matches for image {i}. Reusing previous pose.")
            cameraPoses.append(cameraPoses[-1])
            continue

        # Collect 2D-3D correspondences for PnP
        objectPoints, imagePoints, pnpm_matches = [], [], []
        for m in matches:
            if (i-1, m[0]) in associations:
                objectPoints.append(associations[(i-1, m[0])])
                imagePoints.append(keypointsCurr[m[1]])
                pnpm_matches.append(m)

        if len(objectPoints) < 6:
            print(f"[WARNING] Not enough 2D-3D correspondences for image {i}. Reusing previous pose.")
            cameraPoses.append(cameraPoses[-1])
            continue

        # Solve PnP with RANSAC
        R, t = solvePNP(objectPoints, imagePoints, cameraMatrix)
        if R is None:
            print(f"[WARNING] solvePnP failed for image {i}. Reusing previous pose.")
            cameraPoses.append(cameraPoses[-1])
            continue

        cameraPoses.append((R, t))

        # Filter matches to only inliers from PnP
        pts1 = np.float32([keypointsPrev[m[0]] for m in pnpm_matches])
        pts2 = np.float32([keypointsCurr[m[1]] for m in pnpm_matches])

        # Triangulate new points between i-1 and i using only new matches
        matches_to_triangulate = []
        for m in matches:
            if (i-1, m[0]) not in associations and (i, m[1]) not in associations:
                matches_to_triangulate.append(m)

        if len(matches_to_triangulate) > 0:
            pts1_new = np.float32([keypointsPrev[m[0]] for m in matches_to_triangulate])
            pts2_new = np.float32([keypointsCurr[m[1]] for m in matches_to_triangulate])
            newPoints3D = triangulatePoints(cameraPoses[i-1][0], cameraPoses[i-1][1], R, t, pts1_new, pts2_new, cameraMatrix)

            # Add new associations
            for m, pt in zip(matches_to_triangulate, newPoints3D):
                associations[(i-1, m[0])] = pt.tolist()
                associations[(i, m[1])] = pt.tolist()
                pointCloud.append(pt.tolist())

    writePLY("mycloud.ply", pointCloud)
    print("Saved mycloud.ply with", len(pointCloud), "points.")

main()