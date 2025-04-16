import cv2
import copy
import torch
import numpy as np
from scipy.spatial import cKDTree
from accelerated_features.modules import xfeat

class XFeatWrapper():

    def __init__(self, device = "cpu", top_k = 4096, min_cossim = -1):
        self.device = device
        self.xfeat_instance = xfeat.XFeat()
        self.top_k = top_k
        self.min_cossim = min_cossim


    def detect_feature_sparse(self, image, top_k = None):
        '''
        Detects keypoints, descriptors and reliability map for sparse matching (XFeat).
        input:
            image -> np.ndarray (H,W,C): grayscale or rgb image
        return:
            Dict:
                'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
                'scores'       ->   torch.Tensor(N): keypoint scores
                'descriptors'  ->   torch.Tensor(N, 64): local features
        '''
        if top_k is None:
            top_k = self.top_k

        output = self.xfeat_instance.detectAndCompute(image, top_k)
        return output[0]


    def detect_feature_dense(self, imset, top_k = None, multiscale = True):
        '''
        Detects keypoints, descriptors and reliability map for semi-dense matching (XFeat Star).
        It works in batched mode because it use different scales of the image.
        input:
            imset -> torch.Tensor(B, C, H, W): grayscale or rgb image
        return:
            Dict:
                'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
                'scores'       ->   torch.Tensor(N): keypoint scores
                'descriptors'  ->   torch.Tensor(N, 64): local features
        '''

        if top_k is None: top_k = self.top_k

        imset = self.parse_input(imset)

        output = self.xfeat_instance.detectAndComputeDense(imset, top_k = top_k, multiscale=multiscale)

        output_ret = {}
        for key in output.keys():
            if key == "scales":
                output_ret["scores"] = output[key].squeeze(0)
            else:
                output_ret[key] = output[key].squeeze(0)

        return output_ret


    def match_xfeat_star_original(self, imset1, imset2, top_k = None):
        """
			Extracts coarse feats, then match pairs and finally refine matches, currently supports batched mode.
			input:
				im_set1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				im_set2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				top_k -> int: keep best k features
			returns:
				matches -> List[torch.Tensor(N, 4)]: List of size B containing tensor of pairwise matches (x1,y1,x2,y2)
		"""

        if top_k is None: top_k = self.top_k

        imset1 = self.parse_input(imset1)
        imset2 = self.parse_input(imset2)


        return self.xfeat_instance.match_xfeat_star(imset1, imset2, top_k=top_k)


    def match_xfeat_original(self, image1, image2, top_k = None):
        """
			Simple extractor and MNN matcher.
			For simplicity it does not support batched mode due to possibly different number of kpts.
			input:
				img1 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
				img2 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
				top_k -> int: keep best k features
			returns:
				mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
		"""

        if top_k is None: top_k = self.top_k
        image1 = self.parse_input(image1)
        image2 = self.parse_input(image2)

        return self.xfeat_instance.match_xfeat(image1, image2, top_k=top_k)


    def parse_input(self, x):
        '''
            Parse the input to the correct format
            return:
                x -> torch.Tensor (B, C, H, W)
        '''
        if len(x.shape) == 3:
            x = x[None, ...]

        if isinstance(x, np.ndarray):
            x = torch.tensor(x).permute(0,3,1,2)/255

        return x


# UNIFY FEATURES WITH HOMOGRAPHT TRANSFORMATION
############################################################################################################
    def get_homography(self, type_transformation, image):
        '''
            Create the homography matrix for the trasformation given in input
            input:
                type_transformation -> Dict:
                    'type': rotation - traslation
                    'angle': grades
                    'pixel': traslation pixels number
                image -> np.ndarray (H,W,C): grayscale or rgb image
            return:
                homography_matrix -> np.ndarray (3,3): homography matrix
        '''

        homography_matrix = None

        if type_transformation["type"] == "rotation":
            angle = type_transformation["angle"]
            theta = np.radians(angle)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            (h, w) = image.shape[:2]

            center_x, center_y = w / 2, h / 2
            translation_to_center = np.array([[1, 0, -center_x],
                                            [0, 1, -center_y],
                                            [0, 0, 1]])
            rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                        [sin_theta, cos_theta, 0],
                                        [0, 0, 1]])
            translation_back = np.array([[1, 0, center_x],
                                        [0, 1, center_y],
                                        [0, 0, 1]])
            homography_matrix = translation_back @ rotation_matrix @ translation_to_center

        elif type_transformation["type"] == "traslation":
            # Placeholder for translation if needed in the future
            tx = type_transformation.get("pixel_x", 0) # Example: get x translation, default 0
            ty = type_transformation.get("pixel_y", 0) # Example: get y translation, default 0
            homography_matrix = np.array([[1, 0, tx],
                                          [0, 1, ty],
                                          [0, 0, 1]])
        else:
            # Return the transformation dictionary itself if type is not recognized
            # Or raise an error: raise ValueError("Unsupported transformation type")
            return type_transformation

        return homography_matrix


    def get_image_trasformed(self, image, homography_matrix):
        '''
            Apply the homography matrix to the image
            input:
                image -> np.ndarray (H,W,C): grayscale or rgb image
                homography_matrix -> np.ndarray (3,3): homography matrix
            return:
                trasformed_image -> np.ndarray (H,W,C): grayscale or rgb image
            '''
        trasformed_image = copy.deepcopy(image)
        (h, w) = image.shape[:2]
        trasformed_image = cv2.warpPerspective(trasformed_image, homography_matrix, (w, h))
        return trasformed_image


    def filter_points(self, transformed_points_coords, target_keypoints_coords, threshold=5, merge=False):
        '''
            Filter or unify the points that are near to each other using spatial proximity.
            input:
                transformed_points_coords -> np.ndarray (N, 2): coordinates of points transformed from another image/view.
                target_keypoints_coords -> np.ndarray (M, 2): coordinates of keypoints in the target image to compare against.
                threshold -> float: distance threshold in pixels.
                merge -> bool:
                    if True (Intersect): returns indices of transformed_points that ARE close to a target_keypoint.
                    if False (Unify): returns indices of transformed_points that ARE NOT close to any target_keypoint.

        '''
        if target_keypoints_coords.shape[0] == 0: # Handle empty target keypoints
             return list(range(transformed_points_coords.shape[0])) if not merge else []

        if len(target_keypoints_coords.shape) == 3: # Should already be (N, 2) but ensure
             target_keypoints_coords = target_keypoints_coords.squeeze(0)

        tree = cKDTree(target_keypoints_coords)

        # Query for nearest neighbor within the threshold for all transformed points at once
        distances, indices = tree.query(transformed_points_coords, k=1, distance_upper_bound=threshold)

        # distances will be inf if no neighbor is within threshold
        if merge: # Intersect: Keep points that *have* a close neighbor (distance < threshold)
            idx_ret = np.where(distances < threshold)[0].tolist()
        else: # Unify: Keep points that *do not* have a close neighbor (distance == inf or >= threshold)
            idx_ret = np.where(distances >= threshold)[0].tolist()

        return idx_ret


    def unify_features(self, features1, features2, homography, merge=False):
        '''
            Unify or Intersect features based on transformed keypoint proximity.
            If merge=True (Intersect): Keep only features from features1 whose keypoints,
                                      when transformed by homography, land close to a keypoint in features2.
            If merge=False (Unify): Keep all features from features1, plus features from features2 whose keypoints,
                                   when transformed by the *inverse* homography, do *not* land close to a keypoint in features1.
                                   (Simplified: keep features1 + unique features2).
            input:
                features1 -> Dict:{keypoints, scores, descriptors} (considered primary or base set)
                features2 -> Dict:{keypoints, scores, descriptors} (features from the transformed image)
                homography -> np.ndarray (3,3): Homography mapping points from image1 coordinate system to image2 coordinate system.
                merge -> bool: if True intersect the points, if False unify the points
            return:
                Dict:{keypoints, scores, descriptors} (combined/filtered features)
        '''

        kpts1_tensor = features1["keypoints"]
        kpts2_tensor = features2["keypoints"]
        kpts1_np = kpts1_tensor.cpu().numpy()
        kpts2_np = kpts2_tensor.cpu().numpy()

        if kpts1_np.shape[0] == 0: return features2 # If features1 is empty, return features2
        if kpts2_np.shape[0] == 0: return features1 # If features2 is empty, return features1

        if merge: # --- INTERSECTION ---
            # Transform points from features1 space to features2 space
            homogeneous_points1 = np.hstack([kpts1_np, np.ones((kpts1_np.shape[0], 1))])
            transformed_points1 = (homography @ homogeneous_points1.T).T
            transformed_points1 /= transformed_points1[:, 2][:, np.newaxis] # Normalize

            # Find which transformed points from features1 are close to *any* point in features2
            idx_selected = self.filter_points(transformed_points1[:, :2], kpts2_np, threshold=5, merge=True)

            # Select only the features from features1 corresponding to these indices
            keypoints_selected = features1["keypoints"][idx_selected]
            scores_selected = features1["scores"][idx_selected]
            descriptors_selected = features1["descriptors"][idx_selected]

        else: # --- UNIFICATION (Keep all of features1 + unique features from features2) ---
            # Need inverse homography to map points from features2 space back to features1 space
            try:
                inv_homography = np.linalg.inv(homography)
            except np.linalg.LinAlgError:
                print("Warning: Homography matrix is singular, cannot compute inverse. Returning features1 only.")
                return features1

            # Transform points from features2 space back to features1 space
            homogeneous_points2 = np.hstack([kpts2_np, np.ones((kpts2_np.shape[0], 1))])
            transformed_points2 = (inv_homography @ homogeneous_points2.T).T
            transformed_points2 /= transformed_points2[:, 2][:, np.newaxis] # Normalize

            # Find which points from features2, when transformed back, are *not* close to any point in features1
            idx_unique_in_f2 = self.filter_points(transformed_points2[:, :2], kpts1_np, threshold=5, merge=False) # merge=False to get non-overlapping

            # Combine all features from features1 with the unique features from features2
            keypoints_selected = torch.cat((features1["keypoints"], features2["keypoints"][idx_unique_in_f2]), dim=0)
            scores_selected = torch.cat((features1["scores"], features2["scores"][idx_unique_in_f2]), dim=0)
            descriptors_selected = torch.cat((features1["descriptors"], features2["descriptors"][idx_unique_in_f2]), dim=0)


        return {"keypoints": keypoints_selected,
                "scores": scores_selected,
                "descriptors": descriptors_selected}


    def trasformed_detection_features(self, image, trasformations, merge=False, top_k = None):
        '''
            Take an image, apply transformations, detect features on each, and unify/intersect them.
            input:
                image -> np.ndarray (H,W,C): grayscale or rgb image
                trasformations -> List[Dict]: List of transformation parameters.
                merge -> bool: Passed to unify_features (True=Intersect, False=Unify).
                top_k -> int: Number of keypoints to detect per image (original and transformed).
            return:
                Dict:{keypoints, scores, descriptors} (Final combined/filtered features)

        '''
        # Detect features on the original image
        features_aggregated = self.detect_feature_sparse(image, top_k=top_k)

        if not trasformations: # If no transformations provided, return original features
            return features_aggregated

        for trasformation in trasformations:
            homography = self.get_homography(trasformation, image)
            if not isinstance(homography, np.ndarray): # Check if get_homography returned a valid matrix
                print(f"Warning: Could not compute homography for transformation {trasformation}. Skipping.")
                continue

            image_transformed = self.get_image_trasformed(image, homography)

            # Detect features on the transformed image
            features_transformed = self.detect_feature_sparse(image_transformed, top_k=top_k)

            # Unify/Intersect features_aggregated with features_transformed
            # Homography maps from original image space to transformed image space
            features_aggregated = self.unify_features(features_aggregated, features_transformed, homography, merge=merge)

        return features_aggregated


    def match_xfeat_trasformed(self, image1, image2, trasformations = None, top_k=4092, min_cossim = None, merge=True):
        '''
            Detect features on original and transformed versions of images, unify/intersect them, then match.
            input:
                image1 -> np.ndarray (H,W,C): grayscale or rgb image
                image2 -> np.ndarray (H,W,C): grayscale or rgb image
                trasformations -> List[Dict]: List of transformation parameters to apply to *each* image.
                top_k -> int: Keypoints to detect per view.
                min_cossim -> float: Minimum cosine similarity for matching.
                merge -> bool: Passed to unify_features (True=Intersect, False=Unify).
            return:
                points1 -> np.ndarray (N, 2): matched points in image1
                points2 -> np.ndarray (N, 2): matched points in image2
        '''
        if trasformations is None:
             trasformations = [] # Ensure it's an empty list if None

        # Process image 1: Detect original + transformed features, then combine
        features_image1 = self.trasformed_detection_features(image1, trasformations, merge=merge, top_k=top_k)
        # Process image 2: Detect original + transformed features, then combine
        features_image2 = self.trasformed_detection_features(image2, trasformations, merge=merge, top_k=top_k)

        # Check if any features were found
        if features_image1['keypoints'].shape[0] == 0 or features_image2['keypoints'].shape[0] == 0:
            print("Warning: No keypoints found in one or both images after transformation/unification.")
            return np.empty((0, 2)), np.empty((0, 2)) # Return empty arrays

        kpts1, descs1 = features_image1['keypoints'], features_image1['descriptors']
        kpts2, descs2 = features_image2['keypoints'], features_image2['descriptors']

        if min_cossim is None: min_cossim = self.min_cossim

        # Perform matching between the final feature sets
        idx0, idx1 = self.xfeat_instance.match(descs1, descs2, min_cossim=min_cossim)

        points1 = kpts1[idx0].cpu().numpy()
        points2 = kpts2[idx1].cpu().numpy()

        return points1, points2


    def trasformed_detection_features_dense(self, imset, trasformations, merge=True, top_k = None, multiscale = True):
        '''
        Dense version of trasformed_detection_features. Operates on batches (imset).
        input:
            imset -> torch.Tensor(B, C, H, W): Batch of images.
            trasformations -> List[Dict]: Transformations to apply.
            merge -> bool: Passed to unify_features.
            top_k -> int: Keypoints per view.
            multiscale -> bool: Use multiscale detection for dense features.
        return:
            Dict:{keypoints, scores, descriptors} (Combined/filtered features for the batch - currently assumes B=1)
        '''
        # NOTE: This dense version currently assumes a batch size B=1 for simplicity in handling transformations.
        # Proper batch handling would require applying transforms/unification per image in the batch.
        if imset.shape[0] > 1:
            print("Warning: Dense transformed detection currently supports B=1. Using only the first image.")
            imset = imset[:1] # Process only the first image

        image_np = imset.permute(0, 2, 3, 1).squeeze(0).cpu().numpy() # Convert tensor to numpy for CV functions
        image_np = (image_np * 255).astype(np.uint8) # Assuming input tensor was 0-1 range

        # Detect features on the original image (tensor input needed for dense)
        features_aggregated = self.detect_feature_dense(imset, top_k, multiscale)

        if not trasformations:
            return features_aggregated

        for trasformation in trasformations:
            homography = self.get_homography(trasformation, image_np)
            if not isinstance(homography, np.ndarray):
                 print(f"Warning: Could not compute homography for transformation {trasformation}. Skipping.")
                 continue

            image_transformed_np = self.get_image_trasformed(image_np, homography)

            # Detect dense features on the transformed image (needs tensor input)
            image_transformed_tensor = torch.tensor(image_transformed_np).permute(2, 0, 1).unsqueeze(0) / 255.0
            image_transformed_tensor = image_transformed_tensor.to(self.device) # Move to appropriate device if needed
            features_transformed = self.detect_feature_dense(image_transformed_tensor, top_k, multiscale)

            # Unify/Intersect features
            features_aggregated = self.unify_features(features_aggregated, features_transformed, homography, merge=merge)

        return features_aggregated


    def match_xfeat_star_trasformed(self, imset1, imset2, trasformations, top_k = None):
        '''
        Dense matching (XFeat*) with prior transformation-based feature unification/intersection.
        input:
            imset1, imset2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): Input images/batches.
            trasformations -> List[Dict]: Transformations to apply.
            top_k -> int: Keypoints per view.
        return:
             Tuple[np.ndarray, np.ndarray] or List[torch.Tensor]: Matches (format depends on original match_xfeat_star output).
        '''
        # NOTE: Assumes B=1 due to limitations in trasformed_detection_features_dense batch handling.
        if top_k is None: top_k = self.top_k
        imset1 = self.parse_input(imset1).to(self.device) # Ensure tensor and on device
        imset2 = self.parse_input(imset2).to(self.device)

        if imset1.shape[0] > 1 or imset2.shape[0] > 1:
            print("Warning: match_xfeat_star_trasformed currently supports B=1. Using only the first image pair.")
            imset1 = imset1[:1]
            imset2 = imset2[:1]
        B = 1 # Effective batch size is 1

        # Get aggregated features for each image
        feature_images1 = self.trasformed_detection_features_dense(imset1, trasformations, merge=True, top_k=top_k, multiscale = True)
        feature_images2 = self.trasformed_detection_features_dense(imset2, trasformations, merge=True, top_k=top_k, multiscale = True)

        # Check for empty features
        if feature_images1['keypoints'].shape[0] == 0 or feature_images2['keypoints'].shape[0] == 0:
            print("Warning: No keypoints found in one or both images after dense transformation/unification.")
            # Return empty result in the expected format (tuple of numpy arrays for B=1)
            return np.empty((0, 2)), np.empty((0, 2))

        # Prepare feature dictionaries for matching function (needs batch dimension)
        feat1 = {}
        feat2 = {}
        # Ensure all feature components are unsqueezed to have a batch dimension
        for key in feature_images1:
            if key == "scores": # Handle 'scores' which maps to 'scales'
                 feat1["scales"] = feature_images1[key].unsqueeze(0)
                 feat2["scales"] = feature_images2[key].unsqueeze(0)
            elif key in ["keypoints", "descriptors"]: # These should exist
                 feat1[key] = feature_images1[key].unsqueeze(0)
                 feat2[key] = feature_images2[key].unsqueeze(0)
            # else: # Handle potential other keys if necessary
            #     feat1[key] = feature_images1[key].unsqueeze(0)
            #     feat2[key] = feature_images2[key].unsqueeze(0)


        # Match batches of pairs (B=1 here)
        idxs_list = self.xfeat_instance.batch_match(feat1['descriptors'], feat2['descriptors'] )

        # Refine coarse matches
        matches = []
        for b in range(B): # Loop will run once for B=1
            # Need to ensure feat1/feat2 have all keys expected by refine_matches
            # (keypoints, descriptors, scales might be needed)
            matches.append(self.xfeat_instance.refine_matches(feat1, feat2, matches=idxs_list, batch_idx=b))

        # Return in the expected format (tuple of numpy arrays for B=1)
        # The original function returns list for B>1, tensor for B=1. We adapt to return numpy arrays like sparse version.
        if B > 1:
             # This part is currently unreachable due to B=1 constraint, but kept for structure
             return matches # List of tensors
        else:
             if matches and matches[0].numel() > 0: # Check if matches list is not empty and tensor has elements
                 return (matches[0][:, :2].cpu().detach().numpy(), matches[0][:, 2:].cpu().detach().numpy())
             else:
                 return np.empty((0, 2)), np.empty((0, 2)) # Return empty numpy arrays if no matches


# REFINE FEATURES WITH FUNDAMENTAL AND HOMOGRAPHY
############################################################################################################
    def match_xfeat_refined(self, imset1, imset2, top_k=None, threshold=90, iterations=1, method="homography"):
        '''
            Get initial matches with XFeat sparse, then refine using geometric constraints (Homography/Fundamental Matrix).
            input:
                imset1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
                imset2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
                top_k -> int: keep best k features for initial matching.
                threshold -> float: RANSAC reprojection threshold in pixels.
                iterations -> int: number of refinement iterations.
                method -> str: "homography" or "fundamental".
            return:
                refined_pts1 -> np.ndarray (N, 2): refined points in image1
                refined_pts2 -> np.ndarray (N, 2): refined points in image2
        '''
        # Get initial raw matches using the standard sparse matcher
        raw_pts1, raw_pts2 = self.match_xfeat_original(imset1, imset2, top_k)

        if raw_pts1.shape[0] < 4: # Need at least 4 points for Homography, 7 for Fundamental
             print(f"Warning: Insufficient initial matches ({raw_pts1.shape[0]}) for geometric refinement. Returning raw matches.")
             return raw_pts1, raw_pts2

        refined_pts1, refined_pts2 = raw_pts1, raw_pts2
        for i in range(iterations):
            current_count = refined_pts1.shape[0]
            if current_count < 4 or (method == "fundamental" and current_count < 7):
                print(f"Warning: Insufficient points ({current_count}) for refinement in iteration {i+1}.")
                break # Stop iterating if too few points remain

            if method == "homography":
                refined_pts1, refined_pts2 = self.filter_by_Homography(refined_pts1, refined_pts2, threshold=threshold)
            elif method == "fundamental":
                refined_pts1, refined_pts2 = self.filter_by_Fundamental(refined_pts1, refined_pts2, threshold=threshold)
            else:
                 raise ValueError("Refinement method must be 'homography' or 'fundamental'")

            #print(f"Iteration {i+1}: {len(refined_pts1)} matches")
            if refined_pts1.shape[0] == current_count: # Stop if no points were removed
                 #print(f"No change in matches after iteration {i+1}. Stopping refinement.")
                 break

        return refined_pts1, refined_pts2


    def match_xfeat_star_refined(self, imset1, imset2, top_k=None, threshold=90, iterations=1, method="homography"):
        '''
            Get initial matches with XFeat* dense, then refine using geometric constraints (Homography/Fundamental Matrix).
            input:
                imset1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
                imset2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
                top_k -> int: keep best k features for initial matching.
                threshold -> float: RANSAC reprojection threshold in pixels.
                iterations -> int: number of refinement iterations.
                method -> str: "homography" or "fundamental".
            return:
                refined_pts1 -> np.ndarray (N, 2): refined points in image1
                refined_pts2 -> np.ndarray (N, 2): refined points in image2
        '''
        # Get initial raw matches using the standard dense matcher
        raw_pts1, raw_pts2 = self.match_xfeat_star_original(imset1, imset2, top_k)

        if raw_pts1.shape[0] < 4: # Need at least 4 points for Homography, 7 for Fundamental
             print(f"Warning: Insufficient initial matches ({raw_pts1.shape[0]}) for geometric refinement. Returning raw matches.")
             return raw_pts1, raw_pts2

        refined_pts1, refined_pts2 = raw_pts1, raw_pts2
        for i in range(iterations):
            current_count = refined_pts1.shape[0]
            if current_count < 4 or (method == "fundamental" and current_count < 7):
                print(f"Warning: Insufficient points ({current_count}) for refinement in iteration {i+1}.")
                break # Stop iterating if too few points remain

            if method == "homography":
                refined_pts1, refined_pts2 = self.filter_by_Homography(refined_pts1, refined_pts2, threshold=threshold)
            elif method == "fundamental":
                refined_pts1, refined_pts2 = self.filter_by_Fundamental(refined_pts1, refined_pts2, threshold=threshold)
            else:
                 raise ValueError("Refinement method must be 'homography' or 'fundamental'")

            #print(f"Iteration {i+1}: {len(refined_pts1)} matches")
            if refined_pts1.shape[0] == current_count: # Stop if no points were removed
                 #print(f"No change in matches after iteration {i+1}. Stopping refinement.")
                 break

        return refined_pts1, refined_pts2


    def filter_by_Fundamental(self, pts1, pts2, threshold):
        '''
            Filter point correspondences using the Fundamental Matrix estimated with RANSAC.
            input:
                pts1 -> np.ndarray (N, 2): points in the first image
                pts2 -> np.ndarray (N, 2): corresponding points in the second image
                threshold -> float: RANSAC reprojection threshold.
            return:
                inlier_pts1 -> np.ndarray (M, 2): inlier points in the first image (M <= N)
                inlier_pts2 -> np.ndarray (M, 2): corresponding inlier points in the second image
        '''
        if pts1.shape[0] < 7: # Need at least 7 points to estimate Fundamental matrix
             print("Warning: Less than 7 points provided for Fundamental matrix estimation.")
             return pts1, pts2 # Return original points if not enough

        # Ensure points are float32 as expected by OpenCV
        pts1_f32 = np.float32(pts1)
        pts2_f32 = np.float32(pts2)

        F, mask = cv2.findFundamentalMat(pts1_f32, pts2_f32, method=cv2.FM_RANSAC, ransacReprojThreshold=threshold, confidence=0.99)

        if F is None or mask is None:
            # print("Warning: Fundamental matrix estimation failed. Returning original points.")
            return pts1, pts2 # Return original points if estimation fails

        mask = mask.ravel().astype(bool) # Convert mask to boolean array
        # Filter matches based on inliers
        return pts1[mask], pts2[mask]


    def filter_by_Homography(self, pts1, pts2, threshold):
        '''
            Filter point correspondences using the Homography Matrix estimated with RANSAC.
            input:
                pts1 -> np.ndarray (N, 2): points in the first image
                pts2 -> np.ndarray (N, 2): corresponding points in the second image
                threshold -> float: RANSAC reprojection threshold.
            return:
                inlier_pts1 -> np.ndarray (M, 2): inlier points in the first image (M <= N)
                inlier_pts2 -> np.ndarray (M, 2): corresponding inlier points in the second image
        '''
        if pts1.shape[0] < 4: # Need at least 4 points to estimate Homography
            print("Warning: Less than 4 points provided for Homography estimation.")
            return pts1, pts2 # Return original points if not enough

        # Ensure points are float32 as expected by OpenCV
        pts1_f32 = np.float32(pts1)
        pts2_f32 = np.float32(pts2)

        H, mask = cv2.findHomography(pts1_f32, pts2_f32, cv2.RANSAC, threshold, confidence=0.99)

        if H is None or mask is None:
            # print("Warning: Homography estimation failed. Returning original points.")
            return pts1, pts2 # Return original points if estimation fails

        mask = mask.ravel().astype(bool) # Convert mask to boolean array
        # Filter matches based on inliers
        return pts1[mask], pts2[mask]

