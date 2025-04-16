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
            # Placeholder for translation implementation if needed
            tx = type_transformation.get("pixel_x", 0) # Example: Get x translation
            ty = type_transformation.get("pixel_y", 0) # Example: Get y translation
            homography_matrix = np.array([[1, 0, tx],
                                          [0, 1, ty],
                                          [0, 0, 1]])
            print(f"Warning: Translation homography applied ({tx=}, {ty=}). Consider its effect on feature unification logic.")
        else:
             # Allow passing pre-computed homography matrices
            if isinstance(type_transformation, np.ndarray) and type_transformation.shape == (3, 3):
                homography_matrix = type_transformation
            else:
                print(f"Warning: Unknown or invalid transformation type: {type_transformation.get('type', 'N/A')}")
                # Return identity matrix or handle error as appropriate
                homography_matrix = np.identity(3)


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


    def filter_points(self, transformed_points, keypoints, threshold=5, merge=False):
        '''
            Filter or unify the points that are near to each other
            input:
                transformed_points -> np.ndarray (N, 2): points trasformed by the homography matrix
                keypoints -> np.ndarray (N, 2): points original to compare
                threshold -> int: distance threshold
                merge -> bool: if True intersect the points (keep original points close to transformed ones),
                               if False unify the points (keep transformed points far from original ones).
                               NOTE: The original description implies 'merge=True' means intersect
                                     and 'merge=False' means unify (keep points *not* close).
        '''

        if len(keypoints) == 0 or len(transformed_points) == 0:
             return [] # Return empty list if either set is empty

        if len(keypoints.shape) == 3:
            keypoints = keypoints.squeeze(0)

        # Ensure keypoints is 2D
        if len(keypoints.shape) != 2 or keypoints.shape[1] != 2:
             raise ValueError(f"Invalid shape for keypoints: {keypoints.shape}. Expected (N, 2).")


        tree = cKDTree(keypoints)

        idx_ret = []
        # Ensure transformed_points has the expected shape (N, 3) before slicing
        if transformed_points.shape[1] < 2:
             raise ValueError(f"Transformed points have unexpected shape: {transformed_points.shape}. Expected at least 2 columns.")

        points_to_query = transformed_points[:, :2]

        for idx, coord in enumerate(points_to_query):
            distances, indices = tree.query(coord, k=1, distance_upper_bound=threshold)

            # tree.query returns inf if no neighbor is within distance_upper_bound
            is_close = distances < threshold # True if a neighbor is within threshold

            if merge: # Keep points from the *first* set (transformed) that ARE close to points in the *second* set (keypoints)
                if is_close:
                    idx_ret.append(idx)
            else: # Keep points from the *first* set (transformed) that ARE NOT close to points in the *second* set (keypoints)
                if not is_close:
                    idx_ret.append(idx)

        return idx_ret


    def unify_features(self, features1, features2, homography, merge=False):
        '''
            Unify or intersect the features of two sets based on geometric proximity after transformation.
            input:
                features1 -> Dict:{keypoints, scores, descriptors} - Base set of features.
                features2 -> Dict:{keypoints, scores, descriptors} - Features from the transformed image.
                homography -> np.ndarray (3,3): Homography matrix that transforms points *from* image1's space *to* image2's space (H_21).
                                               Or, if transforming points *from* image2 to image1, it's H_12 = inv(H_21).
                                               The current implementation assumes homography transforms points from features1 space TO features2 space.
                merge -> bool: if True, intersect (keep points in features1 that correspond closely to points in features2 after transformation).
                               if False, unify (keep points from features1, plus points from features2 that DON'T correspond closely to points in features1 after inverse transformation).
            return:
                Dict:{keypoints, scores, descriptors} - The resulting merged or unified feature set.
        '''

        # Ensure features are not empty
        if features1['keypoints'].numel() == 0:
            return features2 if not merge else features1 # Return the other set if unifying, empty set if intersecting
        if features2['keypoints'].numel() == 0:
            return features1 # Return the base set

        keypoints1_np = copy.deepcopy(features1["keypoints"].cpu().numpy())
        keypoints2_np = copy.deepcopy(features2["keypoints"].cpu().numpy())

        if keypoints1_np.ndim == 1: keypoints1_np = keypoints1_np.reshape(-1, 2) # Handle single point case
        if keypoints2_np.ndim == 1: keypoints2_np = keypoints2_np.reshape(-1, 2)


        if merge: # Intersect: Keep points from features1 that are close to features2 points after H maps points from 1 -> 2
            if keypoints1_np.shape[0] == 0: return {"keypoints": torch.empty((0, 2)), "scores": torch.empty(0), "descriptors": torch.empty((0, features1["descriptors"].shape[1]))}

            homogeneous_points1 = np.hstack([keypoints1_np, np.ones((keypoints1_np.shape[0], 1))])
            # Transform points from features1 space to features2 space
            transformed_points1_h = (homography @ homogeneous_points1.T).T
            # Normalize homogeneous coordinates
            # Add epsilon to prevent division by zero
            transformed_points1 = transformed_points1_h[:, :2] / (transformed_points1_h[:, 2][:, np.newaxis] + 1e-8)

            # Find indices in features1 whose transformed points are close to keypoints in features2
            idx_selected = self.filter_points(transformed_points1, keypoints2_np, threshold=5, merge=True) # merge=True to find close points

            if not idx_selected: # Handle case where no points match
                 return {"keypoints": torch.empty((0, 2), device=features1['keypoints'].device),
                        "scores": torch.empty(0, device=features1['scores'].device),
                        "descriptors": torch.empty((0, features1['descriptors'].shape[1]), device=features1['descriptors'].device)}


            keypoints_selected = features1["keypoints"][idx_selected]
            scores_selected = features1["scores"][idx_selected]
            descriptors_selected = features1["descriptors"][idx_selected]

            # Result contains only the subset of features1 that matched
            return {"keypoints": keypoints_selected,
                    "scores": scores_selected,
                    "descriptors": descriptors_selected}

        else: # Unify: Keep all of features1, plus points from features2 that are far from features1 points after inverse H maps points from 2 -> 1
            if keypoints2_np.shape[0] == 0: return features1 # Nothing to add from features2

            try:
                inv_homography = np.linalg.inv(homography)
            except np.linalg.LinAlgError:
                print("Warning: Homography matrix is singular, cannot invert for unification. Returning original features1.")
                return features1

            homogeneous_points2 = np.hstack([keypoints2_np, np.ones((keypoints2_np.shape[0], 1))])
            # Transform points from features2 space back to features1 space
            transformed_points2_h = (inv_homography @ homogeneous_points2.T).T
             # Normalize homogeneous coordinates
            transformed_points2 = transformed_points2_h[:, :2] / (transformed_points2_h[:, 2][:, np.newaxis] + 1e-8)

            # Find indices in features2 whose transformed points are *far* from keypoints in features1
            idx_to_add = self.filter_points(transformed_points2, keypoints1_np, threshold=5, merge=False) # merge=False to find points that are far

            # Combine features1 with the selected subset of features2
            if not idx_to_add: # If no points from features2 are added
                return features1

            # Concatenate tensors
            final_keypoints = torch.cat((features1["keypoints"], features2["keypoints"][idx_to_add]), dim=0)
            final_scores = torch.cat((features1["scores"], features2["scores"][idx_to_add]), dim=0)
            final_descriptors = torch.cat((features1["descriptors"], features2["descriptors"][idx_to_add]), dim=0)


            return {"keypoints": final_keypoints,
                    "scores": final_scores,
                    "descriptors": final_descriptors}


    def trasformed_detection_features(self, image, trasformations, merge=False, top_k = None):
        '''
            Take an image and apply the trasformations given in input and detect the features unifying or intersecting them
            input:
                image -> np.ndarray (H,W,C): grayscale or rgb image
                trasformations -> List[Dict]: list of transformation parameters or precomputed homographies
                merge -> bool: Passed to unify_features. True=intersect, False=unify.
            return:
                Dict:{keypoints, scores, descriptors}

        '''
        if top_k is None: top_k = self.top_k

        features_original = self.detect_feature_sparse(image, top_k=top_k)
        if features_original['keypoints'].numel() == 0:
             print("Warning: No features detected in the original image.")
             # Depending on the logic, you might want to return early or handle this case
             # return features_original # Or process transformations anyway?

        features_accumulated = copy.deepcopy(features_original) # Start with original features


        for transformation_spec in trasformations:
            try:
                # Get homography (either computes from spec or uses provided matrix)
                homography = self.get_homography(transformation_spec, image)
                if homography is None:
                    print(f"Warning: Could not compute homography for transformation: {transformation_spec}. Skipping this transformation.")
                    continue

                image_transformed = self.get_image_trasformed(image, homography)

                # Detect features on the *transformed* image
                features_transformed = self.detect_feature_sparse(image_transformed, top_k=top_k)
                if features_transformed['keypoints'].numel() == 0:
                    # print(f"Warning: No features detected in transformed image for {transformation_spec}. Skipping unification for this transform.")
                    continue # Skip if no features found in transformed image

                # Unify/Intersect the *accumulated* features with the *newly detected transformed* features
                # Homography H maps from original image space to current transformed image space
                features_accumulated = self.unify_features(features_accumulated, features_transformed, homography, merge=merge)

            except Exception as e:
                 print(f"Error processing transformation {transformation_spec}: {e}")
                 continue # Skip this transformation on error


        return features_accumulated


    def match_xfeat_trasformed(self, image1, image2, trasformations = {}, top_k=4092, min_cossim = None, merge=True):
        '''
            Inference of the xfeat algorithm with our version of the trasformation and the match
            input:
                image1 -> np.ndarray (H,W,C): grayscale or rgb image
                image2 -> np.ndarray (H,W,C): grayscale or rgb image
                trasformations -> List[Dict]: Transformations applied *identically* to both images before matching.
                min_cossim -> float: minimum cosine similarity to consider a match
                merge -> bool: How to combine features from transformed views (True=intersect, False=unify)
            return:
                points1 -> np.ndarray (N, 2): points of the first image
                points2 -> np.ndarray (N, 2): points of the second image
        '''
        if min_cossim is None: min_cossim = self.min_cossim
        if top_k is None: top_k = self.top_k

        # Apply the *same* set of transformations to detect features in both images
        features_image1 = self.trasformed_detection_features(image1, trasformations, merge=merge, top_k=top_k)
        features_image2 = self.trasformed_detection_features(image2, trasformations, merge=merge, top_k=top_k)

        # Check if features were detected
        if features_image1['keypoints'].numel() == 0 or features_image2['keypoints'].numel() == 0:
            print("Warning: No features found in one or both images after transformation. Returning empty matches.")
            return np.empty((0, 2)), np.empty((0, 2))

        kpts1, descs1 = features_image1['keypoints'], features_image1['descriptors']
        kpts2, descs2 = features_image2['keypoints'], features_image2['descriptors']

        # Use the base XFeat matcher on the (potentially augmented) feature sets
        idx0, idx1 = self.xfeat_instance.match(descs1, descs2, min_cossim=min_cossim)

        points1 = kpts1[idx0].cpu().numpy()
        points2 = kpts2[idx1].cpu().numpy()

        return points1, points2


    def trasformed_detection_features_dense(self, imset, trasformations, merge=True, top_k = None, multiscale = True):
        '''
            Dense feature detection with transformations (similar to sparse version).
            input:
                imset -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C)
                trasformations -> List[Dict]: Transformations applied to the image(s).
                merge -> bool: Passed to unify_features. True=intersect, False=unify.
            return:
                Dict:{keypoints, scores, descriptors}
        '''
        if top_k is None: top_k = self.top_k

        # Detect initial dense features
        features_original = self.detect_feature_dense(imset, top_k, multiscale)
        if features_original['keypoints'].numel() == 0:
            print("Warning: No dense features detected in the original image set.")
            # Handle as appropriate, e.g., return empty features
            # return features_original

        features_accumulated = copy.deepcopy(features_original)

        # Need to handle image format correctly (numpy for transformations)
        if isinstance(imset, torch.Tensor):
             # Assuming batch size B=1 for transformation processing here
             if imset.shape[0] != 1:
                 print("Warning: Dense transformed detection currently assumes B=1. Using first image.")
             image_np = imset[0].permute(1, 2, 0).cpu().numpy() # Convert CHW to HWC numpy
             # If image was grayscale, might need adjustment
             if image_np.shape[2] == 1:
                 image_np = image_np.squeeze(2)
             # Ensure it's uint8 if needed by cv2 functions
             if image_np.max() <= 1.0: # If tensor was normalized (0-1)
                 image_np = (image_np * 255).astype(np.uint8)

        elif isinstance(imset, np.ndarray):
             image_np = imset # Assume it's already HWC or HW
             if len(imset.shape) == 4 and imset.shape[0] == 1: # Handle (1, H, W, C) numpy
                 image_np = imset.squeeze(0)
        else:
             raise TypeError("Unsupported input type for dense transformed detection.")


        for transformation_spec in trasformations:
            try:
                homography = self.get_homography(transformation_spec, image_np)
                if homography is None:
                    print(f"Warning: Could not compute homography for dense transformation: {transformation_spec}. Skipping.")
                    continue

                # Apply transformation to the numpy image
                image_transformed_np = self.get_image_trasformed(image_np, homography)

                # Detect dense features on the *transformed* numpy image
                # Note: detect_feature_dense expects tensor, so convert back
                features_transformed = self.detect_feature_dense(image_transformed_np, top_k, multiscale)

                if features_transformed['keypoints'].numel() == 0:
                     # print(f"Warning: No dense features detected in transformed image for {transformation_spec}. Skipping unification.")
                    continue

                # Unify accumulated features with the new transformed features
                # Homography maps original space -> transformed space
                features_accumulated = self.unify_features(features_accumulated, features_transformed, homography, merge=merge)

            except Exception as e:
                print(f"Error processing dense transformation {transformation_spec}: {e}")
                continue

        return features_accumulated


    def match_xfeat_star_trasformed(self, imset1, imset2, trasformations, top_k = None, merge=True):
        '''
            Dense matching (XFeat*) using features augmented by transformations.
            input:
                imset1, imset2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C)
                trasformations -> List[Dict]: Transformations applied identically to both images.
                top_k -> int: Max features per image/scale.
                merge -> bool: How to combine features from transformed views (True=intersect, False=unify)
            returns:
                mkpts_0, mkpts_1 -> np.ndarray (N,2) matches if B=1, else List[torch.Tensor(N, 4)]
        '''

        if top_k is None: top_k = self.top_k
        imset1 = self.parse_input(imset1) # Ensures tensor format (B, C, H, W)
        imset2 = self.parse_input(imset2)

        # Detect augmented dense features for both image sets
        # Note: Current implementation assumes B=1 for transformation application inside the function
        feature_images1 = self.trasformed_detection_features_dense(imset1, trasformations, merge=merge, top_k=top_k, multiscale=True)
        feature_images2 = self.trasformed_detection_features_dense(imset2, trasformations, merge=merge, top_k=top_k, multiscale=True)


        # Check for empty features before proceeding
        if feature_images1['keypoints'].numel() == 0 or feature_images2['keypoints'].numel() == 0:
             print("Warning: No dense features found in one or both images after transformation for matching. Returning empty matches.")
             # Adjust return type based on expected batch size B
             B = imset1.shape[0]
             return (np.empty((0, 2)), np.empty((0, 2))) if B == 1 else []


        # Prepare features for batch matching (add batch dimension back)
        feat1 = {}
        feat2 = {}
        for key in feature_images1: # Assumes keys are consistent ('keypoints', 'scores', 'descriptors')
            # Handle 'scores' key which might be named 'scales' internally
            if key == "scores":
                 feat1["scales"] = feature_images1[key].unsqueeze(0).to(self.device)
                 feat2["scales"] = feature_images2[key].unsqueeze(0).to(self.device)
                 # Also add standard 'scores' if needed by refine_matches downstream
                 feat1["scores"] = feature_images1[key].unsqueeze(0).to(self.device)
                 feat2["scores"] = feature_images2[key].unsqueeze(0).to(self.device)

            elif key in ["keypoints", "descriptors"]:
                 feat1[key] = feature_images1[key].unsqueeze(0).to(self.device)
                 feat2[key] = feature_images2[key].unsqueeze(0).to(self.device)
            # else: Don't include unexpected keys


        # --- Original XFeat Star Matching/Refinement Logic ---
        # Match batches of pairs (coarse matching)
        # Ensure descriptors are on the correct device
        idxs_list = self.xfeat_instance.batch_match(feat1['descriptors'], feat2['descriptors'])
        B = len(imset1) # Get original batch size

        # Refine coarse matches
        matches = []
        for b in range(B): # Iterate through batch
             # Ensure all required keys ('keypoints', 'scales', 'descriptors') are in feat1/feat2 for refine_matches
             # Check if idxs_list[b] is valid before refining
             if idxs_list and b < len(idxs_list) and idxs_list[b] is not None and idxs_list[b].numel() > 0:
                try:
                     refined = self.xfeat_instance.refine_matches(feat1, feat2, matches=idxs_list, batch_idx=b)
                     matches.append(refined)
                except Exception as e:
                     print(f"Error during refine_matches for batch {b}: {e}")
                     # Append empty tensor or handle as appropriate
                     matches.append(torch.empty((0, 4), device=self.device)) # Example: append empty tensor

             else:
                 # Handle case where coarse matching failed for this batch item
                 # print(f"No coarse matches to refine for batch index {b}.")
                 matches.append(torch.empty((0, 4), device=self.device)) # Append empty tensor

        # Return based on original batch size
        if B > 1:
            return matches # Return list of tensors for batch > 1
        elif B == 1 and matches: # B=1 and matches list is not empty
             # Convert the single tensor in the list to numpy arrays (mkpts0, mkpts1)
             match_tensor = matches[0].cpu().detach().numpy()
             return match_tensor[:, :2], match_tensor[:, 2:]
        else: # B=0 or matches list is empty
             return np.empty((0, 2)), np.empty((0, 2))


############################################################################################################
