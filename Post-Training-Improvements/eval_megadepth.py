import cv2
import json
import copy
import tqdm
import torch
import poselib
import numpy as np
import argparse, os
from xfeat_wrapper import XFeatWrapper
from torch.utils.data import Dataset, DataLoader


class MegaDepth1500(Dataset):
    """
        Streamlined MegaDepth-1500 dataloader. The camera poses & metadata are stored in a formatted json for facilitating 
        the download of the dataset and to keep the setup as simple as possible.
    """
    def __init__(self, json_file, root_dir):
        # Load the info & calibration from the JSON
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.root_dir = root_dir

        if not os.path.exists(self.root_dir):
            raise RuntimeError(
            f"Dataset {self.root_dir} does not exist! \n \
              > If you didn't download the dataset, use the downloader tool: python3 -m modules.dataset.download -h")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.data[idx])

        h1, w1 = data['size0_hw']
        h2, w2 = data['size1_hw']

        # Here we resize the images to max_dim = 1200, as described in the paper, and adjust the image such that it is divisible by 32
        # following the protocol of the LoFTR's Dataloader (intrinsics are corrected accordingly). 
        # For adapting this with different resolution, you would need to re-scale intrinsics below.
        print(f"data/Mega1500/megadepth_test_1500/{self.root_dir}{data['pair_names'][0]}")
        image0 = cv2.resize( cv2.imread(f"data/Mega1500/megadepth_test_1500/{self.root_dir}{data['pair_names'][0]}"),
                             (w1, h1))

        image1 = cv2.resize( cv2.imread(f"data/Mega1500/megadepth_test_1500/{self.root_dir}{data['pair_names'][1]}"),
                             (w2, h2))

        data['image0'] = torch.tensor(image0.astype(np.float32)/255).permute(2,0,1)
        data['image1'] = torch.tensor(image1.astype(np.float32)/255).permute(2,0,1)

        for k,v in data.items():
            if k not in ('dataset_name', 'scene_id', 'pair_id', 'pair_names', 'size0_hw', 'size1_hw', 'image0', 'image1'):
                data[k] = torch.tensor(np.array(v, dtype=np.float32))

        return data


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    if n == 0:
        t_err = 0
    else:
        t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
        t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def intrinsics_to_camera(K):
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return {
        "model": "PINHOLE",
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py],
    }


def estimate_pose_poselib(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    M, info = poselib.estimate_relative_pose(
        kpts0, kpts1,
        intrinsics_to_camera(K0),
        intrinsics_to_camera(K1),
        {"max_epipolar_error": thresh,
         "success_prob": conf,
         "min_iterations": 20,
         "max_iterations": 1_000},
    )

    R, t, inl = M.R, M.t, info["inliers"]
    inl = np.array(inl)
    ret = (R, t, inl)

    return ret, (kpts0, kpts1)


def tensor2bgr(t):
    return (t.cpu()[0].permute(1,2,0).numpy()*255).astype(np.uint8)


def compute_pose_error(pair):
    """ 
    Input:
        pair (dict):{
            "pts0": ndrray(N,2)
            "pts1": ndrray(N,2)
            "K0": ndrray(3,3)
            "K1": ndrray(3,3)
            "T_0to1": ndrray(4,4)

        }
    Update:
        pair (dict):{
            "R_err" List[float]: [N]
            "t_err" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = 1.0 if 'ransac_thr' not in pair else pair['ransac_thr']
    conf = 0.99999
    pair.update({'R_err':  np.inf, 't_err': np.inf, 'inliers': []})

    pts0 = pair['pts0']
    pts1 = pair['pts1']
    K0 = pair['K0'].cpu().numpy()[0]
    K1 = pair['K1'].cpu().numpy()[0]
    T_0to1 = pair['T_0to1'].cpu().numpy()[0]

    ret, corrs = estimate_pose_poselib(pts0, pts1, K0, K1, pixel_thr, conf=conf)

    if ret is not None:
        R, t, inliers = ret

        t_err, R_err = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)

        pair['R_err'] = R_err
        pair['t_err'] = t_err


def error_auc(errors, thresholds=[5, 10, 20]):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []

    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def compute_maa(pairs, thresholds=[5, 10, 20]):
    print("auc / mAcc on %d pairs" % (len(pairs)))
    errors = []

    for p in pairs:
        et = p['t_err']
        er = p['R_err']
        errors.append(max(et, er))

    d_err_auc = error_auc(errors)

    for k,v in d_err_auc.items():
        print(k, ': ', '%.1f'%(v*100))

    errors = np.array(errors)

    for t in thresholds:
        acc = (errors <= t).sum() / len(errors)
        print("mAcc@%d: %.1f "%(t, acc*100))


@torch.inference_mode()
def run_pose_benchmark(matcher_fn, loader, top_k=4092, ransac_thr=2.5, trasformations=None, min_cossim=0.5, method="homography", threshold=90):
    """
        Run relative pose estimation benchmark using a specified matcher function and data loader.

        Parameters
        ----------
        matcher_fn : callable
            The matching function to be evaluated for pose estimation. It should accept two np.array RGB images (H,W,3)
            and return mkpts_0, mkpts_1 which are np.array(N,2) matching coordinates.
        
        loader : iterable
            Data loader that provides batches of data. Each batch should contain two images, along 
            with their groundtruth camera poses.
        
        ransac_thr : float, optional, default=2.5
            The RANSAC threshold for considering a point as an inlier in pixels.
    """


    pairs = []
    cnt = 0
    for d in tqdm.tqdm(loader):
        d_error = {}

        if matcher_fn.__name__ == 'match_xfeat_star_original' or matcher_fn.__name__ == 'match_xfeat_original':
            src_pts, dst_pts = matcher_fn(tensor2bgr(d['image0']), tensor2bgr(d['image1']), top_k=top_k) #trasformations=trasformation, top_k=10000, eps=0.1, min_samples=5)
        elif matcher_fn.__name__ == 'match_alike':
            src_pts, dst_pts = matcher_fn(tensor2bgr(d['image0']), tensor2bgr(d['image1'])) 
        elif matcher_fn.__name__ == 'match_xfeat_trasformed':
            src_pts, dst_pts = matcher_fn(tensor2bgr(d['image0']), tensor2bgr(d['image1']), top_k=top_k, trasformations=trasformations, min_cossim=min_cossim, merge=True)
        elif matcher_fn.__name__ == matcher_fn.__name__ == 'match_xfeat_star_trasformed':
            src_pts, dst_pts = matcher_fn(tensor2bgr(d['image0']), tensor2bgr(d['image1']), top_k=top_k, trasformations=trasformations, merge=True)
        elif matcher_fn.__name__ == 'match_xfeat_refined' or matcher_fn.__name__ == 'match_xfeat_star_refined':
            src_pts, dst_pts = matcher_fn(tensor2bgr(d['image0']), tensor2bgr(d['image1']), top_k=top_k, method=method, threshold=90)
        elif matcher_fn.__name__ == 'match_xfeat_star_clustering':
            src_pts, dst_pts = matcher_fn(tensor2bgr(d['image0']), tensor2bgr(d['image1']), top_k=top_k, eps=0.1, min_samples=5)
        else:
            raise ValueError("Invalid matcher")

        
        #delete images to avoid OOM, happens in low mem machines
        del d['image0']
        del d['image1']
        #rescale kpts
        src_pts = src_pts * d['scale0'].numpy()
        dst_pts = dst_pts * d['scale1'].numpy()
        d.update({"pts0":src_pts, "pts1": dst_pts,'ransac_thr': ransac_thr})
        compute_pose_error(d)
        pairs.append(d)
        cnt+=1

    compute_maa(pairs)


def parse_args():
    parser = argparse.ArgumentParser(description="Run pose benchmark with matcher")
    parser.add_argument('--dataset-dir', type=str, required=False,
                        default='data/Mega1500',
                        help="Path to MegaDepth dataset root")
    parser.add_argument('--matcher', type=str, 
                        choices=['xfeat', 'xfeat-star', 'alike', "xfeat-trasformed", "xfeat-star-trasformed", "xfeat-refined", "xfeat-star-refined", "xfeat-star-clustering" ], 
                        default='xfeat',
                        help="Matcher to use")
    parser.add_argument('--ransac-thr', type=float, default=2.5,
                        help="RANSAC threshold value in pixels (default: 2.5)")
    parser.add_argument('--method', type=str, 
                        choices=['homography', 'fundamental' ], 
                        default='homography',
                        help="Method for xfeat-refined and xfeat-star-refined (homography or fundamental)")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    
    dataset = MegaDepth1500( json_file = 'data/megadepth_1500.json',
                             root_dir = './')
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("running benchmarck for XFeat trasformed")

    trasformation= [
        {
            'type': "rotation",
            'angle': 45,
            'pixel': 0
        },
        {
            'type': "rotation",
            'angle': 90,
            'pixel': 0
        }
        # {
        #     'type': "rotation",
        #     'angle': 180,
        #     'pixel': 0
        # }
    ]

    modality = args.matcher
    xfeat = XFeatWrapper()

    if modality == 'xfeat':
        print("Running benchmark for XFeat..")
        run_pose_benchmark(matcher_fn = xfeat.match_xfeat_original, top_k=4092, loader = loader, ransac_thr = args.ransac_thr)
    elif modality == 'xfeat-star':
        print("Running benchmark for XFeat*..")
        run_pose_benchmark(matcher_fn = xfeat.match_xfeat_star_original, top_k=10000, loader = loader, ransac_thr = args.ransac_thr)
    elif modality == 'alike':
        print("Running benchmark for alike..")
        run_pose_benchmark(matcher_fn = alike.match_alike, top_k=None, loader = loader, ransac_thr = args.ransac_thr)
    elif modality == 'xfeat-trasformed':
        print("Running benchmark for XFeat with homography trasformation..")
        run_pose_benchmark(matcher_fn = xfeat.match_xfeat_trasformed, top_k=4092, loader = loader, ransac_thr = args.ransac_thr, trasformations=trasformation, min_cossim=0.5)   
    elif modality == 'xfeat-star-trasformed':
        print("Running benchmark for XFeat* with homography trasformation..")
        run_pose_benchmark(matcher_fn = xfeat.match_xfeat_star_trasformed, top_k=10000, loader = loader, ransac_thr = args.ransac_thr, trasformations=trasformation, min_cossim=0.5)
    elif modality == 'xfeat-refined':
        print("Running benchmark for XFeat refined..")
        run_pose_benchmark(matcher_fn = xfeat.match_xfeat_refined, top_k=4092, loader = loader, ransac_thr = args.ransac_thr, method=args.method)
    elif modality == 'xfeat-star-refined':
        print("Running benchmark for XFeat refined..")
        run_pose_benchmark(matcher_fn = xfeat.match_xfeat_star_refined, top_k=10000, loader = loader, ransac_thr = args.ransac_thr, method=args.method)
    elif modality == 'xfeat-star-clustering':
        print("Running benchmark for XFeat clustering..")
        run_pose_benchmark(matcher_fn = xfeat.match_xfeat_star_clustering, top_k=10000, loader = loader, ransac_thr = args.ransac_thr, method=args.method)
    else:
        print("Invalid matcher")
