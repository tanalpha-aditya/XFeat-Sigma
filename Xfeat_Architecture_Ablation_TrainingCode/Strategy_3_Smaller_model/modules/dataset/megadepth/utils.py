# /home/yash/Desktop/GithubDesktop/project-xfeat-sigma/accelerated_features/modules/dataset/megadepth/utils.py

import io
import cv2
import numpy as np
import h5py
import torch
from numpy.linalg import inv
import os # <-- Import os for path checking if needed


try:
    # for internel use only
    from .client import MEGADEPTH_CLIENT, SCANNET_CLIENT
except Exception:
    MEGADEPTH_CLIENT = SCANNET_CLIENT = None

# --- DATA IO ---

def load_array_from_s3(
    path, client, cv_type,
    use_h5py=False,
):
    byte_str = client.Get(path)
    try:
        if not use_h5py:
            raw_array = np.frombuffer(byte_str, np.uint8) # Use frombuffer for bytes
            data = cv2.imdecode(raw_array, cv_type)
        else:
            f = io.BytesIO(byte_str)
            data = np.array(h5py.File(f, 'r')['/depth'])
    except Exception as ex:
        print(f"==> Data loading failure from S3: {path}")
        # Optionally re-raise or return None, but the check below is better
        # raise ex
        data = None # Ensure data is None on failure

    # Removed assert data is not None here, will check later
    return data


def imread_gray(path, augment_fn=None, client=SCANNET_CLIENT):
    # Determine the correct flag based on whether augmentation (which needs color) is applied
    # If augmenting, read color. If not, read grayscale.
    cv_type = cv2.IMREAD_COLOR if augment_fn is not None else cv2.IMREAD_GRAYSCALE

    image = None # Initialize image

    if str(path).startswith('s3://'):
        # S3 loading needs careful checking depending on client implementation
        # Assuming load_array_from_s3 handles potential errors and might return None
        image = load_array_from_s3(str(path), client, cv_type)
    else:
        # Ensure path is a string
        path_str = str(path)
        # Check if file exists before trying to read (optional but helpful)
        # if not os.path.exists(path_str):
        #     print(f"Warning: Image file not found at local path: {path_str}")
        # else:
        # Use the calculated cv_type
        image = cv2.imread(path_str, cv_type)

    # *** Add check immediately after trying to read the image ***
    if image is None:
        # Raise a specific error if reading failed
        raise FileNotFoundError(f"Failed to read image at path: {path}. Check if the file exists, is not corrupted, and has read permissions.")

    # --- Augmentation Logic ---
    if augment_fn is not None:
        # If we read grayscale but need color for augmentation, re-read as color
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
             image_color = cv2.imread(str(path), cv2.IMREAD_COLOR)
             if image_color is None:
                  raise FileNotFoundError(f"Failed to re-read image as color for augmentation at path: {path}.")
             image = image_color # Use the color image for augmentation steps

        # Proceed with augmentation (assuming cv2 reads BGR by default)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Convert back to gray after augmentation

    # If image is still color and wasn't augmented (shouldn't happen with current cv_type logic, but defensive)
    elif image.ndim == 3:
         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    return image  # (h, w)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


# --- MEGADEPTH ---

def fix_path_from_d2net(path):
    if not path:
        return None
    # print(path)
    path = path.replace('Undistorted_SfM/', '')
    path = path.replace('images', 'dense0/imgs')
    path = path.replace('phoenix/S6/zl548/MegaDepth_v1/', '')

    return path

def read_megadepth_gray(path, resize=None, df=None, padding=False, augment_fn=None):
    """
    Args:
        resize (tuple or list): Target size (w, h) or single int for longer edge.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (C, h, w) - Changed to return C channels even if gray
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]
    """
    # read image using the fixed imread_gray
    image = imread_gray(path, augment_fn, client=MEGADEPTH_CLIENT) # image is now guaranteed non-None or raises error

    # image is grayscale (h, w) at this point
    h, w = image.shape

    if resize is not None:
        if len(resize) == 2:
            w_new, h_new = resize
        elif len(resize) == 1: # Assuming single int means longer edge
            res = resize[0]
            w_new, h_new = get_resized_wh(w, h, res)
            w_new, h_new = get_divisible_wh(w_new, h_new, df)
        else:
            raise ValueError("resize must be a tuple/list of length 1 or 2")
    else:
        w_new, h_new = w, h # No resize needed


    if (w, h) != (w_new, h_new):
        image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR) # Use linear for resize
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        # Need image to be 3D for pad_bottom_right if mask is needed? No, handles 2D.
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
        mask = torch.from_numpy(mask)
    else:
        mask = None

    # Convert to tensor, add channel dimension, normalize
    # Original code converted color to (C,H,W), let's keep grayscale as (1,H,W)
    image = torch.from_numpy(image).float().unsqueeze(0) / 255  # (h, w) -> (1, h, w) and normalized

    # The original code had a permute(2,0,1) which suggests it expected color input (H,W,C)
    # Since we ensure grayscale (H,W) from imread_gray, we just unsqueeze.
    # image = torch.from_numpy(image).float().permute(2,0,1) / 255 # This was for (H,W,C) input

    return image, mask, scale


def read_megadepth_depth(path, pad_to=None):

    depth = None
    if str(path).startswith('s3://'):
        try:
            # Assuming load_array_from_s3 can load depth, check its implementation
            depth = load_array_from_s3(path, MEGADEPTH_CLIENT, None, use_h5py=True)
        except Exception as e:
             print(f"Error loading depth from S3 path: {path}")
             print(f"S3 Error: {e}")
             # Decide how to handle - raise error or return empty/None?
             # raise e # Or handle downstream
             return torch.tensor([]) # Return empty tensor on error
    else:
        try:
            with h5py.File(path, 'r') as f:
                depth = np.array(f['depth'])
        except Exception as e:
            print(f"Error loading depth from local path: {path}")
            print(f"File Error: {e}")
            # Decide how to handle
            # raise e
            return torch.tensor([]) # Return empty tensor on error

    if depth is None: # If loading failed silently
         print(f"Warning: Depth loading resulted in None for path: {path}")
         return torch.tensor([])

    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth
