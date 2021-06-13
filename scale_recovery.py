import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def get_ocg_map_2d(pcl, sigma=1, grid_size=1, max_height=None, max_width=None):
    x = pcl[:, 0] / grid_size
    y = pcl[:, 1] / grid_size

    if max_height is None:
        max_height = int(np.max(y))
    if max_width is None:
        max_width = int(np.max(x))

    grid, _, _ = np.histogram2d(y, x, bins=(max_height+1, max_width+1))
    # Apply convolution
    grid = gaussian_filter(grid, sigma, mode='constant', cval=0)
    # Normalize
    grid = grid / np.sum(grid)
    return grid



def compute_entropy_from_ocg_map(ocg_map):
    mask = ocg_map > 0
    # * Entropy
    H = np.sum(-ocg_map[mask] * np.log2(ocg_map[mask]))
    return H


def align_map_size(map1, map2):
    height = max(map1.shape[0], map2.shape[0])
    width = max(map1.shape[1], map2.shape[1])

    if map1.shape[0] < height or map1.shape[1] < width:
        new_map1 = np.zeros((height, width), dtype=map1.dtype)
        new_map1[:map1.shape[0], :map1.shape[1]] = map1
        map1 = new_map1
    

    if map2.shape[0] < height or map2.shape[1] < width:
        new_map2 = np.zeros((height, width), dtype=map2.dtype)
        new_map2[:map2.shape[0], :map2.shape[1]] = map2
        map2 = new_map2
    
    return map1, map2



def estimate_by_searching_in_range(corners, ref_corners, max_scale=3, initial_scale=1, scale_step=0.1, delta=1e-8, plot=False, plot_fn='entropy.png'):

    hist_entropy = []
    hist_scale = []
    best_scale = None
    best_entropy = None

    scale = initial_scale
    while scale < max_scale:
        ref_map = get_ocg_map_2d(ref_corners, sigma=1)
        cur_map = get_ocg_map_2d(corners * scale, sigma=1)

        # Align and padding the size of map
        ref_map, cur_map = align_map_size(ref_map, cur_map)
        joint_map = np.multiply(ref_map, cur_map) + delta
        joint_map = joint_map / np.sum(joint_map)

        h = compute_entropy_from_ocg_map(joint_map)
        hist_entropy.append(h)
        hist_scale.append(scale)

        if best_entropy is None or h < best_entropy:
            best_entropy = h
            best_scale = scale

        # Update scale
        scale += scale_step

    if plot:
        plt.plot(hist_scale, hist_entropy)
        plt.xlabel('scale')
        plt.ylabel('entropy')
        plt.savefig(plot_fn)
    return best_scale
