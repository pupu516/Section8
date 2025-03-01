#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Part (c): Box downsampling + boundary detection
# ----------------------------------------------------------------------

# You must define or import the following from your Part (a) code:
# pixels_bool (bool array)
# re_grid (float array)
# im_grid (float array)
# re_min, re_max, im_min, im_max (floats)
# re_width, im_width (ints)

def scale_down(arr, block_size):
    """
    Reduce the resolution of 'arr' by factor 'block_size' in both dimensions.
    
    - For a bool array, we do a logical OR over each block (any True => True).
    - For an integer/float array, we do a mean over each block.
    """
    m, n = arr.shape
    mb, nb = m // block_size, n // block_size
    # Reshape into (mb, block_size, nb, block_size)
    if arr.dtype == bool:
        return arr.reshape(mb, block_size, nb, block_size).any(axis=(1, 3))
    else:
        # For numeric types, we just average each block
        return arr.reshape(mb, block_size, nb, block_size).mean(axis=(1, 3))

def pixels_boundary(bool_array):
    """
    Return a boolean mask that is True where 'bool_array' is True
    but has at least one neighbor that is False
    (i.e., boundary pixels in a 4-neighborhood sense).
    """
    up    = np.roll(bool_array,  1, axis=0)
    down  = np.roll(bool_array, -1, axis=0)
    left  = np.roll(bool_array,  1, axis=1)
    right = np.roll(bool_array, -1, axis=1)
    
    # A pixel is boundary if it is True but not all neighbors are True
    return bool_array & ~(up & down & left & right)

def main_part_c(
    pixels_bool, re_grid, im_grid, 
    re_min, re_max, im_min, im_max,
    re_width, im_width,
    downsample_factor=8
):
    """
    Execute part (c) of the assignment:
    
    1. Downsample the boolean mask (and the coordinate grids).
    2. Identify boundary pixels in the downsampled mask.
    3. Estimate area from the number of True pixels in the downsampled mask.
    4. Plot the boundary points for visualization.
    
    :param downsample_factor: how much to reduce resolution (default 8).
    """
    # 1) Downsample
    sd_pixels_bool = scale_down(pixels_bool, downsample_factor)
    sd_re = scale_down(re_grid, downsample_factor)
    sd_im = scale_down(im_grid, downsample_factor)
    
    # 2) Identify boundary pixels
    contour_mask = pixels_boundary(sd_pixels_bool)
    
    # 3) Compute area from the 'inside' pixels in the downsampled mask
    scaled_nx = re_width  // downsample_factor
    scaled_ny = im_width // downsample_factor
    
    # Each downsampled pixel corresponds to this area in the original plane
    pixel_area = ((re_max - re_min)*(im_max - im_min)) / (scaled_nx * scaled_ny)
    region_area = np.sum(sd_pixels_bool) * pixel_area
    
    print(f"Approximate area of the set (from downsampled mask): {region_area}")
    
    # 4) Plot
    plt.figure()
    # Optional: you could also show your original iteration data behind this,
    # e.g. with plt.pcolor(...). For clarity, here we just plot boundary points.

    plt.scatter(sd_re[contour_mask], sd_im[contour_mask], s=5, c='red')
    plt.title(f"Boundary Points (Downsample factor={downsample_factor})")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.savefig('blabla.png')

# If you want to run this script directly, you'll need to define the necessary
# variables below or import them from part (a). For example:
if __name__ == "__main__":
    # Dummy placeholders (won't actually show anything meaningful)
    # You should replace these with real data from part (a).
    re_width, im_width = 800, 800
    re_min, re_max = -1.5, 1.5
    im_min, im_max = -1.0, 1.0
    pixels_bool = np.zeros((im_width, re_width), dtype=bool)
    
    re_vals = np.linspace(re_min, re_max, re_width)
    im_vals = np.linspace(im_min, im_max, im_width)
    re_grid, im_grid = np.meshgrid(re_vals, im_vals)
    
    # Example: set the center region to True
    # (just a fake example so we have something to see)
    center = ( (re_grid**2 + im_grid**2) < 0.4 )
    pixels_bool[center] = True
    
    main_part_c(
        pixels_bool, re_grid, im_grid,
        re_min, re_max, im_min, im_max,
        re_width, im_width,
        downsample_factor=8
    )

