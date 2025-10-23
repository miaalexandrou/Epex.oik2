# nonlinear_filters.py
# -*- coding: utf-8 -*-
"""
Non-Linear Image Filtering Implementation
----------------------------------------
Implementation of various non-linear filters for the Image Filtering GUI.
Non-linear filters don't follow the convolution operation and often use
statistical or morphological operations.
"""
import cv2
import numpy as np


def apply_median_filter(img, ksize):
    """
    Apply Median filter (removes salt-and-pepper noise effectively).
    
    The median filter replaces each pixel value with the median value
    of the pixels in its neighborhood. It's excellent for removing
    impulse noise while preserving edges.
    """
    if len(img.shape) == 3:
        # For RGB images, apply to each channel separately
        result = np.zeros_like(img)
        for c in range(img.shape[2]):
            result[:, :, c] = cv2.medianBlur(img[:, :, c], ksize)
        return result
    else:
        # Grayscale image
        return cv2.medianBlur(img, ksize)


def apply_bilateral_filter(img, d, sigma_color, sigma_space):
    """
    Apply Bilateral filter (edge-preserving smoothing).
    
    The bilateral filter smooths images while preserving edges.
    It considers both spatial distance and intensity difference.
    
    Parameters:
    - d: diameter of each pixel neighborhood
    - sigma_color: filter sigma in the color space (larger = colors farther apart mix more)
    - sigma_space: filter sigma in the coordinate space (larger = pixels farther apart influence more)
    """
    if len(img.shape) == 3:
        # For RGB images, OpenCV expects BGR format for bilateral filter
        # Convert RGB to BGR, apply filter, then convert back
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        filtered_bgr = cv2.bilateralFilter(bgr, d, sigma_color, sigma_space)
        return cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
    else:
        # Grayscale image
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def apply_morphological_filter(img, operation, kernel_size, kernel_shape='ellipse'):
    """
    Apply Morphological operations (erosion, dilation, opening, closing).
    
    Morphological operations are used for:
    - Erosion: shrinks white regions
    - Dilation: expands white regions  
    - Opening: erosion followed by dilation (removes small noise)
    - Closing: dilation followed by erosion (fills small holes)
    """
    # Create morphological kernel
    if kernel_shape == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif kernel_shape == 'rect':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    else:  # cross
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    
    # Apply morphological operation
    if operation == 'erosion':
        morph_op = cv2.MORPH_ERODE
    elif operation == 'dilation':
        morph_op = cv2.MORPH_DILATE
    elif operation == 'opening':
        morph_op = cv2.MORPH_OPEN
    elif operation == 'closing':
        morph_op = cv2.MORPH_CLOSE
    elif operation == 'gradient':
        morph_op = cv2.MORPH_GRADIENT  # Difference between dilation and erosion
    else:
        raise ValueError(f"Unknown morphological operation: {operation}")
    
    if len(img.shape) == 3:
        # For RGB images, apply to each channel separately
        result = np.zeros_like(img)
        for c in range(img.shape[2]):
            result[:, :, c] = cv2.morphologyEx(img[:, :, c], morph_op, kernel)
        return result
    else:
        # Grayscale image
        return cv2.morphologyEx(img, morph_op, kernel)


def apply_nonlinear_filter(img_rgb, filter_name, ksize=5, sigma_color=75, sigma_space=75, 
                          morph_operation='opening', kernel_shape='ellipse', 
                          grayscale_only=False, iterations=1):
    """
    Apply non-linear filter to an RGB image.
    
    Parameters:
    -----------
    img_rgb : np.ndarray
        Input RGB image (H×W×3, uint8)
    filter_name : str
        Name of the filter to apply ('Median', 'Bilateral', 'Morphological')
    ksize : int
        Kernel size (must be odd for median filter)
    sigma_color : float
        Color sigma for bilateral filter
    sigma_space : float
        Space sigma for bilateral filter  
    morph_operation : str
        Morphological operation ('erosion', 'dilation', 'opening', 'closing', 'gradient')
    kernel_shape : str
        Kernel shape for morphological operations ('ellipse', 'rect', 'cross')
    grayscale_only : bool
        If True, convert to grayscale before processing
    iterations : int
        Number of times to apply the filter
    
    Returns:
    --------
    np.ndarray
        Filtered image (H×W×3, uint8)
    """
    # Ensure kernel size is odd for median filter
    if filter_name == "Median" and ksize % 2 == 0:
        ksize += 1
    
    # Convert to working format
    img = img_rgb.copy()
    
    # Convert to grayscale if requested
    if grayscale_only:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = gray
    
    # Apply the selected filter multiple times if requested
    for _ in range(iterations):
        if filter_name == "Median":
            img = apply_median_filter(img, ksize)
        
        elif filter_name == "Bilateral":
            # For bilateral filter, use ksize as diameter
            d = ksize
            img = apply_bilateral_filter(img, d, sigma_color, sigma_space)
        
        elif filter_name == "Morphological":
            img = apply_morphological_filter(img, morph_operation, ksize, kernel_shape)
        
        else:
            raise ValueError(f"Unknown non-linear filter: {filter_name}")
    
    # Convert back to RGB if we processed in grayscale
    if grayscale_only and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Ensure the result is uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img


if __name__ == "__main__":
    # Test the non-linear filters with a simple test image
    print("Testing non-linear filters...")
    
    # Create a simple test image with noise
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Add some salt and pepper noise for median filter testing
    noise_mask = np.random.random((100, 100)) < 0.05
    test_img[noise_mask] = [255, 255, 255]  # Salt noise
    noise_mask = np.random.random((100, 100)) < 0.05  
    test_img[noise_mask] = [0, 0, 0]  # Pepper noise
    
    # Test each filter
    filters_to_test = [
        ("Median", {"ksize": 5}),
        ("Bilateral", {"ksize": 9, "sigma_color": 75, "sigma_space": 75}),
        ("Morphological", {"ksize": 5, "morph_operation": "opening"}),
        ("Morphological", {"ksize": 5, "morph_operation": "closing"}),
        ("Morphological", {"ksize": 3, "morph_operation": "erosion"}),
        ("Morphological", {"ksize": 3, "morph_operation": "dilation"}),
    ]
    
    for filter_name, params in filters_to_test:
        try:
            result = apply_nonlinear_filter(
                test_img, 
                filter_name,
                **params
            )
            print(f"✓ {filter_name} ({params}): OK (shape: {result.shape}, dtype: {result.dtype})")
        except Exception as e:
            print(f"✗ {filter_name} ({params}): Error - {e}")
    
    print("Non-linear filter testing completed!")