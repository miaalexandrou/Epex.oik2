# linear_filters.py
# -*- coding: utf-8 -*-
"""
Linear Image Filtering Implementation
------------------------------------
Implementation of various linear filters for the Image Filtering GUI.
Students implement the actual filtering logic here.
"""
import cv2
import numpy as np


def get_border_type(border_mode_str):
    """Convert border mode string to OpenCV constant."""
    border_map = {
        'reflect': cv2.BORDER_REFLECT,
        'replicate': cv2.BORDER_REPLICATE,
        'constant': cv2.BORDER_CONSTANT
    }
    return border_map.get(border_mode_str, cv2.BORDER_REFLECT)


def apply_box_filter(img, ksize, border_type):
    """Apply Box/Average filter."""
    return cv2.blur(img, (ksize, ksize), borderType=border_type)


def apply_gaussian_filter(img, ksize, sigma, border_type):
    """Apply Gaussian filter."""
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, borderType=border_type)


def apply_sobel_filter(img, direction, ksize, border_type):
    """Apply Sobel filter in X or Y direction."""
    if len(img.shape) == 3:
        # For RGB images, apply to each channel separately
        result = np.zeros_like(img, dtype=np.float64)
        for c in range(img.shape[2]):
            if direction == 'x':
                sobel = cv2.Sobel(img[:, :, c], cv2.CV_64F, 1, 0, ksize=ksize, borderType=border_type)
            else:  # direction == 'y'
                sobel = cv2.Sobel(img[:, :, c], cv2.CV_64F, 0, 1, ksize=ksize, borderType=border_type)
            result[:, :, c] = sobel
    else:
        # Grayscale image
        if direction == 'x':
            result = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize, borderType=border_type)
        else:  # direction == 'y'
            result = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize, borderType=border_type)
    
    # Convert to absolute values and scale to 0-255
    result = cv2.convertScaleAbs(result)
    return result


def apply_laplacian_filter(img, ksize, border_type):
    """Apply Laplacian filter."""
    if len(img.shape) == 3:
        # For RGB images, apply to each channel separately
        result = np.zeros_like(img, dtype=np.float64)
        for c in range(img.shape[2]):
            laplacian = cv2.Laplacian(img[:, :, c], cv2.CV_64F, ksize=ksize, borderType=border_type)
            result[:, :, c] = laplacian
    else:
        # Grayscale image
        result = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize, borderType=border_type)
    
    # Convert to absolute values and scale to 0-255
    result = cv2.convertScaleAbs(result)
    return result


def apply_unsharp_filter(img, ksize, sigma, alpha, border_type):
    """Apply Unsharp Masking filter."""
    # Create blurred version
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, borderType=border_type)
    
    # Apply unsharp masking: sharp = original + alpha * (original - blurred)
    img_float = img.astype(np.float64)
    blurred_float = blurred.astype(np.float64)
    
    # Calculate the unsharp mask
    mask = img_float - blurred_float
    sharpened = img_float + alpha * mask
    
    # Clip values to valid range [0, 255]
    sharpened = np.clip(sharpened, 0, 255)
    
    return sharpened.astype(np.uint8)


def apply_linear_filter(img_rgb, filter_name, ksize, sigma=1.0, border_mode_str='reflect', 
                       grayscale_only=False, iterations=1, unsharp_alpha=1.0):
    """
    Apply linear filter to an RGB image.
    
    Parameters:
    -----------
    img_rgb : np.ndarray
        Input RGB image (H×W×3, uint8)
    filter_name : str
        Name of the filter to apply
    ksize : int
        Kernel size (must be odd)
    sigma : float
        Standard deviation for Gaussian filter
    border_mode_str : str
        Border handling mode ('reflect', 'replicate', 'constant')
    grayscale_only : bool
        If True, convert to grayscale before processing
    iterations : int
        Number of times to apply the filter
    unsharp_alpha : float
        Alpha parameter for unsharp masking
    
    Returns:
    --------
    np.ndarray
        Filtered image (H×W×3, uint8)
    """
    # Ensure kernel size is odd
    if ksize % 2 == 0:
        ksize += 1
    
    # Convert border mode string to OpenCV constant
    border_type = get_border_type(border_mode_str)
    
    # Convert to working format
    img = img_rgb.copy()
    
    # Convert to grayscale if requested
    if grayscale_only:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = gray
    
    # Apply the selected filter multiple times if requested
    for _ in range(iterations):
        if filter_name == "Box/Average":
            img = apply_box_filter(img, ksize, border_type)
        
        elif filter_name == "Gaussian":
            img = apply_gaussian_filter(img, ksize, sigma, border_type)
        
        elif filter_name == "Sobel X":
            img = apply_sobel_filter(img, 'x', ksize, border_type)
        
        elif filter_name == "Sobel Y":
            img = apply_sobel_filter(img, 'y', ksize, border_type)
        
        elif filter_name == "Laplacian":
            img = apply_laplacian_filter(img, ksize, border_type)
        
        elif filter_name == "Unsharp (α)":
            img = apply_unsharp_filter(img, ksize, sigma, unsharp_alpha, border_type)
        
        else:
            raise ValueError(f"Unknown filter: {filter_name}")
    
    # Convert back to RGB if we processed in grayscale
    if grayscale_only and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Ensure the result is uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img


if __name__ == "__main__":
    # Test the filters with a simple test image
    print("Testing linear filters...")
    
    # Create a simple test image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test each filter
    filters_to_test = [
        ("Box/Average", {}),
        ("Gaussian", {"sigma": 2.0}),
        ("Sobel X", {}),
        ("Sobel Y", {}),
        ("Laplacian", {}),
        ("Unsharp (α)", {"unsharp_alpha": 1.5})
    ]
    
    for filter_name, params in filters_to_test:
        try:
            result = apply_linear_filter(
                test_img, 
                filter_name, 
                ksize=5,
                **params
            )
            print(f"✓ {filter_name}: OK (shape: {result.shape}, dtype: {result.dtype})")
        except Exception as e:
            print(f"✗ {filter_name}: Error - {e}")
    
    print("Testing completed!")