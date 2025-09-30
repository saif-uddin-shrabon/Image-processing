"""
Advanced Image Filtering Algorithms Module

This module provides comprehensive image filtering techniques including:
- Enhanced versions of basic filters (Gaussian, Bilateral, Non-Local Means)
- Advanced denoising algorithms (Wiener, Adaptive filters)
- Edge-preserving filters
- Frequency domain filters

Based on the original filtering work and extended with state-of-the-art techniques.

Author: Image Filtering Expert
Date: 2024
"""

import numpy as np
import cv2
from typing import Tuple, Union, Optional
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage import restoration, filters
import warnings


class AdvancedImageFilters:
    """
    A comprehensive class for advanced image filtering and denoising.
    """
    
    def __init__(self):
        """Initialize the advanced filters class."""
        pass
    
    # Enhanced Basic Filters (from original README)
    def gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply gamma correction to an image.
        
        Args:
            image (np.ndarray): Input image
            gamma (float): Gamma value for correction
            
        Returns:
            np.ndarray: Gamma corrected image
        """
        # Normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Apply gamma correction
        corrected = np.power(normalized, gamma)
        
        # Convert back to [0, 255]
        result = (corrected * 255).astype(np.uint8)
        return result
    
    def low_pass_filter(self, image: np.ndarray, cutoff_freq: float = 0.1) -> np.ndarray:
        """
        Apply low-pass filter in frequency domain.
        
        Args:
            image (np.ndarray): Input image
            cutoff_freq (float): Cutoff frequency (0-1)
            
        Returns:
            np.ndarray: Filtered image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply FFT
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        
        # Create low-pass filter
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create mask
        mask = np.zeros((rows, cols), np.uint8)
        r = int(cutoff_freq * min(rows, cols) / 2)
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= r ** 2
        mask[mask_area] = 1
        
        # Apply mask and inverse FFT
        f_shift_filtered = f_shift * mask
        f_ishift = ifftshift(f_shift_filtered)
        img_back = ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        return img_back.astype(np.uint8)
    
    def high_pass_filter(self, image: np.ndarray, cutoff_freq: float = 0.1) -> np.ndarray:
        """
        Apply high-pass filter in frequency domain.
        
        Args:
            image (np.ndarray): Input image
            cutoff_freq (float): Cutoff frequency (0-1)
            
        Returns:
            np.ndarray: Filtered image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply FFT
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        
        # Create high-pass filter (inverse of low-pass)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create mask
        mask = np.ones((rows, cols), np.uint8)
        r = int(cutoff_freq * min(rows, cols) / 2)
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= r ** 2
        mask[mask_area] = 0
        
        # Apply mask and inverse FFT
        f_shift_filtered = f_shift * mask
        f_ishift = ifftshift(f_shift_filtered)
        img_back = ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        return img_back.astype(np.uint8)
    
    def enhanced_gaussian_filter(self, image: np.ndarray, sigma: float = 1.0, 
                                kernel_size: Optional[int] = None) -> np.ndarray:
        """
        Enhanced Gaussian filter with automatic kernel size calculation.
        
        Args:
            image (np.ndarray): Input image
            sigma (float): Standard deviation for Gaussian kernel
            kernel_size (int, optional): Kernel size (auto-calculated if None)
            
        Returns:
            np.ndarray: Filtered image
        """
        if kernel_size is None:
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel @ kernel.T
        
        # Apply convolution
        if len(image.shape) == 3:
            filtered = np.zeros_like(image)
            for i in range(image.shape[2]):
                filtered[:, :, i] = cv2.filter2D(image[:, :, i], -1, kernel)
        else:
            filtered = cv2.filter2D(image, -1, kernel)
        
        return filtered
    
    def enhanced_bilateral_filter(self, image: np.ndarray, d: int = 9, 
                                 sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        Enhanced bilateral filter with optimized parameters.
        
        Args:
            image (np.ndarray): Input image
            d (int): Diameter of pixel neighborhood
            sigma_color (float): Filter sigma in the color space
            sigma_space (float): Filter sigma in the coordinate space
            
        Returns:
            np.ndarray: Filtered image
        """
        if len(image.shape) == 3:
            # Apply to each channel
            filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        else:
            # Convert to 3-channel for bilateral filter
            temp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            filtered_temp = cv2.bilateralFilter(temp, d, sigma_color, sigma_space)
            filtered = cv2.cvtColor(filtered_temp, cv2.COLOR_BGR2GRAY)
        
        return filtered
    
    def enhanced_nlm_filter(self, image: np.ndarray, h: float = 10, 
                           template_window_size: int = 7, search_window_size: int = 21) -> np.ndarray:
        """
        Enhanced Non-Local Means denoising filter.
        
        Args:
            image (np.ndarray): Input image
            h (float): Filter strength. Higher h removes more noise but removes details too
            template_window_size (int): Size of template patch
            search_window_size (int): Size of search window
            
        Returns:
            np.ndarray: Denoised image
        """
        if len(image.shape) == 3:
            # Color image
            filtered = cv2.fastNlMeansDenoisingColored(image, None, h, h, 
                                                      template_window_size, search_window_size)
        else:
            # Grayscale image
            filtered = cv2.fastNlMeansDenoising(image, None, h, 
                                              template_window_size, search_window_size)
        
        return filtered
    
    # Advanced Filtering Techniques
    def wiener_filter(self, image: np.ndarray, noise_variance: Optional[float] = None) -> np.ndarray:
        """
        Apply Wiener filter for image denoising.
        
        Args:
            image (np.ndarray): Input noisy image
            noise_variance (float, optional): Noise variance (estimated if None)
            
        Returns:
            np.ndarray: Denoised image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Estimate noise variance if not provided
        if noise_variance is None:
            noise_variance = restoration.estimate_sigma(gray) ** 2
        
        # Apply Wiener filter using scikit-image
        try:
            # Create a simple PSF (point spread function) - identity for denoising
            psf = np.zeros((3, 3))
            psf[1, 1] = 1
            
            # Apply Wiener deconvolution
            denoised = restoration.wiener(gray, psf, noise_variance)
            
            # Normalize to [0, 255]
            denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Wiener filter error: {e}")
            # Fallback to Gaussian filter
            denoised = self.enhanced_gaussian_filter(gray, sigma=1.0)
        
        return denoised
    
    def adaptive_median_filter(self, image: np.ndarray, window_size: int = 3, 
                              max_window_size: int = 7) -> np.ndarray:
        """
        Apply adaptive median filter for salt-and-pepper noise removal.
        
        Args:
            image (np.ndarray): Input image
            window_size (int): Initial window size
            max_window_size (int): Maximum window size
            
        Returns:
            np.ndarray: Filtered image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        rows, cols = gray.shape
        filtered = gray.copy()
        
        for i in range(rows):
            for j in range(cols):
                current_window_size = window_size
                
                while current_window_size <= max_window_size:
                    # Define window boundaries
                    half_window = current_window_size // 2
                    row_min = max(0, i - half_window)
                    row_max = min(rows, i + half_window + 1)
                    col_min = max(0, j - half_window)
                    col_max = min(cols, j + half_window + 1)
                    
                    # Extract window
                    window = gray[row_min:row_max, col_min:col_max]
                    
                    # Calculate statistics
                    z_min = np.min(window)
                    z_max = np.max(window)
                    z_med = np.median(window)
                    z_xy = gray[i, j]
                    
                    # Stage A
                    A1 = z_med - z_min
                    A2 = z_med - z_max
                    
                    if A1 > 0 and A2 < 0:
                        # Stage B
                        B1 = z_xy - z_min
                        B2 = z_xy - z_max
                        
                        if B1 > 0 and B2 < 0:
                            filtered[i, j] = z_xy
                        else:
                            filtered[i, j] = z_med
                        break
                    else:
                        current_window_size += 2
                        
                    if current_window_size > max_window_size:
                        filtered[i, j] = z_med
                        break
        
        return filtered
    
    def anisotropic_diffusion(self, image: np.ndarray, num_iterations: int = 20, 
                             delta_t: float = 0.14, kappa: float = 20) -> np.ndarray:
        """
        Apply Perona-Malik anisotropic diffusion for edge-preserving smoothing.
        
        Args:
            image (np.ndarray): Input image
            num_iterations (int): Number of iterations
            delta_t (float): Time step
            kappa (float): Conduction coefficient
            
        Returns:
            np.ndarray: Filtered image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Convert to float
        img = gray.astype(np.float64)
        
        for _ in range(num_iterations):
            # Calculate gradients
            grad_n = np.roll(img, -1, axis=0) - img  # North
            grad_s = np.roll(img, 1, axis=0) - img   # South
            grad_e = np.roll(img, -1, axis=1) - img  # East
            grad_w = np.roll(img, 1, axis=1) - img   # West
            
            # Calculate conduction coefficients
            c_n = np.exp(-(grad_n / kappa) ** 2)
            c_s = np.exp(-(grad_s / kappa) ** 2)
            c_e = np.exp(-(grad_e / kappa) ** 2)
            c_w = np.exp(-(grad_w / kappa) ** 2)
            
            # Update image
            img += delta_t * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)
        
        # Normalize and convert back to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def guided_filter(self, image: np.ndarray, guide: Optional[np.ndarray] = None, 
                     radius: int = 8, epsilon: float = 0.01) -> np.ndarray:
        """
        Apply guided filter for edge-preserving smoothing.
        
        Args:
            image (np.ndarray): Input image to be filtered
            guide (np.ndarray, optional): Guide image (uses input if None)
            radius (int): Radius of the filter
            epsilon (float): Regularization parameter
            
        Returns:
            np.ndarray: Filtered image
        """
        if guide is None:
            guide = image.copy()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
            p = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        else:
            I = image.astype(np.float64) / 255.0
            p = image.astype(np.float64) / 255.0
        
        # Box filter
        def box_filter(img, r):
            return cv2.boxFilter(img, -1, (2*r+1, 2*r+1))
        
        # Calculate coefficients
        mean_I = box_filter(I, radius)
        mean_p = box_filter(p, radius)
        mean_Ip = box_filter(I * p, radius)
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = box_filter(I * I, radius)
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + epsilon)
        b = mean_p - a * mean_I
        
        mean_a = box_filter(a, radius)
        mean_b = box_filter(b, radius)
        
        q = mean_a * I + mean_b
        
        # Convert back to uint8
        result = np.clip(q * 255, 0, 255).astype(np.uint8)
        return result
    
    def total_variation_denoising(self, image: np.ndarray, weight: float = 0.1, 
                                 max_iterations: int = 200) -> np.ndarray:
        """
        Apply Total Variation denoising.
        
        Args:
            image (np.ndarray): Input noisy image
            weight (float): Denoising weight
            max_iterations (int): Maximum number of iterations
            
        Returns:
            np.ndarray: Denoised image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to [0, 1]
        img_normalized = gray.astype(np.float64) / 255.0
        
        try:
            # Apply TV denoising using scikit-image
            denoised = restoration.denoise_tv_chambolle(img_normalized, weight=weight, 
                                                       max_num_iter=max_iterations)
            
            # Convert back to [0, 255]
            result = (denoised * 255).astype(np.uint8)
            
        except Exception as e:
            print(f"TV denoising error: {e}")
            # Fallback to Gaussian filter
            result = self.enhanced_gaussian_filter(gray, sigma=1.0)
        
        return result
    
    def bm3d_denoising(self, image: np.ndarray, sigma: float = 25) -> np.ndarray:
        """
        BM3D denoising (simplified implementation using available methods).
        
        Args:
            image (np.ndarray): Input noisy image
            sigma (float): Noise standard deviation
            
        Returns:
            np.ndarray: Denoised image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to [0, 1]
        img_normalized = gray.astype(np.float64) / 255.0
        
        try:
            # Use scikit-image's wavelet denoising as approximation
            from skimage.restoration import denoise_wavelet
            
            denoised = denoise_wavelet(img_normalized, method='BayesShrink', 
                                     mode='soft', rescale_sigma=True)
            
            # Convert back to [0, 255]
            result = (denoised * 255).astype(np.uint8)
            
        except ImportError:
            print("Wavelet denoising not available, using NLM instead")
            result = self.enhanced_nlm_filter(gray)
        except Exception as e:
            print(f"BM3D approximation error: {e}")
            result = self.enhanced_nlm_filter(gray)
        
        return result
    
    def apply_all_filters(self, image: np.ndarray, noise_level: float = 25) -> dict:
        """
        Apply all available filters to an image for comparison.
        
        Args:
            image (np.ndarray): Input image
            noise_level (float): Estimated noise level for adaptive methods
            
        Returns:
            dict: Dictionary of filtered images {method_name: filtered_image}
        """
        results = {}
        
        try:
            results['Original'] = image.copy()
            results['Gaussian'] = self.enhanced_gaussian_filter(image, sigma=1.0)
            results['Bilateral'] = self.enhanced_bilateral_filter(image)
            results['Non-Local Means'] = self.enhanced_nlm_filter(image)
            results['Median'] = cv2.medianBlur(image, 5)
            results['Adaptive Median'] = self.adaptive_median_filter(image)
            results['Wiener'] = self.wiener_filter(image)
            results['Anisotropic Diffusion'] = self.anisotropic_diffusion(image)
            results['Guided Filter'] = self.guided_filter(image)
            results['Total Variation'] = self.total_variation_denoising(image)
            results['BM3D Approximation'] = self.bm3d_denoising(image, sigma=noise_level)
            
        except Exception as e:
            print(f"Error applying filters: {e}")
        
        return results
    
    def visualize_filter_comparison(self, original: np.ndarray, filtered_results: dict, 
                                   figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize comparison of different filtering results.
        
        Args:
            original (np.ndarray): Original image
            filtered_results (dict): Dictionary of filtered images
            figsize (tuple): Figure size
        """
        n_filters = len(filtered_results)
        cols = 4
        rows = (n_filters + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
        
        for i, (method_name, filtered_img) in enumerate(filtered_results.items()):
            if i < len(axes):
                if len(filtered_img.shape) == 3:
                    axes[i].imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
                else:
                    axes[i].imshow(filtered_img, cmap='gray')
                axes[i].set_title(method_name)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(filtered_results), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Create test image
    test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    # Initialize filter class
    filters = AdvancedImageFilters()
    
    # Test basic filter
    gaussian_result = filters.enhanced_gaussian_filter(test_image, sigma=1.0)
    
    print("Advanced Image Filters Module loaded successfully!")
    print(f"Test image shape: {test_image.shape}")
    print(f"Gaussian filtered shape: {gaussian_result.shape}")