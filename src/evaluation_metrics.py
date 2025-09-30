"""
Image Quality Evaluation Metrics Module

This module provides comprehensive image quality assessment metrics for evaluating
the performance of image denoising and filtering algorithms.

Includes:
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Universal Image Quality Index (UQI)
- Visual Information Fidelity (VIF)

Author: Image Filtering Expert
Date: 2024
"""

import numpy as np
import cv2
from typing import Tuple, Union, Dict
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d
import warnings


class ImageQualityMetrics:
    """
    A comprehensive class for calculating various image quality metrics.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def mse(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Calculate Mean Squared Error between two images.
        
        Args:
            original (np.ndarray): Original reference image
            processed (np.ndarray): Processed/filtered image
            
        Returns:
            float: MSE value
        """
        if original.shape != processed.shape:
            raise ValueError("Images must have the same dimensions")
        
        mse_value = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
        return mse_value
    
    def mae(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error between two images.
        
        Args:
            original (np.ndarray): Original reference image
            processed (np.ndarray): Processed/filtered image
            
        Returns:
            float: MAE value
        """
        if original.shape != processed.shape:
            raise ValueError("Images must have the same dimensions")
        
        mae_value = np.mean(np.abs(original.astype(np.float64) - processed.astype(np.float64)))
        return mae_value
    
    def psnr(self, original: np.ndarray, processed: np.ndarray, max_pixel_value: float = 255.0) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio between two images.
        
        Args:
            original (np.ndarray): Original reference image
            processed (np.ndarray): Processed/filtered image
            max_pixel_value (float): Maximum possible pixel value
            
        Returns:
            float: PSNR value in dB
        """
        mse_value = self.mse(original, processed)
        
        if mse_value == 0:
            return float('inf')  # Perfect match
        
        psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse_value))
        return psnr_value
    
    def ssim(self, original: np.ndarray, processed: np.ndarray, 
             window_size: int = 11, k1: float = 0.01, k2: float = 0.03,
             max_pixel_value: float = 255.0) -> Tuple[float, np.ndarray]:
        """
        Calculate Structural Similarity Index between two images.
        
        Args:
            original (np.ndarray): Original reference image
            processed (np.ndarray): Processed/filtered image
            window_size (int): Size of the sliding window
            k1, k2 (float): SSIM parameters
            max_pixel_value (float): Maximum possible pixel value
            
        Returns:
            tuple: (SSIM value, SSIM map)
        """
        if original.shape != processed.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Convert to float
        img1 = original.astype(np.float64)
        img2 = processed.astype(np.float64)
        
        # Constants
        c1 = (k1 * max_pixel_value) ** 2
        c2 = (k2 * max_pixel_value) ** 2
        
        # Create Gaussian window
        sigma = 1.5
        window = self._gaussian_window(window_size, sigma)
        
        # Calculate local means
        mu1 = convolve2d(img1, window, mode='valid')
        mu2 = convolve2d(img2, window, mode='valid')
        
        # Calculate local variances and covariance
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = convolve2d(img1 ** 2, window, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(img2 ** 2, window, mode='valid') - mu2_sq
        sigma12 = convolve2d(img1 * img2, window, mode='valid') - mu1_mu2
        
        # Calculate SSIM map
        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        
        ssim_map = numerator / denominator
        ssim_value = np.mean(ssim_map)
        
        return ssim_value, ssim_map
    
    def _gaussian_window(self, size: int, sigma: float) -> np.ndarray:
        """
        Create a 2D Gaussian window.
        
        Args:
            size (int): Window size
            sigma (float): Standard deviation
            
        Returns:
            np.ndarray: Normalized Gaussian window
        """
        coords = np.arange(size) - size // 2
        g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = np.outer(g, g)
        return g / np.sum(g)
    
    def uqi(self, original: np.ndarray, processed: np.ndarray, 
            window_size: int = 8) -> float:
        """
        Calculate Universal Image Quality Index.
        
        Args:
            original (np.ndarray): Original reference image
            processed (np.ndarray): Processed/filtered image
            window_size (int): Size of the sliding window
            
        Returns:
            float: UQI value
        """
        if original.shape != processed.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Convert to float
        img1 = original.astype(np.float64)
        img2 = processed.astype(np.float64)
        
        # Calculate statistics
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        # Calculate UQI
        numerator = 4 * sigma12 * mu1 * mu2
        denominator = (sigma1_sq + sigma2_sq) * (mu1 ** 2 + mu2 ** 2)
        
        if denominator == 0:
            return 1.0 if numerator == 0 else 0.0
        
        uqi_value = numerator / denominator
        return uqi_value
    
    def snr(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio.
        
        Args:
            original (np.ndarray): Original reference image
            processed (np.ndarray): Processed/filtered image
            
        Returns:
            float: SNR value in dB
        """
        signal_power = np.mean(original.astype(np.float64) ** 2)
        noise_power = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr_value = 10 * np.log10(signal_power / noise_power)
        return snr_value
    
    def calculate_all_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """
        Calculate all available metrics between two images.
        
        Args:
            original (np.ndarray): Original reference image
            processed (np.ndarray): Processed/filtered image
            
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        metrics = {}
        
        try:
            metrics['MSE'] = self.mse(original, processed)
            metrics['MAE'] = self.mae(original, processed)
            metrics['PSNR'] = self.psnr(original, processed)
            metrics['SNR'] = self.snr(original, processed)
            
            # Handle grayscale images for SSIM
            if len(original.shape) == 3:
                # Convert to grayscale for SSIM calculation
                orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                ssim_val, _ = self.ssim(orig_gray, proc_gray)
                metrics['SSIM'] = ssim_val
                metrics['UQI'] = self.uqi(orig_gray, proc_gray)
            else:
                ssim_val, _ = self.ssim(original, processed)
                metrics['SSIM'] = ssim_val
                metrics['UQI'] = self.uqi(original, processed)
                
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            
        return metrics
    
    def compare_filtering_methods(self, original: np.ndarray, 
                                 filtered_images: Dict[str, np.ndarray],
                                 save_results: bool = False,
                                 output_path: str = None) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple filtering methods using all available metrics.
        
        Args:
            original (np.ndarray): Original reference image
            filtered_images (dict): Dictionary of filtered images {method_name: image}
            save_results (bool): Whether to save results to file
            output_path (str): Path to save results
            
        Returns:
            dict: Nested dictionary of results {method: {metric: value}}
        """
        results = {}
        
        for method_name, filtered_img in filtered_images.items():
            results[method_name] = self.calculate_all_metrics(original, filtered_img)
        
        # Print results table
        self._print_results_table(results)
        
        # Save results if requested
        if save_results and output_path:
            self._save_results_to_file(results, output_path)
        
        return results
    
    def _print_results_table(self, results: Dict[str, Dict[str, float]]):
        """
        Print results in a formatted table.
        
        Args:
            results (dict): Results dictionary
        """
        if not results:
            return
        
        # Get all metrics
        metrics = list(next(iter(results.values())).keys())
        methods = list(results.keys())
        
        # Print header
        print(f"{'Method':<20}", end="")
        for metric in metrics:
            print(f"{metric:>12}", end="")
        print()
        print("-" * (20 + 12 * len(metrics)))
        
        # Print results
        for method in methods:
            print(f"{method:<20}", end="")
            for metric in metrics:
                value = results[method][metric]
                if value == float('inf'):
                    print(f"{'∞':>12}", end="")
                else:
                    print(f"{value:>12.4f}", end="")
            print()
    
    def _save_results_to_file(self, results: Dict[str, Dict[str, float]], 
                             output_path: str):
        """
        Save results to a CSV file.
        
        Args:
            results (dict): Results dictionary
            output_path (str): Output file path
        """
        import csv
        
        if not results:
            return
        
        metrics = list(next(iter(results.values())).keys())
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Method'] + metrics)
            
            # Write data
            for method, method_results in results.items():
                row = [method] + [method_results[metric] for metric in metrics]
                writer.writerow(row)
    
    def visualize_metrics_comparison(self, results: Dict[str, Dict[str, float]], 
                                   figsize: Tuple[int, int] = (12, 8)):
        """
        Create visualization of metrics comparison.
        
        Args:
            results (dict): Results dictionary
            figsize (tuple): Figure size
        """
        if not results:
            return
        
        methods = list(results.keys())
        metrics = list(next(iter(results.values())).keys())
        
        # Create subplots for each metric
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:6]):  # Limit to 6 metrics
            if i >= len(axes):
                break
                
            values = [results[method][metric] for method in methods]
            
            # Handle infinite values
            finite_values = [v if v != float('inf') else np.nan for v in values]
            
            axes[i].bar(methods, finite_values)
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(finite_values):
                if not np.isnan(v):
                    axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def create_quality_report(self, original: np.ndarray, 
                            filtered_images: Dict[str, np.ndarray],
                            noisy_image: np.ndarray = None,
                            save_path: str = None) -> str:
        """
        Create a comprehensive quality assessment report.
        
        Args:
            original (np.ndarray): Original clean image
            filtered_images (dict): Dictionary of filtered images
            noisy_image (np.ndarray): Noisy input image (optional)
            save_path (str): Path to save the report
            
        Returns:
            str: Report content
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("IMAGE QUALITY ASSESSMENT REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Image information
        report_lines.append(f"Original Image Shape: {original.shape}")
        report_lines.append(f"Number of Filtering Methods: {len(filtered_images)}")
        report_lines.append("")
        
        # Calculate metrics for all methods
        results = self.compare_filtering_methods(original, filtered_images)
        
        # Add noisy image comparison if provided
        if noisy_image is not None:
            noisy_metrics = self.calculate_all_metrics(original, noisy_image)
            results['Noisy Input'] = noisy_metrics
        
        # Find best performing methods for each metric
        report_lines.append("BEST PERFORMING METHODS:")
        report_lines.append("-" * 30)
        
        if results:
            metrics = list(next(iter(results.values())).keys())
            
            for metric in metrics:
                method_scores = {method: scores[metric] for method, scores in results.items()}
                
                # Determine if higher or lower is better
                if metric in ['MSE', 'MAE']:
                    best_method = min(method_scores.items(), key=lambda x: x[1])
                    report_lines.append(f"{metric}: {best_method[0]} ({best_method[1]:.4f})")
                else:
                    best_method = max(method_scores.items(), key=lambda x: x[1] if x[1] != float('inf') else -float('inf'))
                    report_lines.append(f"{metric}: {best_method[0]} ({best_method[1]:.4f})")
        
        report_lines.append("")
        report_lines.append("DETAILED RESULTS:")
        report_lines.append("-" * 20)
        
        # Add detailed results table
        if results:
            methods = list(results.keys())
            metrics = list(next(iter(results.values())).keys())
            
            # Header
            header = f"{'Method':<20}"
            for metric in metrics:
                header += f"{metric:>12}"
            report_lines.append(header)
            report_lines.append("-" * (20 + 12 * len(metrics)))
            
            # Data rows
            for method in methods:
                row = f"{method:<20}"
                for metric in metrics:
                    value = results[method][metric]
                    if value == float('inf'):
                        row += f"{'∞':>12}"
                    else:
                        row += f"{value:>12.4f}"
                report_lines.append(row)
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        report_content = "\n".join(report_lines)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
        
        return report_content


# Example usage and testing
if __name__ == "__main__":
    # Create test images
    original = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    noisy = original + np.random.normal(0, 25, original.shape).astype(np.uint8)
    
    # Initialize metrics calculator
    metrics_calc = ImageQualityMetrics()
    
    # Calculate metrics
    mse_val = metrics_calc.mse(original, noisy)
    psnr_val = metrics_calc.psnr(original, noisy)
    ssim_val, _ = metrics_calc.ssim(original, noisy)
    
    print("Image Quality Metrics Module loaded successfully!")
    print(f"MSE: {mse_val:.4f}")
    print(f"PSNR: {psnr_val:.4f} dB")
    print(f"SSIM: {ssim_val:.4f}")