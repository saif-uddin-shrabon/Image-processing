"""
Batch Image Processing Module

This module provides functionality for processing multiple images in batch mode,
including parallel processing, progress tracking, and result management.

Features:
- Batch noise simulation
- Batch filtering with multiple algorithms
- Parallel processing support
- Progress tracking and logging
- Result organization and export

Author: Image Filtering Expert
Date: 2024
"""

import os
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Callable, Any
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import time
import logging

# Import our custom modules
from noise_simulation import NoiseSimulator
from advanced_filters import AdvancedImageFilters
from evaluation_metrics import ImageQualityMetrics


class BatchImageProcessor:
    """
    A comprehensive class for batch processing of images with various filtering techniques.
    """
    
    def __init__(self, max_workers: Optional[int] = None, use_multiprocessing: bool = True):
        """
        Initialize the batch processor.
        
        Args:
            max_workers (int, optional): Maximum number of worker threads/processes
            use_multiprocessing (bool): Whether to use multiprocessing or threading
        """
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.use_multiprocessing = use_multiprocessing
        
        # Initialize processing modules
        self.noise_simulator = NoiseSimulator()
        self.filters = AdvancedImageFilters()
        self.metrics = ImageQualityMetrics()
        
        # Setup logging
        self._setup_logging()
        
        # Results storage
        self.results = {}
        self.processing_stats = {}
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('batch_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_images_from_directory(self, directory_path: str, 
                                  supported_formats: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Load all images from a directory.
        
        Args:
            directory_path (str): Path to directory containing images
            supported_formats (list): List of supported image formats
            
        Returns:
            dict: Dictionary of loaded images {filename: image_array}
        """
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        images = {}
        directory = Path(directory_path)
        
        if not directory.exists():
            self.logger.error(f"Directory {directory_path} does not exist")
            return images
        
        self.logger.info(f"Loading images from {directory_path}")
        
        for file_path in directory.iterdir():
            if file_path.suffix.lower() in supported_formats:
                try:
                    image = cv2.imread(str(file_path))
                    if image is not None:
                        images[file_path.name] = image
                        self.logger.debug(f"Loaded {file_path.name}: {image.shape}")
                    else:
                        self.logger.warning(f"Could not load {file_path.name}")
                except Exception as e:
                    self.logger.error(f"Error loading {file_path.name}: {e}")
        
        self.logger.info(f"Successfully loaded {len(images)} images")
        return images
    
    def batch_add_noise(self, images: Dict[str, np.ndarray], 
                       noise_configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Add various types of noise to multiple images.
        
        Args:
            images (dict): Dictionary of images {filename: image_array}
            noise_configs (list): List of noise configuration dictionaries
            
        Returns:
            dict: Nested dictionary {image_name: {noise_type: noisy_image}}
        """
        self.logger.info(f"Adding noise to {len(images)} images with {len(noise_configs)} configurations")
        
        results = {}
        
        for img_name, image in tqdm(images.items(), desc="Adding noise to images"):
            results[img_name] = {'original': image.copy()}
            
            for noise_config in noise_configs:
                noise_type = noise_config.get('type', 'gaussian')
                noise_params = {k: v for k, v in noise_config.items() if k != 'type'}
                
                try:
                    if noise_type == 'gaussian':
                        noisy_img = self.noise_simulator.add_gaussian_noise(image, **noise_params)
                    elif noise_type == 'salt_pepper':
                        noisy_img = self.noise_simulator.add_salt_pepper_noise(image, **noise_params)
                    elif noise_type == 'poisson':
                        noisy_img = self.noise_simulator.add_poisson_noise(image, **noise_params)
                    elif noise_type == 'speckle':
                        noisy_img = self.noise_simulator.add_speckle_noise(image, **noise_params)
                    elif noise_type == 'uniform':
                        noisy_img = self.noise_simulator.add_uniform_noise(image, **noise_params)
                    else:
                        self.logger.warning(f"Unknown noise type: {noise_type}")
                        continue
                    
                    config_name = f"{noise_type}_{hash(str(noise_params)) % 10000}"
                    results[img_name][config_name] = noisy_img
                    
                except Exception as e:
                    self.logger.error(f"Error adding {noise_type} noise to {img_name}: {e}")
        
        return results
    
    def _process_single_image(self, args: Tuple) -> Tuple[str, str, np.ndarray, Dict[str, float]]:
        """
        Process a single image with a specific filter (for parallel processing).
        
        Args:
            args (tuple): (image_name, filter_name, image_array, filter_params, original_image)
            
        Returns:
            tuple: (image_name, filter_name, filtered_image, metrics)
        """
        img_name, filter_name, image, filter_params, original = args
        
        try:
            # Apply filter
            if filter_name == 'gaussian':
                filtered = self.filters.enhanced_gaussian_filter(image, **filter_params)
            elif filter_name == 'bilateral':
                filtered = self.filters.enhanced_bilateral_filter(image, **filter_params)
            elif filter_name == 'nlm':
                filtered = self.filters.enhanced_nlm_filter(image, **filter_params)
            elif filter_name == 'median':
                kernel_size = filter_params.get('kernel_size', 5)
                filtered = cv2.medianBlur(image, kernel_size)
            elif filter_name == 'adaptive_median':
                filtered = self.filters.adaptive_median_filter(image, **filter_params)
            elif filter_name == 'wiener':
                filtered = self.filters.wiener_filter(image, **filter_params)
            elif filter_name == 'anisotropic':
                filtered = self.filters.anisotropic_diffusion(image, **filter_params)
            elif filter_name == 'guided':
                filtered = self.filters.guided_filter(image, **filter_params)
            elif filter_name == 'tv':
                filtered = self.filters.total_variation_denoising(image, **filter_params)
            elif filter_name == 'bm3d':
                filtered = self.filters.bm3d_denoising(image, **filter_params)
            else:
                raise ValueError(f"Unknown filter: {filter_name}")
            
            # Calculate metrics if original is provided
            metrics = {}
            if original is not None:
                metrics = self.metrics.calculate_all_metrics(original, filtered)
            
            return img_name, filter_name, filtered, metrics
            
        except Exception as e:
            logging.error(f"Error processing {img_name} with {filter_name}: {e}")
            return img_name, filter_name, None, {}
    
    def batch_apply_filters(self, images: Dict[str, np.ndarray], 
                           filter_configs: List[Dict[str, Any]],
                           original_images: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Apply multiple filters to multiple images in parallel.
        
        Args:
            images (dict): Dictionary of images to process
            filter_configs (list): List of filter configuration dictionaries
            original_images (dict, optional): Original clean images for metrics calculation
            
        Returns:
            dict: Nested dictionary {image_name: {filter_name: {image: filtered, metrics: dict}}}
        """
        self.logger.info(f"Applying {len(filter_configs)} filters to {len(images)} images")
        
        # Prepare tasks for parallel processing
        tasks = []
        for img_name, image in images.items():
            original = original_images.get(img_name) if original_images else None
            
            for filter_config in filter_configs:
                filter_name = filter_config.get('name', 'unknown')
                filter_params = {k: v for k, v in filter_config.items() if k != 'name'}
                
                tasks.append((img_name, filter_name, image, filter_params, original))
        
        # Process in parallel
        results = {}
        
        executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(self._process_single_image, task) for task in tasks]
            
            # Collect results with progress bar
            for future in tqdm(futures, desc="Processing images"):
                try:
                    img_name, filter_name, filtered_img, metrics = future.result()
                    
                    if img_name not in results:
                        results[img_name] = {}
                    
                    if filtered_img is not None:
                        results[img_name][filter_name] = {
                            'image': filtered_img,
                            'metrics': metrics
                        }
                    
                except Exception as e:
                    self.logger.error(f"Error collecting result: {e}")
        
        return results
    
    def batch_evaluate_performance(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance across all processed images and filters.
        
        Args:
            results (dict): Results from batch_apply_filters
            
        Returns:
            dict: Performance statistics {filter_name: {metric: avg_value}}
        """
        self.logger.info("Evaluating batch processing performance")
        
        performance_stats = {}
        
        # Collect all metrics by filter
        filter_metrics = {}
        
        for img_name, img_results in results.items():
            for filter_name, filter_result in img_results.items():
                metrics = filter_result.get('metrics', {})
                
                if filter_name not in filter_metrics:
                    filter_metrics[filter_name] = {metric: [] for metric in metrics.keys()}
                
                for metric, value in metrics.items():
                    if value != float('inf') and not np.isnan(value):
                        filter_metrics[filter_name][metric].append(value)
        
        # Calculate statistics
        for filter_name, metrics_dict in filter_metrics.items():
            performance_stats[filter_name] = {}
            
            for metric, values in metrics_dict.items():
                if values:
                    performance_stats[filter_name][f'{metric}_mean'] = np.mean(values)
                    performance_stats[filter_name][f'{metric}_std'] = np.std(values)
                    performance_stats[filter_name][f'{metric}_min'] = np.min(values)
                    performance_stats[filter_name][f'{metric}_max'] = np.max(values)
        
        return performance_stats
    
    def save_results(self, results: Dict[str, Dict[str, Any]], 
                    output_directory: str, save_images: bool = True,
                    save_metrics: bool = True) -> None:
        """
        Save batch processing results to disk.
        
        Args:
            results (dict): Results from batch processing
            output_directory (str): Directory to save results
            save_images (bool): Whether to save processed images
            save_metrics (bool): Whether to save metrics data
        """
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving results to {output_directory}")
        
        # Save images
        if save_images:
            images_path = output_path / 'images'
            images_path.mkdir(exist_ok=True)
            
            for img_name, img_results in tqdm(results.items(), desc="Saving images"):
                img_dir = images_path / Path(img_name).stem
                img_dir.mkdir(exist_ok=True)
                
                for filter_name, filter_result in img_results.items():
                    filtered_img = filter_result.get('image')
                    if filtered_img is not None:
                        output_file = img_dir / f"{filter_name}.png"
                        cv2.imwrite(str(output_file), filtered_img)
        
        # Save metrics
        if save_metrics:
            metrics_data = {}
            
            for img_name, img_results in results.items():
                metrics_data[img_name] = {}
                for filter_name, filter_result in img_results.items():
                    metrics_data[img_name][filter_name] = filter_result.get('metrics', {})
            
            # Save as JSON
            metrics_file = output_path / 'metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            # Save as pickle for Python objects
            pickle_file = output_path / 'results.pkl'
            with open(pickle_file, 'wb') as f:
                pickle.dump(results, f)
        
        self.logger.info("Results saved successfully")
    
    def create_batch_report(self, results: Dict[str, Dict[str, Any]], 
                           performance_stats: Dict[str, Dict[str, float]],
                           output_path: str) -> str:
        """
        Create a comprehensive batch processing report.
        
        Args:
            results (dict): Batch processing results
            performance_stats (dict): Performance statistics
            output_path (str): Path to save the report
            
        Returns:
            str: Report content
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BATCH IMAGE PROCESSING REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Processing summary
        total_images = len(results)
        total_filters = len(performance_stats) if performance_stats else 0
        
        report_lines.append("PROCESSING SUMMARY:")
        report_lines.append("-" * 20)
        report_lines.append(f"Total Images Processed: {total_images}")
        report_lines.append(f"Total Filters Applied: {total_filters}")
        report_lines.append(f"Total Processing Tasks: {total_images * total_filters}")
        report_lines.append("")
        
        # Performance statistics
        if performance_stats:
            report_lines.append("PERFORMANCE STATISTICS:")
            report_lines.append("-" * 25)
            
            # Find best performing filters for each metric
            metrics_list = []
            if performance_stats:
                first_filter = next(iter(performance_stats.values()))
                metrics_list = [key.replace('_mean', '') for key in first_filter.keys() if key.endswith('_mean')]
            
            for metric in metrics_list:
                metric_key = f'{metric}_mean'
                filter_scores = {filter_name: stats.get(metric_key, 0) 
                               for filter_name, stats in performance_stats.items() 
                               if metric_key in stats}
                
                if filter_scores:
                    if metric in ['MSE', 'MAE']:
                        best_filter = min(filter_scores.items(), key=lambda x: x[1])
                    else:
                        best_filter = max(filter_scores.items(), key=lambda x: x[1])
                    
                    report_lines.append(f"Best {metric}: {best_filter[0]} ({best_filter[1]:.4f})")
            
            report_lines.append("")
            
            # Detailed statistics table
            report_lines.append("DETAILED STATISTICS:")
            report_lines.append("-" * 20)
            
            header = f"{'Filter':<20}"
            for metric in metrics_list:
                header += f"{metric + '_avg':>12}"
            report_lines.append(header)
            report_lines.append("-" * (20 + 12 * len(metrics_list)))
            
            for filter_name, stats in performance_stats.items():
                row = f"{filter_name:<20}"
                for metric in metrics_list:
                    metric_key = f'{metric}_mean'
                    value = stats.get(metric_key, 0)
                    row += f"{value:>12.4f}"
                report_lines.append(row)
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_content = "\n".join(report_lines)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        return report_content
    
    def visualize_batch_results(self, results: Dict[str, Dict[str, Any]], 
                               sample_images: List[str] = None,
                               figsize: Tuple[int, int] = (20, 12)) -> None:
        """
        Create visualization of batch processing results.
        
        Args:
            results (dict): Batch processing results
            sample_images (list): List of image names to visualize (all if None)
            figsize (tuple): Figure size
        """
        if not results:
            return
        
        # Select sample images
        if sample_images is None:
            sample_images = list(results.keys())[:4]  # Show first 4 images
        
        # Get filter names
        first_img_results = next(iter(results.values()))
        filter_names = list(first_img_results.keys())
        
        # Create subplots
        n_images = len(sample_images)
        n_filters = len(filter_names)
        
        fig, axes = plt.subplots(n_images, n_filters, figsize=figsize)
        if n_images == 1:
            axes = axes.reshape(1, -1)
        if n_filters == 1:
            axes = axes.reshape(-1, 1)
        
        for i, img_name in enumerate(sample_images):
            for j, filter_name in enumerate(filter_names):
                if img_name in results and filter_name in results[img_name]:
                    filtered_img = results[img_name][filter_name]['image']
                    
                    if len(filtered_img.shape) == 3:
                        axes[i, j].imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
                    else:
                        axes[i, j].imshow(filtered_img, cmap='gray')
                    
                    # Add title with metrics if available
                    metrics = results[img_name][filter_name].get('metrics', {})
                    title = filter_name
                    if 'PSNR' in metrics:
                        title += f"\nPSNR: {metrics['PSNR']:.2f}"
                    
                    axes[i, j].set_title(title, fontsize=8)
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Initialize batch processor
    processor = BatchImageProcessor(max_workers=4)
    
    # Example noise configurations
    noise_configs = [
        {'type': 'gaussian', 'mean': 0, 'std': 25},
        {'type': 'salt_pepper', 'amount': 0.05},
        {'type': 'poisson'}
    ]
    
    # Example filter configurations
    filter_configs = [
        {'name': 'gaussian', 'sigma': 1.0},
        {'name': 'bilateral', 'd': 9, 'sigma_color': 75, 'sigma_space': 75},
        {'name': 'nlm', 'h': 10},
        {'name': 'median', 'kernel_size': 5}
    ]
    
    print("Batch Image Processing Module loaded successfully!")
    print(f"Max workers: {processor.max_workers}")
    print(f"Available filters: {len(filter_configs)}")
    print(f"Available noise types: {len(noise_configs)}")