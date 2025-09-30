"""
Image Noise Simulation Module

This module provides comprehensive noise simulation capabilities for testing
image denoising algorithms. It includes various types of noise commonly
found in real-world imaging scenarios.

Author: Image Filtering Expert
Date: 2024
"""

import numpy as np
import cv2
from typing import Tuple, Union
import matplotlib.pyplot as plt


class NoiseSimulator:
    """
    A comprehensive noise simulation class for adding various types of noise to images.
    
    Supports:
    - Gaussian noise
    - Salt-and-pepper noise
    - Poisson noise
    - Speckle noise
    - Uniform noise
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the noise simulator.
        
        Args:
            seed (int): Random seed for reproducible results
        """
        np.random.seed(seed)
        self.seed = seed
    
    def add_gaussian_noise(self, image: np.ndarray, mean: float = 0, 
                          std: float = 25, clip: bool = True) -> np.ndarray:
        """
        Add Gaussian (normal) noise to an image.
        
        Args:
            image (np.ndarray): Input image
            mean (float): Mean of the Gaussian distribution
            std (float): Standard deviation of the Gaussian distribution
            clip (bool): Whether to clip values to valid range [0, 255]
            
        Returns:
            np.ndarray: Noisy image
        """
        noise = np.random.normal(mean, std, image.shape)
        noisy_image = image.astype(np.float64) + noise
        
        if clip:
            noisy_image = np.clip(noisy_image, 0, 255)
            
        return noisy_image.astype(np.uint8)
    
    def add_salt_pepper_noise(self, image: np.ndarray, salt_prob: float = 0.05, 
                             pepper_prob: float = 0.05) -> np.ndarray:
        """
        Add salt-and-pepper noise to an image.
        
        Args:
            image (np.ndarray): Input image
            salt_prob (float): Probability of salt noise (white pixels)
            pepper_prob (float): Probability of pepper noise (black pixels)
            
        Returns:
            np.ndarray: Noisy image
        """
        noisy_image = image.copy()
        
        # Generate random matrix
        random_matrix = np.random.random(image.shape[:2])
        
        # Add salt noise (white pixels)
        salt_mask = random_matrix < salt_prob
        if len(image.shape) == 3:
            noisy_image[salt_mask] = [255, 255, 255]
        else:
            noisy_image[salt_mask] = 255
            
        # Add pepper noise (black pixels)
        pepper_mask = random_matrix > (1 - pepper_prob)
        if len(image.shape) == 3:
            noisy_image[pepper_mask] = [0, 0, 0]
        else:
            noisy_image[pepper_mask] = 0
            
        return noisy_image
    
    def add_poisson_noise(self, image: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        Add Poisson noise to an image.
        
        Poisson noise is signal-dependent and commonly occurs in low-light conditions.
        
        Args:
            image (np.ndarray): Input image
            scale (float): Scaling factor for noise intensity
            
        Returns:
            np.ndarray: Noisy image
        """
        # Normalize image to [0, 1] range
        normalized_image = image.astype(np.float64) / 255.0
        
        # Scale the image
        scaled_image = normalized_image * scale
        
        # Generate Poisson noise
        noisy_image = np.random.poisson(scaled_image) / scale
        
        # Convert back to [0, 255] range
        noisy_image = np.clip(noisy_image * 255, 0, 255)
        
        return noisy_image.astype(np.uint8)
    
    def add_speckle_noise(self, image: np.ndarray, variance: float = 0.1) -> np.ndarray:
        """
        Add speckle (multiplicative) noise to an image.
        
        Speckle noise is commonly found in ultrasound and radar images.
        
        Args:
            image (np.ndarray): Input image
            variance (float): Variance of the speckle noise
            
        Returns:
            np.ndarray: Noisy image
        """
        # Generate multiplicative noise
        noise = np.random.normal(1, variance, image.shape)
        
        # Apply multiplicative noise
        noisy_image = image.astype(np.float64) * noise
        
        # Clip to valid range
        noisy_image = np.clip(noisy_image, 0, 255)
        
        return noisy_image.astype(np.uint8)
    
    def add_uniform_noise(self, image: np.ndarray, low: float = -20, 
                         high: float = 20) -> np.ndarray:
        """
        Add uniform noise to an image.
        
        Args:
            image (np.ndarray): Input image
            low (float): Lower bound of uniform distribution
            high (float): Upper bound of uniform distribution
            
        Returns:
            np.ndarray: Noisy image
        """
        noise = np.random.uniform(low, high, image.shape)
        noisy_image = image.astype(np.float64) + noise
        
        # Clip to valid range
        noisy_image = np.clip(noisy_image, 0, 255)
        
        return noisy_image.astype(np.uint8)
    
    def add_mixed_noise(self, image: np.ndarray, noise_types: list, 
                       noise_params: list) -> np.ndarray:
        """
        Add multiple types of noise to an image.
        
        Args:
            image (np.ndarray): Input image
            noise_types (list): List of noise type strings
            noise_params (list): List of parameter dictionaries for each noise type
            
        Returns:
            np.ndarray: Noisy image with mixed noise
        """
        noisy_image = image.copy()
        
        for noise_type, params in zip(noise_types, noise_params):
            if noise_type == 'gaussian':
                noisy_image = self.add_gaussian_noise(noisy_image, **params)
            elif noise_type == 'salt_pepper':
                noisy_image = self.add_salt_pepper_noise(noisy_image, **params)
            elif noise_type == 'poisson':
                noisy_image = self.add_poisson_noise(noisy_image, **params)
            elif noise_type == 'speckle':
                noisy_image = self.add_speckle_noise(noisy_image, **params)
            elif noise_type == 'uniform':
                noisy_image = self.add_uniform_noise(noisy_image, **params)
                
        return noisy_image
    
    def visualize_noise_comparison(self, original_image: np.ndarray, 
                                  noise_configs: dict, figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize the effect of different noise types on an image.
        
        Args:
            original_image (np.ndarray): Original clean image
            noise_configs (dict): Dictionary of noise configurations
            figsize (tuple): Figure size for the plot
        """
        num_noise_types = len(noise_configs) + 1  # +1 for original
        cols = 3
        rows = (num_noise_types + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
        
        # Display original image
        if len(original_image.shape) == 3:
            axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Display noisy images
        for idx, (noise_name, config) in enumerate(noise_configs.items(), 1):
            noise_type = config['type']
            params = config.get('params', {})
            
            if noise_type == 'gaussian':
                noisy_img = self.add_gaussian_noise(original_image, **params)
            elif noise_type == 'salt_pepper':
                noisy_img = self.add_salt_pepper_noise(original_image, **params)
            elif noise_type == 'poisson':
                noisy_img = self.add_poisson_noise(original_image, **params)
            elif noise_type == 'speckle':
                noisy_img = self.add_speckle_noise(original_image, **params)
            elif noise_type == 'uniform':
                noisy_img = self.add_uniform_noise(original_image, **params)
            
            if len(noisy_img.shape) == 3:
                axes[idx].imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
            else:
                axes[idx].imshow(noisy_img, cmap='gray')
            axes[idx].set_title(f'{noise_name}')
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(num_noise_types, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def generate_noise_dataset(self, image: np.ndarray, output_dir: str = None) -> dict:
        """
        Generate a comprehensive dataset with various noise types and levels.
        
        Args:
            image (np.ndarray): Original clean image
            output_dir (str): Directory to save noisy images (optional)
            
        Returns:
            dict: Dictionary containing all generated noisy images
        """
        dataset = {'original': image}
        
        # Gaussian noise with different standard deviations
        for std in [10, 25, 50]:
            key = f'gaussian_std_{std}'
            dataset[key] = self.add_gaussian_noise(image, std=std)
        
        # Salt-and-pepper noise with different probabilities
        for prob in [0.01, 0.05, 0.1]:
            key = f'salt_pepper_{prob}'
            dataset[key] = self.add_salt_pepper_noise(image, salt_prob=prob, pepper_prob=prob)
        
        # Poisson noise with different scales
        for scale in [0.5, 1.0, 2.0]:
            key = f'poisson_scale_{scale}'
            dataset[key] = self.add_poisson_noise(image, scale=scale)
        
        # Speckle noise with different variances
        for var in [0.05, 0.1, 0.2]:
            key = f'speckle_var_{var}'
            dataset[key] = self.add_speckle_noise(image, variance=var)
        
        # Save images if output directory is provided
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            for name, img in dataset.items():
                cv2.imwrite(os.path.join(output_dir, f'{name}.jpg'), img)
        
        return dataset


def calculate_noise_statistics(original: np.ndarray, noisy: np.ndarray) -> dict:
    """
    Calculate noise statistics between original and noisy images.
    
    Args:
        original (np.ndarray): Original clean image
        noisy (np.ndarray): Noisy image
        
    Returns:
        dict: Dictionary containing noise statistics
    """
    # Calculate noise (difference between images)
    noise = noisy.astype(np.float64) - original.astype(np.float64)
    
    stats = {
        'noise_mean': np.mean(noise),
        'noise_std': np.std(noise),
        'noise_variance': np.var(noise),
        'noise_min': np.min(noise),
        'noise_max': np.max(noise),
        'snr_db': 20 * np.log10(np.mean(original) / (np.std(noise) + 1e-10))
    }
    
    return stats


# Example usage and testing
if __name__ == "__main__":
    # Create a test image
    test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    # Initialize noise simulator
    noise_sim = NoiseSimulator(seed=42)
    
    # Test different noise types
    gaussian_noisy = noise_sim.add_gaussian_noise(test_image, std=25)
    salt_pepper_noisy = noise_sim.add_salt_pepper_noise(test_image, salt_prob=0.05, pepper_prob=0.05)
    poisson_noisy = noise_sim.add_poisson_noise(test_image, scale=1.0)
    speckle_noisy = noise_sim.add_speckle_noise(test_image, variance=0.1)
    
    print("Noise simulation module loaded successfully!")
    print(f"Original image shape: {test_image.shape}")
    print(f"Gaussian noisy image shape: {gaussian_noisy.shape}")
    print(f"Salt-pepper noisy image shape: {salt_pepper_noisy.shape}")