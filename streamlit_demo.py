"""
Interactive Image Filtering and Denoising Web Demo
==================================================

A Streamlit-based web application for demonstrating image noise reduction
and filtering techniques with real-time processing and comparison.

Features:
- Image upload and display
- Multiple noise simulation options
- Advanced filtering techniques
- Real-time quality metrics
- Side-by-side comparisons
- Downloadable results

Author: Image Filtering Expert
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io
import time
import sys
from pathlib import Path

# Add src directory to path
sys.path.append('./src')

# Import our custom modules
try:
    from noise_simulation import NoiseSimulator
    from advanced_filters import AdvancedImageFilters
    from evaluation_metrics import ImageQualityMetrics
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Please ensure all required modules are in the 'src' directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Image Filtering & Denoising Demo",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'noisy_image' not in st.session_state:
    st.session_state.noisy_image = None
if 'filtered_image' not in st.session_state:
    st.session_state.filtered_image = None
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0

# Initialize processing classes
@st.cache_resource
def load_processors():
    """Load and cache processing classes"""
    try:
        noise_sim = NoiseSimulator()
        filters = AdvancedImageFilters()
        metrics = ImageQualityMetrics()
        return noise_sim, filters, metrics
    except Exception as e:
        st.error(f"Error loading processors: {e}")
        return None, None, None

noise_sim, filters, metrics = load_processors()

if noise_sim is None:
    st.error("Failed to load processing modules. Please check your installation.")
    st.stop()

# Helper functions
def convert_image_format(image):
    """Convert PIL image to OpenCV format"""
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image

def convert_to_pil(image):
    """Convert OpenCV image to PIL format"""
    if len(image.shape) == 3:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        return Image.fromarray(image)

def create_download_link(image, filename):
    """Create a download link for processed image"""
    pil_image = convert_to_pil(image)
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ Image Filtering & Denoising Demo</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the comprehensive image filtering and denoising demonstration! 
    Upload an image, add noise, apply various filters, and compare the results with detailed metrics.
    """)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Image upload
        st.subheader("📁 Image Upload")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image to start processing"
        )
        
        # Load sample image option
        if st.button("📷 Use Sample Image"):
            # Create a sample image
            sample_img = np.zeros((256, 256, 3), dtype=np.uint8)
            cv2.rectangle(sample_img, (50, 50), (150, 150), (255, 255, 255), -1)
            cv2.circle(sample_img, (200, 200), 30, (128, 128, 128), -1)
            cv2.line(sample_img, (0, 0), (255, 255), (64, 64, 64), 2)
            
            # Add texture
            for i in range(0, 256, 20):
                cv2.line(sample_img, (i, 0), (i, 255), (32, 32, 32), 1)
                cv2.line(sample_img, (0, i), (255, i), (32, 32, 32), 1)
            
            st.session_state.original_image = sample_img
            st.success("Sample image loaded!")
        
        # Process uploaded image
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.original_image = convert_image_format(image)
            st.success("Image uploaded successfully!")
        
        # Noise configuration
        if st.session_state.original_image is not None:
            st.subheader("🔊 Noise Configuration")
            
            noise_type = st.selectbox(
                "Select Noise Type",
                ["None", "Gaussian", "Salt & Pepper", "Poisson", "Speckle", "Uniform"],
                help="Choose the type of noise to add to the image"
            )
            
            noise_params = {}
            if noise_type == "Gaussian":
                noise_params['mean'] = st.slider("Mean", -50, 50, 0)
                noise_params['std'] = st.slider("Standard Deviation", 1, 50, 25)
            elif noise_type == "Salt & Pepper":
                noise_params['amount'] = st.slider("Amount", 0.01, 0.2, 0.05, 0.01)
            elif noise_type == "Speckle":
                noise_params['std'] = st.slider("Standard Deviation", 0.01, 0.5, 0.1, 0.01)
            elif noise_type == "Uniform":
                noise_params['low'] = st.slider("Low", -50, 0, -30)
                noise_params['high'] = st.slider("High", 0, 50, 30)
            
            # Apply noise
            if st.button("🔊 Add Noise"):
                if noise_type == "None":
                    st.session_state.noisy_image = st.session_state.original_image.copy()
                else:
                    with st.spinner("Adding noise..."):
                        if noise_type == "Gaussian":
                            st.session_state.noisy_image = noise_sim.add_gaussian_noise(
                                st.session_state.original_image, **noise_params)
                        elif noise_type == "Salt & Pepper":
                            st.session_state.noisy_image = noise_sim.add_salt_pepper_noise(
                                st.session_state.original_image, **noise_params)
                        elif noise_type == "Poisson":
                            st.session_state.noisy_image = noise_sim.add_poisson_noise(
                                st.session_state.original_image)
                        elif noise_type == "Speckle":
                            st.session_state.noisy_image = noise_sim.add_speckle_noise(
                                st.session_state.original_image, **noise_params)
                        elif noise_type == "Uniform":
                            st.session_state.noisy_image = noise_sim.add_uniform_noise(
                                st.session_state.original_image, **noise_params)
                    st.success(f"{noise_type} noise added!")
            
            # Filter configuration
            if st.session_state.noisy_image is not None:
                st.subheader("🔧 Filter Configuration")
                
                filter_type = st.selectbox(
                    "Select Filter",
                    ["Gaussian", "Bilateral", "Non-Local Means", "Median", 
                     "Adaptive Median", "Wiener", "Anisotropic Diffusion", 
                     "Guided Filter", "Total Variation", "BM3D"],
                    help="Choose the filtering algorithm to apply"
                )
                
                filter_params = {}
                if filter_type == "Gaussian":
                    filter_params['sigma'] = st.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
                elif filter_type == "Bilateral":
                    filter_params['d'] = st.slider("Diameter", 5, 15, 9, 2)
                    filter_params['sigma_color'] = st.slider("Sigma Color", 10, 150, 75, 5)
                    filter_params['sigma_space'] = st.slider("Sigma Space", 10, 150, 75, 5)
                elif filter_type == "Non-Local Means":
                    filter_params['h'] = st.slider("Filter Strength", 1, 20, 10)
                    filter_params['template_window_size'] = st.slider("Template Window", 3, 9, 7, 2)
                    filter_params['search_window_size'] = st.slider("Search Window", 11, 25, 21, 2)
                elif filter_type == "Median":
                    filter_params['kernel_size'] = st.slider("Kernel Size", 3, 15, 5, 2)
                elif filter_type == "Adaptive Median":
                    filter_params['max_window_size'] = st.slider("Max Window Size", 3, 15, 7, 2)
                elif filter_type == "Wiener":
                    filter_params['noise_variance'] = st.slider("Noise Variance", 0.01, 1.0, 0.1, 0.01)
                elif filter_type == "Anisotropic Diffusion":
                    filter_params['num_iterations'] = st.slider("Iterations", 1, 20, 10)
                    filter_params['kappa'] = st.slider("Kappa", 10, 100, 50)
                    filter_params['gamma'] = st.slider("Gamma", 0.1, 0.3, 0.2, 0.01)
                elif filter_type == "Total Variation":
                    filter_params['weight'] = st.slider("Weight", 0.01, 0.5, 0.1, 0.01)
                    filter_params['max_iterations'] = st.slider("Max Iterations", 50, 300, 100, 10)
                
                # Apply filter
                if st.button("🔧 Apply Filter"):
                    with st.spinner(f"Applying {filter_type} filter..."):
                        start_time = time.time()
                        
                        try:
                            if filter_type == "Gaussian":
                                st.session_state.filtered_image = filters.enhanced_gaussian_filter(
                                    st.session_state.noisy_image, **filter_params)
                            elif filter_type == "Bilateral":
                                st.session_state.filtered_image = filters.enhanced_bilateral_filter(
                                    st.session_state.noisy_image, **filter_params)
                            elif filter_type == "Non-Local Means":
                                st.session_state.filtered_image = filters.enhanced_nlm_filter(
                                    st.session_state.noisy_image, **filter_params)
                            elif filter_type == "Median":
                                st.session_state.filtered_image = cv2.medianBlur(
                                    st.session_state.noisy_image, filter_params['kernel_size'])
                            elif filter_type == "Adaptive Median":
                                st.session_state.filtered_image = filters.adaptive_median_filter(
                                    st.session_state.noisy_image, **filter_params)
                            elif filter_type == "Wiener":
                                st.session_state.filtered_image = filters.wiener_filter(
                                    st.session_state.noisy_image, **filter_params)
                            elif filter_type == "Anisotropic Diffusion":
                                st.session_state.filtered_image = filters.anisotropic_diffusion(
                                    st.session_state.noisy_image, **filter_params)
                            elif filter_type == "Guided Filter":
                                st.session_state.filtered_image = filters.guided_filter(
                                    st.session_state.noisy_image, st.session_state.noisy_image)
                            elif filter_type == "Total Variation":
                                st.session_state.filtered_image = filters.total_variation_denoising(
                                    st.session_state.noisy_image, **filter_params)
                            elif filter_type == "BM3D":
                                st.session_state.filtered_image = filters.bm3d_denoising(
                                    st.session_state.noisy_image)
                            
                            st.session_state.processing_time = time.time() - start_time
                            st.success(f"{filter_type} filter applied in {st.session_state.processing_time:.3f}s!")
                            
                        except Exception as e:
                            st.error(f"Error applying filter: {e}")
    
    # Main content area
    if st.session_state.original_image is not None:
        # Image display tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Comparison", "📈 Metrics", "🔍 Analysis", "📥 Download"])
        
        with tab1:
            st.markdown('<h2 class="section-header">Image Comparison</h2>', unsafe_allow_html=True)
            
            # Display images side by side
            cols = st.columns(3)
            
            with cols[0]:
                st.subheader("Original")
                st.image(convert_to_pil(st.session_state.original_image), 
                        use_column_width=True)
            
            with cols[1]:
                if st.session_state.noisy_image is not None:
                    st.subheader("Noisy")
                    st.image(convert_to_pil(st.session_state.noisy_image), 
                            use_column_width=True)
                else:
                    st.info("Add noise to see noisy image")
            
            with cols[2]:
                if st.session_state.filtered_image is not None:
                    st.subheader("Filtered")
                    st.image(convert_to_pil(st.session_state.filtered_image), 
                            use_column_width=True)
                else:
                    st.info("Apply filter to see result")
        
        with tab2:
            st.markdown('<h2 class="section-header">Quality Metrics</h2>', unsafe_allow_html=True)
            
            if st.session_state.filtered_image is not None:
                # Calculate metrics
                with st.spinner("Calculating quality metrics..."):
                    try:
                        # Metrics between original and noisy
                        noisy_metrics = {
                            'MSE': metrics.calculate_mse(st.session_state.original_image, st.session_state.noisy_image),
                            'MAE': metrics.calculate_mae(st.session_state.original_image, st.session_state.noisy_image),
                            'PSNR': metrics.calculate_psnr(st.session_state.original_image, st.session_state.noisy_image),
                            'SSIM': metrics.calculate_ssim(st.session_state.original_image, st.session_state.noisy_image),
                            'UQI': metrics.calculate_uqi(st.session_state.original_image, st.session_state.noisy_image),
                            'SNR': metrics.calculate_snr(st.session_state.original_image, st.session_state.noisy_image)
                        }
                        
                        # Metrics between original and filtered
                        filtered_metrics = {
                            'MSE': metrics.calculate_mse(st.session_state.original_image, st.session_state.filtered_image),
                            'MAE': metrics.calculate_mae(st.session_state.original_image, st.session_state.filtered_image),
                            'PSNR': metrics.calculate_psnr(st.session_state.original_image, st.session_state.filtered_image),
                            'SSIM': metrics.calculate_ssim(st.session_state.original_image, st.session_state.filtered_image),
                            'UQI': metrics.calculate_uqi(st.session_state.original_image, st.session_state.filtered_image),
                            'SNR': metrics.calculate_snr(st.session_state.original_image, st.session_state.filtered_image)
                        }
                        
                        # Display metrics in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📉 Noisy vs Original")
                            for metric, value in noisy_metrics.items():
                                if value == float('inf'):
                                    st.metric(metric, "∞")
                                else:
                                    st.metric(metric, f"{value:.4f}")
                        
                        with col2:
                            st.subheader("✨ Filtered vs Original")
                            for metric, value in filtered_metrics.items():
                                if value == float('inf'):
                                    st.metric(metric, "∞")
                                else:
                                    # Calculate improvement
                                    if metric in ['MSE', 'MAE']:
                                        improvement = noisy_metrics[metric] - value
                                        st.metric(metric, f"{value:.4f}", f"{improvement:.4f}")
                                    else:
                                        improvement = value - noisy_metrics[metric]
                                        st.metric(metric, f"{value:.4f}", f"{improvement:.4f}")
                        
                        # Processing time
                        st.subheader("⏱️ Performance")
                        st.metric("Processing Time", f"{st.session_state.processing_time:.3f} seconds")
                        
                        # Create comparison chart
                        st.subheader("📊 Metrics Comparison")
                        
                        # Prepare data for plotting
                        metrics_df = pd.DataFrame({
                            'Noisy': [noisy_metrics[m] if noisy_metrics[m] != float('inf') else 50 for m in ['PSNR', 'SSIM', 'UQI', 'SNR']],
                            'Filtered': [filtered_metrics[m] if filtered_metrics[m] != float('inf') else 50 for m in ['PSNR', 'SSIM', 'UQI', 'SNR']]
                        }, index=['PSNR', 'SSIM', 'UQI', 'SNR'])
                        
                        st.bar_chart(metrics_df)
                        
                    except Exception as e:
                        st.error(f"Error calculating metrics: {e}")
            else:
                st.info("Apply a filter to see quality metrics")
        
        with tab3:
            st.markdown('<h2 class="section-header">Detailed Analysis</h2>', unsafe_allow_html=True)
            
            if st.session_state.filtered_image is not None:
                # Histogram comparison
                st.subheader("📈 Histogram Analysis")
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                images = [st.session_state.original_image, st.session_state.noisy_image, st.session_state.filtered_image]
                titles = ['Original', 'Noisy', 'Filtered']
                colors = ['blue', 'red', 'green']
                
                for i, (img, title, color) in enumerate(zip(images, titles, colors)):
                    if len(img.shape) == 3:
                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_img = img
                    
                    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
                    axes[i].plot(hist, color=color)
                    axes[i].set_title(title)
                    axes[i].set_xlabel('Pixel Intensity')
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Difference images
                st.subheader("🔍 Difference Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Noise (Original - Noisy)**")
                    noise_diff = cv2.absdiff(st.session_state.original_image, st.session_state.noisy_image)
                    st.image(convert_to_pil(noise_diff), use_column_width=True)
                
                with col2:
                    st.write("**Residual (Original - Filtered)**")
                    residual_diff = cv2.absdiff(st.session_state.original_image, st.session_state.filtered_image)
                    st.image(convert_to_pil(residual_diff), use_column_width=True)
                
                # Statistics
                st.subheader("📊 Image Statistics")
                
                stats_data = []
                for img, name in [(st.session_state.original_image, 'Original'), 
                                (st.session_state.noisy_image, 'Noisy'), 
                                (st.session_state.filtered_image, 'Filtered')]:
                    if len(img.shape) == 3:
                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_img = img
                    
                    stats_data.append({
                        'Image': name,
                        'Mean': np.mean(gray_img),
                        'Std': np.std(gray_img),
                        'Min': np.min(gray_img),
                        'Max': np.max(gray_img),
                        'Median': np.median(gray_img)
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df.round(2))
            else:
                st.info("Apply a filter to see detailed analysis")
        
        with tab4:
            st.markdown('<h2 class="section-header">Download Results</h2>', unsafe_allow_html=True)
            
            if st.session_state.filtered_image is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Original Image")
                    original_data = create_download_link(st.session_state.original_image, "original.png")
                    st.download_button(
                        label="📥 Download Original",
                        data=original_data,
                        file_name="original.png",
                        mime="image/png"
                    )
                
                with col2:
                    st.subheader("Noisy Image")
                    noisy_data = create_download_link(st.session_state.noisy_image, "noisy.png")
                    st.download_button(
                        label="📥 Download Noisy",
                        data=noisy_data,
                        file_name="noisy.png",
                        mime="image/png"
                    )
                
                with col3:
                    st.subheader("Filtered Image")
                    filtered_data = create_download_link(st.session_state.filtered_image, "filtered.png")
                    st.download_button(
                        label="📥 Download Filtered",
                        data=filtered_data,
                        file_name="filtered.png",
                        mime="image/png"
                    )
                
                # Generate and download report
                st.subheader("📄 Quality Report")
                if st.button("📊 Generate Report"):
                    with st.spinner("Generating comprehensive report..."):
                        try:
                            report_content = metrics.create_quality_report(
                                st.session_state.original_image,
                                {'Filtered': st.session_state.filtered_image},
                                noisy_image=st.session_state.noisy_image,
                                save_path=None
                            )
                            
                            st.download_button(
                                label="📥 Download Report",
                                data=report_content,
                                file_name="quality_report.txt",
                                mime="text/plain"
                            )
                            
                            st.success("Report generated successfully!")
                        except Exception as e:
                            st.error(f"Error generating report: {e}")
            else:
                st.info("Process an image to enable downloads")
    
    else:
        # Welcome screen
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Upload an image** or use the sample image from the sidebar
        2. **Add noise** to simulate real-world conditions
        3. **Apply filters** to reduce noise and enhance quality
        4. **Compare results** and analyze performance metrics
        5. **Download** processed images and reports
        
        ### 🔧 Available Features
        
        - **Noise Types**: Gaussian, Salt & Pepper, Poisson, Speckle, Uniform
        - **Filters**: Gaussian, Bilateral, Non-Local Means, Median, Adaptive Median, Wiener, Anisotropic Diffusion, Guided Filter, Total Variation, BM3D
        - **Metrics**: MSE, MAE, PSNR, SSIM, UQI, SNR
        - **Analysis**: Histograms, difference images, statistics
        - **Export**: High-quality images and detailed reports
        
        ### 📚 Educational Value
        
        This demo is perfect for:
        - Learning about image processing techniques
        - Comparing different denoising algorithms
        - Understanding quality metrics
        - Research and development
        - Academic projects
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🖼️ Image Filtering & Denoising Demo | Built with Streamlit | 
    <a href='https://github.com' target='_blank'>View Source</a></p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()