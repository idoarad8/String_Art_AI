"""
String Art Generator using Radon Transform
Converts an image into string art by placing nails in a circle and 
using the Radon transform to determine optimal string paths.
GPU-accelerated using PyTorch.
"""

import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from skimage.transform import radon
from scipy.ndimage import gaussian_filter
import math

# Try to import PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


class StringArtGenerator:
    def __init__(self, image_path: str, num_nails: int = 300, image_size: int = 500, use_gpu: bool = True):
        """
        Initialize the String Art Generator.
        
        Args:
            image_path: Path to the input image
            num_nails: Number of nails to place in the circle
            image_size: Size of the processed image (will be resized)
            use_gpu: Whether to use GPU acceleration (if available)
        """
        self.num_nails = num_nails
        self.image_size = image_size
        
        # Setup GPU if available
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        if self.use_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cuda':
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("GPU requested but not available, using CPU")
                self.use_gpu = False
        else:
            self.device = None
            print("Using CPU (PyTorch not available or GPU disabled)")
        
        # Load and process image
        self.image = self._load_image(image_path)
        self.target_image = self._process_image(self.image)
        
        # Calculate nail positions in a circle
        self.nail_positions = self._calculate_nail_positions()
        
        # Initialize working image (will be drawn on)
        if self.use_gpu:
            self.working_image_gpu = torch.zeros((image_size, image_size), 
                                                 dtype=torch.float32, device=self.device)
            self.working_image = None  # Will sync when needed
        else:
            self.working_image = np.zeros((image_size, image_size), dtype=np.float32)
            self.working_image_gpu = None
        
        # Store the nail sequence
        self.nail_sequence = []
        
        # Precompute Radon transform for guidance
        self._precompute_radon_transform()
        
        # Precompute nail positions on GPU if using GPU
        if self.use_gpu:
            self.nail_positions_gpu = torch.from_numpy(self.nail_positions).float().to(self.device)
            self.target_image_gpu = torch.from_numpy(self.target_image).float().to(self.device)
        else:
            self.nail_positions_gpu = None
            self.target_image_gpu = None
        
    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from path."""
        return Image.open(image_path)
    
    def _process_image(self, image: Image.Image) -> np.ndarray:
        """Convert image to grayscale and resize."""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to desired size
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Apply slight blur to smooth edges
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Invert: darker areas should have more string
        img_array = 1.0 - img_array
        
        # Enhance contrast slightly
        img_array = np.power(img_array, 0.8)
        
        return img_array
    
    def _calculate_nail_positions(self) -> np.ndarray:
        """Calculate positions of nails in a circle."""
        center = self.image_size / 2
        radius = self.image_size * 0.45  # Leave some margin
        
        angles = np.linspace(0, 2 * np.pi, self.num_nails, endpoint=False)
        
        positions = np.zeros((self.num_nails, 2))
        positions[:, 0] = center + radius * np.cos(angles)
        positions[:, 1] = center + radius * np.sin(angles)
        
        return positions
    
    def _precompute_radon_transform(self):
        """Precompute Radon transform to identify important lines."""
        print("Computing Radon transform...")
        
        # Use more angles for better resolution
        num_angles = min(360, self.num_nails * 2)
        theta = np.linspace(0., 180., num_angles, endpoint=False)
        
        # Compute Radon transform
        sinogram = radon(self.target_image, theta=theta, circle=True)
        
        self.sinogram = sinogram
        self.theta = theta
        self.num_radon_angles = num_angles
        
        # Normalize sinogram
        if sinogram.max() > sinogram.min():
            self.sinogram_normalized = (sinogram - sinogram.min()) / (sinogram.max() - sinogram.min())
        else:
            self.sinogram_normalized = np.zeros_like(sinogram)
        
        print(f"  Radon transform computed: {sinogram.shape}")
    
    def _get_line_radon_score(self, nail_a: int, nail_b: int) -> float:
        """
        Get the score for a line between two nails based on Radon transform.
        Uses proper line equation to find corresponding sinogram value.
        """
        pos_a = self.nail_positions[nail_a]
        pos_b = self.nail_positions[nail_b]
        
        # Line direction
        dx = pos_b[0] - pos_a[0]
        dy = pos_b[1] - pos_a[1]
        line_length = np.sqrt(dx*dx + dy*dy)
        
        if line_length < 1e-6:
            return 0.0
        
        # Normalize direction
        dx_norm = dx / line_length
        dy_norm = dy / line_length
        
        # Angle in degrees (0 to 180 for Radon transform)
        # The angle is the perpendicular to the line direction
        angle_rad = np.arctan2(dy_norm, dx_norm) + np.pi/2
        angle_deg = np.degrees(angle_rad) % 180.0
        
        # Find closest theta index
        theta_idx = int(angle_deg / 180.0 * self.num_radon_angles) % self.num_radon_angles
        
        # Calculate the perpendicular distance from origin to line
        # Line equation: ax + by + c = 0 where (a,b) is perpendicular direction
        center = self.image_size / 2
        # Point on line (midpoint)
        mid_x = (pos_a[0] + pos_b[0]) / 2 - center
        mid_y = (pos_a[1] + pos_b[1]) / 2 - center
        
        # Perpendicular direction (normal to line)
        perp_x = -dy_norm
        perp_y = dx_norm
        
        # Distance from origin along perpendicular
        distance = mid_x * perp_x + mid_y * perp_y
        
        # Map to sinogram column
        sinogram_cols = self.sinogram.shape[1]
        # Radon transform uses distance from -max_dist to +max_dist
        max_dist = self.image_size / np.sqrt(2)  # Maximum distance for circle
        col_idx = int((distance + max_dist) / (2 * max_dist) * sinogram_cols)
        col_idx = max(0, min(sinogram_cols - 1, col_idx))
        
        # Get score from normalized sinogram
        if 0 <= theta_idx < self.num_radon_angles:
            score = self.sinogram_normalized[theta_idx, col_idx]
            return float(score)
        return 0.0
    
    def _draw_line(self, start_nail: int, end_nail: int, intensity: float = 1.0):
        """Draw a line between two nails on the working image using anti-aliasing."""
        if self.use_gpu:
            self._draw_line_gpu(start_nail, end_nail, intensity)
        else:
            self._draw_line_cpu(start_nail, end_nail, intensity)
    
    def _draw_line_cpu(self, start_nail: int, end_nail: int, intensity: float = 1.0):
        """CPU version of line drawing."""
        start_pos = self.nail_positions[start_nail]
        end_pos = self.nail_positions[end_nail]
        
        x0, y0 = start_pos[0], start_pos[1]
        x1, y1 = end_pos[0], end_pos[1]
        
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < 1e-6:
            return
        
        # Optimize: use fewer steps and vectorized operations where possible
        steps = int(length) + 1  # Reduced from 2*length
        if steps < 2:
            steps = 2
        
        # Pre-allocate arrays for faster indexing
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            x = x0 + t * dx
            y = y0 + t * dy
            
            x_floor = int(np.floor(x))
            y_floor = int(np.floor(y))
            x_ceil = x_floor + 1
            y_ceil = y_floor + 1
            
            wx = x - x_floor
            wy = y - y_floor
            
            # Batch boundary checks
            if 0 <= x_floor < self.image_size and 0 <= y_floor < self.image_size:
                self.working_image[y_floor, x_floor] += intensity * (1 - wx) * (1 - wy)
            if 0 <= x_ceil < self.image_size and 0 <= y_floor < self.image_size:
                self.working_image[y_floor, x_ceil] += intensity * wx * (1 - wy)
            if 0 <= x_floor < self.image_size and 0 <= y_ceil < self.image_size:
                self.working_image[y_ceil, x_floor] += intensity * (1 - wx) * wy
            if 0 <= x_ceil < self.image_size and 0 <= y_ceil < self.image_size:
                self.working_image[y_ceil, x_ceil] += intensity * wx * wy
    
    def _draw_line_gpu(self, start_nail: int, end_nail: int, intensity: float = 1.0):
        """GPU-accelerated line drawing using fully vectorized operations."""
        start_pos = self.nail_positions_gpu[start_nail]
        end_pos = self.nail_positions_gpu[end_nail]
        
        x0, y0 = start_pos[0], start_pos[1]
        x1, y1 = end_pos[0], end_pos[1]
        
        dx = x1 - x0
        dy = y1 - y0
        length = torch.sqrt(dx*dx + dy*dy)
        
        if length < 1e-6:
            return
        
        # Vectorized line drawing - use fewer steps for better performance
        steps = int(length.item()) + 1  # Reduced from 2*length for speed
        if steps < 2:
            steps = 2
        
        t = torch.linspace(0, 1, steps, device=self.device)
        
        x = x0 + t * dx
        y = y0 + t * dy
        
        # Bilinear interpolation
        x_floor = torch.floor(x).long()
        y_floor = torch.floor(y).long()
        x_ceil = x_floor + 1
        y_ceil = y_floor + 1
        
        wx = x - x_floor.float()
        wy = y - y_floor.float()
        
        # Create masks for valid pixels
        valid_ff = (x_floor >= 0) & (x_floor < self.image_size) & (y_floor >= 0) & (y_floor < self.image_size)
        valid_cf = (x_ceil >= 0) & (x_ceil < self.image_size) & (y_floor >= 0) & (y_floor < self.image_size)
        valid_fc = (x_floor >= 0) & (x_floor < self.image_size) & (y_ceil >= 0) & (y_ceil < self.image_size)
        valid_cc = (x_ceil >= 0) & (x_ceil < self.image_size) & (y_ceil >= 0) & (y_ceil < self.image_size)
        
        # Calculate weights
        w_ff = intensity * (1 - wx) * (1 - wy)
        w_cf = intensity * wx * (1 - wy)
        w_fc = intensity * (1 - wx) * wy
        w_cc = intensity * wx * wy
        
        # Use scatter_add for vectorized accumulation (much faster than loops)
        if valid_ff.any():
            indices_ff = y_floor[valid_ff] * self.image_size + x_floor[valid_ff]
            self.working_image_gpu.view(-1).index_add_(0, indices_ff, w_ff[valid_ff])
        
        if valid_cf.any():
            indices_cf = y_floor[valid_cf] * self.image_size + x_ceil[valid_cf]
            self.working_image_gpu.view(-1).index_add_(0, indices_cf, w_cf[valid_cf])
        
        if valid_fc.any():
            indices_fc = y_ceil[valid_fc] * self.image_size + x_floor[valid_fc]
            self.working_image_gpu.view(-1).index_add_(0, indices_fc, w_fc[valid_fc])
        
        if valid_cc.any():
            indices_cc = y_ceil[valid_cc] * self.image_size + x_ceil[valid_cc]
            self.working_image_gpu.view(-1).index_add_(0, indices_cc, w_cc[valid_cc])
    
    def _calculate_line_benefit(self, current_nail: int, candidate_nail: int) -> float:
        """Calculate benefit for a single line (CPU fallback)."""
        if self.use_gpu:
            return self._calculate_line_benefit_gpu(current_nail, candidate_nail)
        else:
            return self._calculate_line_benefit_cpu(current_nail, candidate_nail)
    
    def _calculate_line_benefit_cpu(self, current_nail: int, candidate_nail: int) -> float:
        """CPU version of benefit calculation."""
        start_pos = self.nail_positions[current_nail]
        end_pos = self.nail_positions[candidate_nail]
        
        x0, y0 = start_pos[0], start_pos[1]
        x1, y1 = end_pos[0], end_pos[1]
        
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < 1e-6:
            return -1000
        
        # Optimize: use fewer steps and vectorized array operations
        steps = int(length) + 1  # Reduced from 2*length
        if steps < 2:
            steps = 2
        
        # Pre-compute all points along line
        t_values = np.linspace(0, 1, steps)
        x_points = x0 + t_values * dx
        y_points = y0 + t_values * dy
        
        # Round and clip in batch
        x_int = np.round(x_points).astype(np.int32)
        y_int = np.round(y_points).astype(np.int32)
        
        # Filter valid pixels
        valid_mask = (x_int >= 0) & (x_int < self.image_size) & (y_int >= 0) & (y_int < self.image_size)
        x_valid = x_int[valid_mask]
        y_valid = y_int[valid_mask]
        
        if len(x_valid) == 0:
            return -1000
        
        # Vectorized intensity lookup
        target_intensities = self.target_image[y_valid, x_valid]
        current_intensities = np.clip(self.working_image[y_valid, x_valid], 0, 1.0)
        differences = target_intensities - current_intensities
        total_benefit = np.sum(np.maximum(differences, 0))
        pixels_checked = len(x_valid)
        
        if pixels_checked == 0:
            return -1000
        
        avg_benefit = total_benefit / pixels_checked
        radon_score = self._get_line_radon_score(current_nail, candidate_nail)
        combined_score = 0.85 * avg_benefit + 0.15 * radon_score
        length_penalty = length * 0.0003
        
        return combined_score - length_penalty
    
    def _calculate_line_benefit_gpu(self, current_nail: int, candidate_nail: int) -> float:
        """GPU-accelerated benefit calculation."""
        start_pos = self.nail_positions_gpu[current_nail]
        end_pos = self.nail_positions_gpu[candidate_nail]
        
        x0, y0 = start_pos[0], start_pos[1]
        x1, y1 = end_pos[0], end_pos[1]
        
        dx = x1 - x0
        dy = y1 - y0
        length = torch.sqrt(dx*dx + dy*dy)
        
        if length < 1e-6:
            return -1000.0
        
        steps = int(length.item() * 2) + 1
        t = torch.linspace(0, 1, steps, device=self.device)
        
        x = x0 + t * dx
        y = y0 + t * dy
        
        x_int = torch.round(x).long()
        y_int = torch.round(y).long()
        
        # Create mask for valid pixels
        valid = (x_int >= 0) & (x_int < self.image_size) & (y_int >= 0) & (y_int < self.image_size)
        
        if valid.sum() == 0:
            return -1000.0
        
        # Gather pixel values
        x_valid = x_int[valid]
        y_valid = y_int[valid]
        
        target_vals = self.target_image_gpu[y_valid, x_valid]
        current_vals = torch.clamp(self.working_image_gpu[y_valid, x_valid], 0, 1.0)
        
        difference = target_vals - current_vals
        benefit = torch.clamp(difference, min=0).sum()
        
        avg_benefit = (benefit / valid.sum()).item()
        radon_score = self._get_line_radon_score(current_nail, candidate_nail)
        combined_score = 0.85 * avg_benefit + 0.15 * radon_score
        length_penalty = length.item() * 0.0003
        
        return combined_score - length_penalty
    
    def _calculate_line_benefits_batch_gpu(self, current_nail: int, candidates: np.ndarray) -> np.ndarray:
        """GPU-accelerated batch benefit calculation for multiple candidates."""
        if not self.use_gpu:
            # Fallback to CPU
            return np.array([self._calculate_line_benefit_cpu(current_nail, c) for c in candidates])
        
        current_pos = self.nail_positions_gpu[current_nail]
        candidate_positions = self.nail_positions_gpu[candidates]
        
        # Vectorize calculations
        starts = current_pos.unsqueeze(0).expand(len(candidates), -1)
        ends = candidate_positions
        
        dx = ends[:, 0] - starts[:, 0]
        dy = ends[:, 1] - starts[:, 1]
        lengths = torch.sqrt(dx*dx + dy*dy)
        
        # Filter out zero-length lines
        valid_mask = lengths > 1e-6
        
        scores = torch.full((len(candidates),), -1000.0, device=self.device)
        
        if valid_mask.sum() > 0:
            # Process valid candidates - vectorized where possible
            valid_indices = torch.where(valid_mask)[0]
            valid_starts = starts[valid_indices]
            valid_ends = ends[valid_indices]
            valid_lengths = lengths[valid_indices]
            
            # Process in chunks to avoid memory issues
            chunk_size = 50
            for chunk_start in range(0, len(valid_indices), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(valid_indices))
                chunk_indices = valid_indices[chunk_start:chunk_end]
                chunk_starts = valid_starts[chunk_start:chunk_end]
                chunk_ends = valid_ends[chunk_start:chunk_end]
                chunk_lengths = valid_lengths[chunk_start:chunk_end]
                
                # Get max steps for this chunk to create uniform tensors
                max_steps_chunk = int(chunk_lengths.max().item()) + 1
                if max_steps_chunk < 2:
                    max_steps_chunk = 2
                
                # Sample points for all lines in chunk
                t = torch.linspace(0, 1, max_steps_chunk, device=self.device).unsqueeze(0)
                
                for i, idx in enumerate(chunk_indices):
                    start = chunk_starts[i]
                    end = chunk_ends[i]
                    length = chunk_lengths[i]
                    
                    # Use fewer steps for shorter lines
                    n_steps = min(int(length.item()) + 1, max_steps_chunk)
                    t_line = t[0, :n_steps]
                    
                    x = start[0] + t_line * (end[0] - start[0])
                    y = start[1] + t_line * (end[1] - start[1])
                    
                    x_int = torch.round(x).long()
                    y_int = torch.round(y).long()
                    
                    valid = (x_int >= 0) & (x_int < self.image_size) & (y_int >= 0) & (y_int < self.image_size)
                    
                    if valid.sum() > 0:
                        target_vals = self.target_image_gpu[y_int[valid], x_int[valid]]
                        current_vals = torch.clamp(self.working_image_gpu[y_int[valid], x_int[valid]], 0, 1.0)
                        difference = target_vals - current_vals
                        benefit = torch.clamp(difference, min=0).sum()
                        avg_benefit = (benefit / valid.sum()).item()
                        
                        # Get Radon score (CPU operation but cached)
                        radon_score = self._get_line_radon_score(current_nail, candidates[idx.item()])
                        combined = 0.85 * avg_benefit + 0.15 * radon_score
                        length_penalty = length.item() * 0.0003
                        scores[idx] = combined - length_penalty
        
        return scores.cpu().numpy()
    
    def generate_string_art(self, num_iterations: int = 3000, 
                          min_distance: int = 10, 
                          max_distance: Optional[int] = None) -> List[int]:
        """
        Generate the string art by iteratively selecting nails.
        Uses Radon transform to guide line selection.
        
        Args:
            num_iterations: Number of string wraps to perform
            min_distance: Minimum nail distance to jump (avoid adjacent nails)
            max_distance: Maximum nail distance to jump (None = no limit)
        
        Returns:
            List of nail indices in order
        """
        if max_distance is None:
            max_distance = self.num_nails // 2
        
        if self.use_gpu:
            self.working_image_gpu.zero_()
        else:
            self.working_image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        self.nail_sequence = []
        
        # Start at a random nail
        current_nail = np.random.randint(0, self.num_nails)
        self.nail_sequence.append(current_nail)
        
        print(f"Generating string art with {num_iterations} iterations...")
        print(f"  Using Radon transform guidance...")
        if self.use_gpu:
            print(f"  GPU acceleration enabled")
        
        for iteration in range(num_iterations):
            if (iteration + 1) % 100 == 0:
                print(f"  Progress: {iteration + 1}/{num_iterations}")
            
            best_nail = None
            best_score = -float('inf')
            
            # Adaptive candidate selection - start with fewer, expand if needed
            num_candidates = min(150, self.num_nails)  # Reduced from 200
            
            # Create candidate list
            all_nails = np.arange(self.num_nails)
            candidates = np.random.choice(all_nails, num_candidates, replace=False)
            
            # Filter by distance constraints
            distances = np.minimum(np.abs(candidates - current_nail),
                                 self.num_nails - np.abs(candidates - current_nail))
            valid_candidates = candidates[(distances >= min_distance) & (distances <= max_distance)]
            
            if len(valid_candidates) > 0:
                # Always use batch GPU calculation if GPU available and enough candidates
                if self.use_gpu and len(valid_candidates) > 5:
                    # Batch GPU calculation (much faster)
                    scores = self._calculate_line_benefits_batch_gpu(current_nail, valid_candidates)
                    best_idx = np.argmax(scores)
                    best_score = scores[best_idx]
                    if best_score > -1000:
                        best_nail = valid_candidates[best_idx]
                else:
                    # Individual calculations (CPU or small batches)
                    for candidate in valid_candidates[:100]:  # Limit to first 100 for speed
                        score = self._calculate_line_benefit(current_nail, candidate)
                        if score > best_score:
                            best_score = score
                            best_nail = candidate
            
            # If no valid nail found, relax constraints
            if best_nail is None:
                relaxed_candidates = candidates[distances >= min_distance]
                if len(relaxed_candidates) > 0:
                    if self.use_gpu and len(relaxed_candidates) > 5:
                        scores = self._calculate_line_benefits_batch_gpu(current_nail, relaxed_candidates)
                        best_idx = np.argmax(scores)
                        best_score = scores[best_idx]
                        if best_score > -1000:
                            best_nail = relaxed_candidates[best_idx]
                    else:
                        for candidate in relaxed_candidates[:100]:  # Limit to first 100
                            score = self._calculate_line_benefit(current_nail, candidate)
                            if score > best_score:
                                best_score = score
                                best_nail = candidate
            
            # Draw the line
            if best_nail is not None and best_score > 0:
                intensity = 0.3
                self._draw_line(current_nail, best_nail, intensity)
                self.nail_sequence.append(best_nail)
                current_nail = best_nail
            else:
                # Fallback: move to next valid nail
                current_nail = (current_nail + min_distance) % self.num_nails
                self.nail_sequence.append(current_nail)
        
        # Sync GPU to CPU if needed
        if self.use_gpu:
            self.working_image = self.working_image_gpu.cpu().numpy()
        
        # Apply slight smoothing to final result
        self.working_image = gaussian_filter(self.working_image, sigma=0.5)
        
        print(f"  Complete! Generated sequence of {len(self.nail_sequence)} nails.")
        return self.nail_sequence
    
    def visualize(self, show_nails: bool = True, show_sequence: bool = True,
                  figsize: Tuple[int, int] = (15, 5), show_radon: bool = False):
        """
        Visualize the string art result.
        
        Args:
            show_nails: Whether to show nail positions
            show_sequence: Whether to show the string path
            figsize: Figure size
            show_radon: Whether to show Radon transform visualization
        """
        num_plots = 4 if show_radon else 3
        fig, axes = plt.subplots(1, num_plots, figsize=(figsize[0] + 5 if show_radon else figsize[0], figsize[1]))
        
        plot_idx = 0
        
        # Original image (not inverted for display)
        original_display = 1.0 - self.target_image
        axes[plot_idx].imshow(original_display, cmap='gray')
        axes[plot_idx].set_title('Original Image')
        axes[plot_idx].axis('off')
        plot_idx += 1
        
        # Sync GPU image if needed
        if self.use_gpu and self.working_image is None:
            self.working_image = self.working_image_gpu.cpu().numpy()
        
        # String art result
        result = np.clip(self.working_image, 0, 1)
        # Invert for display to match original
        result_display = 1.0 - result
        axes[plot_idx].imshow(result_display, cmap='gray')
        if show_nails:
            axes[plot_idx].plot(self.nail_positions[:, 0], self.nail_positions[:, 1], 
                        'ro', markersize=2, alpha=0.5)
        axes[plot_idx].set_title('String Art Result')
        axes[plot_idx].axis('off')
        plot_idx += 1
        
        # Radon transform visualization
        if show_radon:
            axes[plot_idx].imshow(self.sinogram_normalized, aspect='auto', cmap='gray')
            axes[plot_idx].set_title('Radon Transform (Sinogram)')
            axes[plot_idx].set_xlabel('Offset')
            axes[plot_idx].set_ylabel('Angle (0-180°)')
            plot_idx += 1
        
        # Show the path
        if show_sequence and len(self.nail_sequence) > 1:
            axes[plot_idx].axis([0, self.image_size, 0, self.image_size])
            axes[plot_idx].set_aspect('equal')
            axes[plot_idx].invert_yaxis()  # Match image coordinates
            
            # Draw nails
            axes[plot_idx].plot(self.nail_positions[:, 0], self.nail_positions[:, 1], 
                        'ko', markersize=3)
            
            # Draw first part of sequence
            max_lines_to_show = min(500, len(self.nail_sequence) - 1)
            for i in range(max_lines_to_show):
                start_nail = self.nail_sequence[i]
                end_nail = self.nail_sequence[i + 1]
                start_pos = self.nail_positions[start_nail]
                end_pos = self.nail_positions[end_nail]
                axes[plot_idx].plot([start_pos[0], end_pos[0]], 
                           [start_pos[1], end_pos[1]], 
                           'b-', alpha=0.3, linewidth=0.5)
            
            axes[plot_idx].set_title(f'String Path (first {max_lines_to_show} lines)')
            axes[plot_idx].axis('off')
        else:
            axes[plot_idx].imshow(result_display, cmap='gray')
            axes[plot_idx].set_title('String Art (Alternative View)')
            axes[plot_idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def save_nail_sequence(self, filename: str):
        """Save the nail sequence to a file."""
        with open(filename, 'w') as f:
            for nail in self.nail_sequence:
                f.write(f"{nail}\n")
        print(f"Nail sequence saved to {filename}")
    
    def load_nail_sequence(self, filename: str):
        """Load a nail sequence from a file."""
        self.nail_sequence = []
        with open(filename, 'r') as f:
            for line in f:
                self.nail_sequence.append(int(line.strip()))
        
        # Reconstruct working image
        if self.use_gpu:
            self.working_image_gpu.zero_()
        else:
            self.working_image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        for i in range(len(self.nail_sequence) - 1):
            self._draw_line(self.nail_sequence[i], self.nail_sequence[i + 1], intensity=0.3)
        
        # Sync GPU to CPU if needed
        if self.use_gpu:
            self.working_image = self.working_image_gpu.cpu().numpy()
        
        # Apply smoothing
        self.working_image = gaussian_filter(self.working_image, sigma=0.5)
        
        print(f"Nail sequence loaded from {filename}: {len(self.nail_sequence)} nails")
