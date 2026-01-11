"""
String Art Generator using Radon Transform
Converts an image into string art by placing nails in a circle and 
using the Radon transform to determine optimal string paths.
"""

import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from skimage.transform import radon
import math


class StringArtGenerator:
    def __init__(self, image_path: str, num_nails: int = 300, image_size: int = 500):
        """
        Initialize the String Art Generator.
        
        Args:
            image_path: Path to the input image
            num_nails: Number of nails to place in the circle
            image_size: Size of the processed image (will be resized)
        """
        self.num_nails = num_nails
        self.image_size = image_size
        
        # Load and process image
        self.image = self._load_image(image_path)
        self.target_image = self._process_image(self.image)
        
        # Calculate nail positions in a circle
        self.nail_positions = self._calculate_nail_positions()
        
        # Initialize working image (will be drawn on)
        self.working_image = np.zeros((image_size, image_size), dtype=np.float32)
        
        # Store the nail sequence
        self.nail_sequence = []
        
        # Precompute line integrals for efficiency
        self._precompute_radon_transform()
        
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
        
        # Prepare image for Radon transform (needs to be square)
        # The Radon transform computes line integrals at different angles
        theta = np.linspace(0., 180., self.num_nails, endpoint=False)
        
        # Compute Radon transform
        # This gives us line integrals (projections) at different angles
        sinogram = radon(self.target_image, theta=theta, circle=True)
        
        self.sinogram = sinogram
        self.theta = theta
        
        # Normalize sinogram to [0, 1]
        if sinogram.max() > 0:
            self.sinogram_normalized = (sinogram - sinogram.min()) / (sinogram.max() - sinogram.min())
        else:
            self.sinogram_normalized = sinogram
        
        print(f"  Radon transform computed: {sinogram.shape}")
    
    def _nail_angle_to_radon_index(self, nail_idx: int) -> int:
        """Convert nail index to corresponding Radon transform angle index."""
        # Nail angle in degrees (0 to 360)
        nail_angle_deg = (nail_idx / self.num_nails) * 360.0
        
        # Radon transform uses angles 0 to 180
        # We need to map nail angles to Radon angles
        # For a line from nail A to nail B, the line angle is (A + B) / 2
        # But we need the actual line direction
        
        # Map to 0-180 range
        if nail_angle_deg >= 180:
            nail_angle_deg -= 180
        
        # Find closest theta in Radon transform
        idx = int(nail_angle_deg / 180.0 * self.num_nails)
        return idx % self.num_nails
    
    def _get_line_radon_score(self, nail_a: int, nail_b: int) -> float:
        """
        Get the score for a line between two nails based on Radon transform.
        Higher score means this line is important in the image.
        """
        # Calculate the angle of the line from nail_a to nail_b
        pos_a = self.nail_positions[nail_a]
        pos_b = self.nail_positions[nail_b]
        
        # Line direction vector
        dx = pos_b[0] - pos_a[0]
        dy = pos_b[1] - pos_a[1]
        
        # Angle in degrees (0 to 180 for Radon transform)
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad) % 180.0
        
        # Find closest theta index in Radon transform
        theta_idx = int(angle_deg / 180.0 * self.num_nails) % self.num_nails
        
        # Calculate distance from center for the line
        center = self.image_size / 2
        # Distance along the line's perpendicular
        mid_point = ((pos_a[0] + pos_b[0]) / 2, (pos_a[1] + pos_b[1]) / 2)
        center_vec = np.array([mid_point[0] - center, mid_point[1] - center])
        
        # Project onto perpendicular to line
        line_dir = np.array([dx, dy])
        line_dir = line_dir / (np.linalg.norm(line_dir) + 1e-10)
        perp_dir = np.array([-line_dir[1], line_dir[0]])
        
        distance = np.dot(center_vec, perp_dir)
        
        # Map distance to sinogram column index
        # Sinogram has columns corresponding to different offsets
        sinogram_cols = self.sinogram.shape[1]
        col_idx = int((distance + self.image_size / 2) / self.image_size * sinogram_cols)
        col_idx = max(0, min(sinogram_cols - 1, col_idx))
        
        # Get score from normalized sinogram
        if 0 <= theta_idx < self.num_nails and 0 <= col_idx < sinogram_cols:
            score = self.sinogram_normalized[theta_idx, col_idx]
            return float(score)
        return 0.0
    
    def _draw_line(self, start_nail: int, end_nail: int, intensity: float = 1.0):
        """Draw a line between two nails on the working image."""
        start_pos = self.nail_positions[start_nail]
        end_pos = self.nail_positions[end_nail]
        
        x0, y0 = int(start_pos[0]), int(start_pos[1])
        x1, y1 = int(end_pos[0]), int(end_pos[1])
        
        # Clip to image bounds
        x0 = max(0, min(self.image_size - 1, x0))
        y0 = max(0, min(self.image_size - 1, y0))
        x1 = max(0, min(self.image_size - 1, x1))
        y1 = max(0, min(self.image_size - 1, y1))
        
        # Draw line
        steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            x = int(x0 + t * (x1 - x0))
            y = int(y0 + t * (y1 - y0))
            
            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                self.working_image[y, x] += intensity
    
    def _calculate_line_benefit(self, current_nail: int, candidate_nail: int) -> float:
        """
        Calculate how beneficial it would be to draw a line from current_nail
        to candidate_nail. Uses both Radon transform and current image difference.
        """
        # Get Radon-based score (how important this line is)
        radon_score = self._get_line_radon_score(current_nail, candidate_nail)
        
        # Also consider how well it matches the current difference
        start_pos = self.nail_positions[current_nail]
        end_pos = self.nail_positions[candidate_nail]
        
        x0, y0 = int(start_pos[0]), int(start_pos[1])
        x1, y1 = int(end_pos[0]), int(end_pos[1])
        
        x0 = max(0, min(self.image_size - 1, x0))
        y0 = max(0, min(self.image_size - 1, y0))
        x1 = max(0, min(self.image_size - 1, x1))
        y1 = max(0, min(self.image_size - 1, y1))
        
        steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
        difference_score = 0
        pixels_checked = 0
        
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            x = int(x0 + t * (x1 - x0))
            y = int(y0 + t * (y1 - y0))
            
            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                pixels_checked += 1
                target_intensity = self.target_image[y, x]
                current_intensity = min(1.0, self.working_image[y, x])
                difference = max(0, target_intensity - current_intensity)
                difference_score += difference
        
        if pixels_checked == 0:
            return -1000
        
        avg_difference = difference_score / pixels_checked
        
        # Combine Radon score and difference score
        # Weight Radon more early on, difference more later
        combined_score = 0.7 * radon_score + 0.3 * avg_difference
        
        # Small penalty for very long lines
        line_length = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        length_penalty = line_length * 0.0005
        
        return combined_score - length_penalty
    
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
        
        self.working_image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        self.nail_sequence = []
        
        # Start at a random nail
        current_nail = np.random.randint(0, self.num_nails)
        self.nail_sequence.append(current_nail)
        
        print(f"Generating string art with {num_iterations} iterations...")
        print(f"  Using Radon transform guidance...")
        
        for iteration in range(num_iterations):
            if (iteration + 1) % 500 == 0:
                print(f"  Progress: {iteration + 1}/{num_iterations}")
            
            best_nail = None
            best_score = -float('inf')
            
            # Try candidates - prioritize nails with good Radon scores
            num_candidates = min(150, self.num_nails)
            
            # Create candidate list with some randomization
            all_nails = np.arange(self.num_nails)
            candidates = np.random.choice(all_nails, num_candidates, replace=False)
            
            for candidate in candidates:
                # Check distance constraints
                distance = min(abs(candidate - current_nail), 
                             self.num_nails - abs(candidate - current_nail))
                
                if distance < min_distance or distance > max_distance:
                    continue
                
                # Calculate benefit using Radon transform
                score = self._calculate_line_benefit(current_nail, candidate)
                
                if score > best_score:
                    best_score = score
                    best_nail = candidate
            
            # If no valid nail found, relax constraints
            if best_nail is None:
                for candidate in range(self.num_nails):
                    distance = min(abs(candidate - current_nail), 
                                 self.num_nails - abs(candidate - current_nail))
                    if distance >= min_distance:
                        score = self._calculate_line_benefit(current_nail, candidate)
                        if score > best_score:
                            best_score = score
                            best_nail = candidate
            
            # Draw the line
            if best_nail is not None:
                intensity = 0.25  # Lower intensity for smoother buildup
                self._draw_line(current_nail, best_nail, intensity)
                self.nail_sequence.append(best_nail)
                current_nail = best_nail
            else:
                # Fallback: move to next valid nail
                current_nail = (current_nail + min_distance) % self.num_nails
                self.nail_sequence.append(current_nail)
        
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
        
        # Original processed image
        axes[plot_idx].imshow(self.target_image, cmap='gray')
        axes[plot_idx].set_title('Target Image (Inverted)')
        axes[plot_idx].axis('off')
        plot_idx += 1
        
        # String art result
        result = np.clip(self.working_image, 0, 1)
        axes[plot_idx].imshow(result, cmap='gray')
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
            axes[plot_idx].imshow(result, cmap='gray')
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
        self.working_image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        for i in range(len(self.nail_sequence) - 1):
            self._draw_line(self.nail_sequence[i], self.nail_sequence[i + 1], intensity=0.25)
        
        print(f"Nail sequence loaded from {filename}: {len(self.nail_sequence)} nails")
