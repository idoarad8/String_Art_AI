"""
Interactive visualization that simulates the string art creation process.
Shows the circle of nails and animates the string wrapping around them.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from string_art import StringArtGenerator


class StringArtSimulation:
    def __init__(self, generator: StringArtGenerator):
        """
        Initialize the simulation with a StringArtGenerator instance.
        
        Args:
            generator: A StringArtGenerator that has already generated a nail sequence
        """
        self.generator = generator
        self.nail_sequence = generator.nail_sequence
        self.nail_positions = generator.nail_positions
        
        # Set up the figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Left plot: Circle and nails with string path
        self.ax1.set_xlim(0, generator.image_size)
        self.ax1.set_ylim(0, generator.image_size)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('String Art Simulation')
        self.ax1.invert_yaxis()  # Match image coordinates
        
        # Right plot: Resulting image
        self.ax2.set_title('Resulting Image')
        self.ax2.axis('off')
        
        # Draw all nails
        self.ax1.plot(self.nail_positions[:, 0], self.nail_positions[:, 1], 
                     'ko', markersize=4, label='Nails')
        
        # Initialize line and image
        self.line, = self.ax1.plot([], [], 'b-', alpha=0.5, linewidth=0.8)
        self.working_image = np.zeros((generator.image_size, generator.image_size), 
                                     dtype=np.float32)
        self.image_plot = self.ax2.imshow(self.working_image, cmap='gray', 
                                         vmin=0, vmax=1)
        
        self.current_line_index = 0
        
    def animate(self, frame):
        """Animation function that draws one line at a time."""
        if self.current_line_index >= len(self.nail_sequence) - 1:
            return self.line, self.image_plot
        
        # Get current and next nail
        start_nail = self.nail_sequence[self.current_line_index]
        end_nail = self.nail_sequence[self.current_line_index + 1]
        
        start_pos = self.nail_positions[start_nail]
        end_pos = self.nail_positions[end_nail]
        
        # Update line path
        x_coords = [start_pos[0], end_pos[0]]
        y_coords = [start_pos[1], end_pos[1]]
        
        # Extend existing path
        if self.current_line_index == 0:
            self.line_path_x = [start_pos[0]]
            self.line_path_y = [start_pos[1]]
        
        self.line_path_x.append(end_pos[0])
        self.line_path_y.append(end_pos[1])
        
        # Keep only last 100 points for performance
        if len(self.line_path_x) > 100:
            self.line_path_x = self.line_path_x[-100:]
            self.line_path_y = self.line_path_y[-100:]
        
        self.line.set_data(self.line_path_x, self.line_path_y)
        
        # Draw line on working image
        self.generator._draw_line(start_nail, end_nail, intensity=0.3)
        result = np.clip(self.generator.working_image, 0, 1)
        self.image_plot.set_array(result)
        
        # Update title
        progress = (self.current_line_index + 1) / len(self.nail_sequence) * 100
        self.ax1.set_title(f'String Art Simulation - Progress: {progress:.1f}%')
        
        self.current_line_index += 1
        
        return self.line, self.image_plot
    
    def run(self, interval: int = 10):
        """
        Run the animation.
        
        Args:
            interval: Delay between frames in milliseconds
        """
        # Limit frames for reasonable animation time
        max_frames = min(2000, len(self.nail_sequence) - 1)
        
        anim = animation.FuncAnimation(
            self.fig, 
            self.animate, 
            frames=max_frames,
            interval=interval,
            blit=False,
            repeat=False
        )
        
        plt.tight_layout()
        return anim


def simulate_string_art(image_path: str, num_nails: int = 300, 
                       num_iterations: int = 2000, animate: bool = True):
    """
    Create and simulate string art generation.
    
    Args:
        image_path: Path to input image
        num_nails: Number of nails in circle
        num_iterations: Number of string wraps
        animate: If True, show animation; if False, show static result
    """
    # Generate string art
    generator = StringArtGenerator(
        image_path=image_path,
        num_nails=num_nails,
        image_size=500
    )
    
    generator.generate_string_art(num_iterations=num_iterations)
    
    if animate:
        # Show animation
        simulation = StringArtSimulation(generator)
        anim = simulation.run(interval=10)
        plt.show()
    else:
        # Show static result
        generator.visualize()
        plt.show()
    
    return generator


if __name__ == "__main__":
    # Example usage
    image_path = 'input_image.jpg'  # Change this to your image path
    
    # Run with animation
    generator = simulate_string_art(
        image_path=image_path,
        num_nails=300,
        num_iterations=2000,
        animate=True
    )
    
    # Save sequence
    generator.save_nail_sequence('nail_sequence.txt')
