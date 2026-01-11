# String Art AI Generator

A Python application that converts any image into string art by placing nails in a circle and determining the optimal path for a single string to wrap around them, creating a visual representation of the original image.

## Features

- **Image to String Art Conversion**: Automatically converts any image into a nail sequence for string art
- **Circle of Nails Simulation**: Places nails evenly in a circle around the image
- **Optimized Path Finding**: Uses Radon transform analysis combined with an iterative algorithm to find the best sequence of nails to approximate the image
- **Visualization**: Includes static and animated visualization of the string art creation process
- **Nail Sequence Export**: Saves the ordered list of nails for physical reproduction

## Installation

1. Install Python 3.7 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from string_art import StringArtGenerator
import matplotlib.pyplot as plt

# Create generator
generator = StringArtGenerator(
    image_path='your_image.jpg',
    num_nails=300,      # Number of nails in circle
    image_size=500      # Image processing size
)

# Generate string art (3000 iterations)
nail_sequence = generator.generate_string_art(
    num_iterations=3000,
    min_distance=10,    # Minimum nail jump distance
    max_distance=None   # Maximum nail jump distance
)

# Visualize result
generator.visualize()
plt.show()

# Save nail sequence
generator.save_nail_sequence('nail_sequence.txt')
```

### Quick Start Example

Run the example script:
```bash
python example.py
```

Make sure to update `input_image.jpg` in `example.py` with your image path.

### Animated Simulation

Watch the string art being created in real-time:
```bash
python visualize_simulation.py
```

This will show an animated visualization of the string wrapping around the nails.

## How It Works

1. **Image Processing**: 
   - Converts image to grayscale
   - Resizes to specified dimensions
   - Inverts brightness (dark areas need more string)

2. **Radon Transform Analysis**:
   - Computes the Radon transform of the image
   - The Radon transform identifies important lines and structures by computing line integrals at various angles
   - This provides a mathematical foundation for identifying which nail connections are most important

3. **Nail Placement**:
   - Places nails evenly in a circle around the image
   - Default: 300 nails, adjustable

4. **Path Generation**:
   - Starts at a random nail
   - Iteratively selects the next nail based on:
     - **Radon transform score**: How important this line is according to the Radon transform
     - **Image difference**: How well the line matches the remaining difference between target and current image
     - Distance constraints (avoids adjacent nails)
   - Accumulates string density to build up the image

5. **Output**:
   - Ordered list of nail indices
   - Visual representation of the string art
   - Can be saved and used for physical string art creation

## Parameters

### StringArtGenerator
- `image_path`: Path to input image (JPG, PNG, etc.)
- `num_nails`: Number of nails in circle (default: 300)
- `image_size`: Processing size in pixels (default: 500)

### generate_string_art()
- `num_iterations`: Number of string wraps (default: 3000)
  - More iterations = better detail but longer processing
- `min_distance`: Minimum nail distance to jump (default: 10)
  - Prevents wrapping around adjacent nails
- `max_distance`: Maximum nail distance to jump (default: None)
  - Limits maximum jump distance for better control

## Output Format

The nail sequence is saved as a text file with one nail index per line:
```
0
45
120
...
```

Each number represents the nail index (0 to num_nails-1) in the order the string should wrap around them.

## Tips

- **Higher nail count**: More nails (400-500) give better detail but slower processing
- **More iterations**: 3000-5000 iterations provide good results
- **Image preparation**: High contrast images work best
- **Distance tuning**: Adjust `min_distance` based on desired effect

## Requirements

- Python 3.7+
- numpy
- Pillow (PIL)
- matplotlib
- scipy
- scikit-image

See `requirements.txt` for specific versions.

Install with:
```bash
pip install -r requirements.txt
```
