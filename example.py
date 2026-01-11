"""
Example usage of the String Art Generator
"""

from string_art import StringArtGenerator
import matplotlib.pyplot as plt

def main():
    # Example: Create string art from an image
    # Replace 'your_image.jpg' with the path to your image
    image_path = './Test_images/Whiter.jpg'  # Change this to your image path
    
    # Create generator with 300 nails and 500x500 image size
    generator = StringArtGenerator(
        image_path=image_path,
        num_nails=300,      # Number of nails in the circle
        image_size=500      # Size of processed image
    )
    
    # Generate the string art
    # num_iterations: how many string wraps to perform
    # min_distance: minimum nail distance (avoid adjacent nails)
    # max_distance: maximum nail distance (None = no limit)
    nail_sequence = generator.generate_string_art(
        num_iterations=3000,
        min_distance=10,
        max_distance=None
    )

    
    # Visualize the result (set show_radon=True to see the Radon transform)
    generator.visualize(show_nails=True, show_sequence=True, show_radon=True)
    plt.show()
    
    # Save the nail sequence
    generator.save_nail_sequence('nail_sequence.txt')
    
    print(f"\nGenerated {len(nail_sequence)} nail positions.")
    print("Nail sequence saved to 'nail_sequence.txt'")
    print("\nFirst 20 nails in sequence:")
    print(nail_sequence[:20])

if __name__ == "__main__":
    main()
