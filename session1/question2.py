import numpy as np
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
from PIL import Image
import os

def extract_object_contours(image_path, structure_size=3, visualize=False, output_folder='imgs1_output'):
    """
    Extracts the contours of objects in a binary image using internal gradient.

    Parameters:
        image_path (str): Path to the binary image file.
        structure_size (int): Size of the square structuring element for erosion (default is 3).
        visualize (bool): If True, displays the resulting contours (default is False).
        output_folder (str): Folder to save the output image (default is 'imgs1_output').

    Returns:
        None
    """

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the binary image
    img = Image.open(image_path).convert('L')
    binary = np.array(img) > 128  # True for object (white)

    # Invert: now black objects are True
    binary = ~binary

    # Structuring element
    structure = np.ones((structure_size, structure_size), dtype=bool)

    # Internal gradient: border = object - erosion
    eroded = binary_erosion(binary, structure=structure)
    borda = binary & ~eroded

    # Invert the color of the contours
    borda = ~borda

    try:
        # Save the output image with the modified name
        base_name = os.path.basename(image_path).replace('objetos', 'contours').replace('_bw', '')
    except Exception:
        base_name = 'contours.png'
    output_path = os.path.join(output_folder, base_name)
    plt.imsave(output_path, borda, cmap='gray')

    # Visualize if requested
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.imshow(borda, cmap='gray')
        plt.title('Contornos dos Objetos', fontsize=16)
        plt.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.show()

if __name__ == "__main__":
    # Example usage
    input_image_path = 'imgs1_output/objetos1_bw.png'  # Path to the input binary image
    extract_object_contours(input_image_path, visualize=False)
    
    input_image_path = 'imgs1_output/objetos2_bw.png'
    extract_object_contours(input_image_path, visualize=True)
    
    input_image_path = 'imgs1_output/objetos3_bw.png'
    extract_object_contours(input_image_path, visualize=False)