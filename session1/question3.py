import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, color, morphology
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from skimage.filters import threshold_otsu

def process_and_display_image(image_path, show_image=True, colored_image_path=None):
    """
    Process a binary image to label regions, numerize them, and display their properties.

    Parameters:
        image_path (str): Path to the binary image file.
        show_image (bool): Whether to display the labeled image. Default is True.
        colored_image_path (str): Optional path to the colored image file to overlay numbers. Default is None.
    """
    # 1. Load binary image (black and white)
    image = io.imread(image_path, as_gray=True)

    # Invert if objects are black
    binary = image < 0.5  # Consider pixels < 0.5 as foreground (objects)

    # 2. Label objects
    label_image = label(binary)

    # 3. Get properties
    props = regionprops(label_image)

    # 4. Display and print properties
    print(f"Number of regions: {len(props)}\n")
    for i, region in enumerate(props):
        print(f"Region {i}: area = {region.area}  perimeter = {region.perimeter:.6f}  "
              f"excentricity = {region.eccentricity:.6f}  solidity = {region.solidity:.6f}")

    # 5. Visualize and save labeled regions with numbers
    output_image_path = f"imgs1_output/labeled{image_path.split('/')[-1].split('_')[0][-1]}.png"
    fig, ax = plt.subplots(figsize=(10, 10))  # Increase figure size for better resolution

    if colored_image_path:
        print("Using colored image for visualization")
        # Load the colored image if provided
        colored_image = io.imread(colored_image_path)
        ax.imshow(colored_image)  # Display the colored image directly
    else:
        print("Using grayscale image for visualization")
        ax.imshow(binary, cmap='gray')

    ax.axis('off')

    for region in props:
        # Get the centroid of the region
        y, x = region.centroid
        ax.text(x, y, str(region.label), color='white', fontsize=10, ha='center', va='center')

    # Save the labeled image with higher DPI for better resolution
    fig.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"Labeled image saved to {output_image_path}")

    # Display the image if requested
    if show_image:
        plt.imshow(io.imread(output_image_path))
        plt.title("Numerized Regions")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    # Example usage
    input_image_path = 'imgs1_output/objetos1_bw.png'  # Path to the input binary image
    process_and_display_image(input_image_path, show_image=False, colored_image_path='imgs/objetos1.png')

    input_image_path = 'imgs1_output/objetos2_bw.png'
    process_and_display_image(input_image_path, show_image=False, colored_image_path='imgs/objetos2.png')

    input_image_path = 'imgs1_output/objetos3_bw.png'
    process_and_display_image(input_image_path, show_image=False, colored_image_path='imgs/objetos3.png')