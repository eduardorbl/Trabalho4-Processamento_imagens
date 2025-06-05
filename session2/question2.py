import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from question1 import to_rgb, center_image_on_canvas

def bilinear_interpolation(img, x_rel, y_rel):
    """
    Perform bilinear interpolation on an image.

    Parameters:
    -----------
    img : numpy.ndarray
        Input image array (height x width x channels).
    x_rel : numpy.ndarray
        Relative x-coordinates for interpolation.
    y_rel : numpy.ndarray
        Relative y-coordinates for interpolation.

    Returns:
    --------
    numpy.ndarray
        Interpolated image array.
    """
    h, w = img.shape[:2]

    # Calculate the integer coordinates of the top-left corner
    x0 = np.floor(x_rel).astype(int)
    y0 = np.floor(y_rel).astype(int)

    # Calculate the integer coordinates of the bottom-right corner
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    # Clip the top-left coordinates to ensure they are within bounds
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    # Calculate the fractional parts for interpolation
    dx = (x_rel - x0)[..., np.newaxis]
    dy = (y_rel - y0)[..., np.newaxis]

    # Perform bilinear interpolation
    top = (1 - dx) * img[y0, x0] + dx * img[y0, x1]
    bottom = (1 - dx) * img[y1, x0] + dx * img[y1, x1]
    return ((1 - dy) * top + dy * bottom).astype(np.uint8)

def process_images(input_folder, output_folder, scale=2.25, display=False):
    """
    Process images by resizing and applying bilinear interpolation.

    Parameters:
    -----------
    input_folder : str
        Path to the folder containing input images.
    output_folder : str
        Path to the folder where output images will be saved.
    scale : float, optional
        Scaling factor for bilinear interpolation (default is 2.25).
    display : bool, optional
        Whether to display the processed images (default is False).

    Returns:
    --------
    None
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        image = io.imread(input_path)
        image_rgb = to_rgb(image)

        height_in, width_in = image_rgb.shape[:2]
        height_out = int(height_in * scale)
        width_out = int(width_in * scale)

        # Generate output grid coordinates
        y_out, x_out = np.meshgrid(np.arange(height_out), np.arange(width_out), indexing='ij')

        # Calculate relative coordinates
        y_rel = (y_out - height_out / 2) / scale + height_in / 2
        x_rel = (x_out - width_out / 2) / scale + width_in / 2

        # Apply bilinear interpolation
        scaled_image = bilinear_interpolation(image_rgb, x_rel, y_rel)

        resized_img = (
            center_image_on_canvas(image, height_out, width_out)
            if scale > 1
            else to_rgb(image)
        )

        H = max(resized_img.shape[0], scaled_image.shape[0])
        W = max(resized_img.shape[1], scaled_image.shape[1])

        padded_resized = center_image_on_canvas(resized_img, H, W)
        padded_scaled = center_image_on_canvas(scaled_image, H, W)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        axes[0].imshow(padded_resized)
        axes[0].set_title("Resized Original")
        axes[1].imshow(padded_scaled)
        axes[1].set_title(f"Bilinear Scaled (Scale {scale}x)")

        for ax in axes:
            ax.set_xticks(np.arange(0, W, 50))
            ax.set_yticks(np.arange(0, H, 50))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(color='black', linestyle=':', linewidth=0.5)
            ax.axis('on')

        grid_output_path = os.path.join(output_folder, f"grid_bilinear_{image_file}")
        plt.savefig(grid_output_path)
        if display:
            plt.show()
        else:
            plt.close()

# Usage:
if __name__ == "__main__":
    input_folder = "imgs"
    output_folder = "imgs2_output"
    scale = 0.5
    display = False
    process_images(input_folder=input_folder, output_folder=output_folder, scale=scale, display=display)
