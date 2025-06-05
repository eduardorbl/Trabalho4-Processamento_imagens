import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def to_rgb(img):
    """
    Convert a grayscale image to RGB format.

    Parameters:
    -----------
    img : numpy.ndarray
        Input image array.

    Returns:
    --------
    numpy.ndarray
        RGB image array.
    """
    return img if img.ndim == 3 else np.repeat(img[:, :, np.newaxis], 3, axis=2)

def center_image_on_canvas(img, H, W):
    """
    Centers an image on a blank canvas of specified dimensions.

    Parameters:
        img (numpy.ndarray): The input image to be centered. Can be grayscale or RGB.
        H (int): The height of the canvas.
        W (int): The width of the canvas.

    Returns:
        numpy.ndarray: A canvas of dimensions (H, W, 3) with the input image centered.
                       The canvas is filled with white (255) in areas not occupied by the image.

    Notes:
        - The input image is converted to RGB format if it is not already.
        - If the canvas dimensions are smaller than the image dimensions, the image will be cropped.
    """
    img_rgb = to_rgb(img)
    h, w = img_rgb.shape[:2]
    offset_y = (H - h) // 2
    offset_x = (W - w) // 2

    canvas = np.full((H, W, 3), 255, dtype=img_rgb.dtype)
    canvas[offset_y:offset_y+h, offset_x:offset_x+w] = img_rgb
    return canvas

def nearest_interpolation(img, x_rel, y_rel):
    """
    Performs nearest neighbor interpolation for the given relative coordinates.

    Parameters:
        img (ndarray): Input image.
        x_rel (ndarray): Relative x-coordinates.
        y_rel (ndarray): Relative y-coordinates.

    Returns:
        ndarray: Interpolated pixel values.
    """
    h, w = img.shape[:2]
    x_nn = np.round(x_rel).astype(int)
    y_nn = np.round(y_rel).astype(int)
    x_nn = np.clip(x_nn, 0, w - 1)
    y_nn = np.clip(y_nn, 0, h - 1)
    return img[y_nn, x_nn]

def process_images(image_paths, output_folder, scale=2.25, display=False):
    """
    Reads images from the provided file paths, scales them using nearest neighbor interpolation,
    saves the scaled images and their comparison grid to the output folder, and optionally displays them.

    Parameters:
        image_paths (list): List of image file paths.
        output_folder (str): Folder to save scaled images.
        scale (float): Scaling factor.
        display (bool): Whether to display the images.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_paths:
        # Read the image
        image = io.imread(image_path)

        # Scale the image
        height_in, width_in = image.shape[:2]
        height_out = int(height_in * scale)
        width_out = int(width_in * scale)

        center_in = np.array([height_in / 2, width_in / 2])
        center_out = np.array([height_out / 2, width_out / 2])

        image_rgb = to_rgb(image)
        scaled_image = np.full((height_out, width_out, 3), 255, dtype=image_rgb.dtype)

        # Create a grid of output coordinates
        y_out, x_out = np.meshgrid(np.arange(height_out), np.arange(width_out), indexing='ij')

        # Calculate relative coordinates in the input image
        y_rel = (y_out - center_out[0]) / scale + center_in[0]
        x_rel = (x_out - center_out[1]) / scale + center_in[1]

        # Use nearest_interpolation function
        valid_mask = (x_rel >= 0) & (x_rel < width_in) & (y_rel >= 0) & (y_rel < height_in)
        scaled_image[y_out[valid_mask], x_out[valid_mask], :] = nearest_interpolation(image_rgb, x_rel[valid_mask], y_rel[valid_mask])

        resized_img = (
            center_image_on_canvas(image, height_out, width_out)
            if scale > 1
            else image.copy()
        )

        # Pad both images to match largest height/width
        H = max(resized_img.shape[0], scaled_image.shape[0])
        W = max(resized_img.shape[1], scaled_image.shape[1])

        padded_resized = center_image_on_canvas(resized_img, H, W)
        padded_scaled = center_image_on_canvas(scaled_image, H, W)

        # Create grid comparison figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        axes[0].imshow(padded_resized)
        axes[0].set_title("Resized Original")
        axes[0].axis('on')
        axes[1].imshow(padded_scaled)
        axes[1].set_title(f"Scaled (Scale {scale}x)")
        axes[1].axis('on')

        for ax in axes:
            ax.set_xticks(np.arange(0, W, 50))
            ax.set_yticks(np.arange(0, H, 50))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(color='black', linestyle=':', linewidth=0.5)

        grid_output_path = os.path.join(output_folder, f"grid_{os.path.basename(image_path)}")
        plt.savefig(grid_output_path)
        if display:
            plt.show()
        else:
            plt.close()

# Example usage:
if __name__ == "__main__":
    image_paths = [
        "imgs/monalisa.png"
    ]
    output_folder = "imgs2_output"
    scale = 0.8
    display = True
    process_images(image_paths=image_paths, output_folder=output_folder, scale=scale, display=display)