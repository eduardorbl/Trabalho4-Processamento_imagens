# Reexecutando após reset

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from question1 import to_rgb, center_image_on_canvas

def P(t):
    """
    Applies a threshold function to the input array or value.

    This function returns the input `t` if it is greater than 0, 
    otherwise it returns 0. It is commonly used in image processing 
    or mathematical operations where negative values are replaced 
    with zero.

    Parameters:
        t (array-like or scalar): Input value or array to apply the threshold function.

    Returns:
        array-like or scalar: The result of applying the threshold function to `t`.
    """
    return np.where(t > 0, t, 0)

def R(s):
    """
    Compute the result of a specific mathematical operation involving the function P.

    This function calculates a weighted sum of cubic powers of the function P evaluated
    at different offsets of the input `s`. The formula used is:

        (1 / 6) * (P(s + 2)**3 - 4 * P(s + 1)**3 + 6 * P(s)**3 - 4 * P(s - 1)**3)

    Args:
        s (float or int): The input value to be used in the computation.

    Returns:
        float: The result of the mathematical operation.
    """
    return (1 / 6) * (
        P(s + 2)**3 - 4 * P(s + 1)**3 + 6 * P(s)**3 - 4 * P(s - 1)**3
    )

def bicubic_interpolation(image_rgb, x_rel, y_rel):
    """
    Perform bicubic interpolation on an RGB image.

    Bicubic interpolation is used to resize or transform an image by calculating 
    the weighted average of surrounding pixels based on cubic polynomials.

    Parameters:
        image_rgb (numpy.ndarray): Input RGB image as a 3D NumPy array 
                                   with shape (height, width, 3).
        x_rel (numpy.ndarray): Relative x-coordinates for interpolation, 
                               typically a 2D array of floating-point values.
        y_rel (numpy.ndarray): Relative y-coordinates for interpolation, 
                               typically a 2D array of floating-point values.

    Returns:
        numpy.ndarray: Interpolated RGB image as a 3D NumPy array with 
                       the same shape as the input coordinates (x_rel, y_rel) 
                       and 3 color channels. Pixel values are clipped to 
                       the range [0, 255] and returned as unsigned 8-bit integers.

    Notes:
        - This function assumes the input image is in RGB format.
        - The interpolation weights are calculated using the R function, 
          which should implement the cubic interpolation kernel.
        - The function handles boundary conditions by clipping coordinates 
          to valid ranges within the input image dimensions.
    """
    height_in, width_in = image_rgb.shape[:2]
    x0 = np.floor(x_rel).astype(int)
    y0 = np.floor(y_rel).astype(int)
    dx = x_rel - x0
    dy = y_rel - y0
    dx = dx[:, :, np.newaxis]
    dy = dy[:, :, np.newaxis]

    result = np.zeros((*x0.shape, 3), dtype=np.float32)

    for m in range(-1, 3):
        for n in range(-1, 3):
            x_m = np.clip(x0 + m, 0, width_in - 1)
            y_n = np.clip(y0 + n, 0, height_in - 1)
            weight = R(m - dx) * R(dy - n)
            pixel = image_rgb[y_n, x_m].astype(np.float32)
            result += weight * pixel

    return np.clip(result, 0, 255).astype(np.uint8)

def process_image_bicubic(image_path, output_folder, scale=2.25, display=False):
    """
    Processes a single image by scaling it using bicubic interpolation 
    and saves the result in the output folder. Optionally displays the comparison 
    between the resized original and bicubic scaled images.

    Args:
        image_path (str): Path to the input image file.
        output_folder (str): Path to the folder where processed images will be saved.
        scale (float, optional): Scaling factor for bicubic interpolation. Default is 2.25.
        display (bool, optional): If True, displays the comparison plot. Default is False.

    Returns:
        None

    Notes:
        - Supported image formats include PNG, JPG, JPEG, BMP, and TIFF.
        - The function creates a side-by-side comparison of the resized original image 
          and the bicubic scaled image, saving the result as a grid image in the output folder.
        - If `display` is set to True, the comparison plot is shown interactively; otherwise, 
          it is closed after saving.
    """
    os.makedirs(output_folder, exist_ok=True)
    image = io.imread(image_path)
    image_rgb = to_rgb(image)

    height_in, width_in = image_rgb.shape[:2]
    height_out = int(height_in * scale)
    width_out = int(width_in * scale)

    y_out, x_out = np.meshgrid(np.arange(height_out), np.arange(width_out), indexing='ij')
    y_rel = (y_out - height_out / 2) / scale + height_in / 2
    x_rel = (x_out - width_out / 2) / scale + width_in / 2

    scaled_image = bicubic_interpolation(image_rgb, x_rel, y_rel)

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
    axes[1].set_title(f"Bicubic Scaled (Scale {scale}x)")

    for ax in axes:
        ax.set_xticks(np.arange(0, W, 50))
        ax.set_yticks(np.arange(0, H, 50))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='black', linestyle=':', linewidth=0.5)
        ax.axis('on')

    grid_output_path = os.path.join(output_folder, f"grid_bicubic_{os.path.basename(image_path)}")
    plt.savefig(grid_output_path)
    if display:
        plt.show()
    else:
        plt.close()

# Uso:
if __name__ == "__main__":
    image_path = "imgs/monalisa.png"
    output_folder = "imgs2_output"
    scale = 3.2
    display = True
    process_image_bicubic(image_path=image_path, output_folder=output_folder, scale=scale, display=display)