import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from question1 import to_rgb, center_image_on_canvas

def lagrange_weights(d):
    """
    Compute Lagrange interpolation weights for a given input array.

    This function calculates the weights for Lagrange interpolation based on 
    the input array `d`. The weights are computed for four points and returned 
    as a 3D array.

    Args:
        d (numpy.ndarray): A 3D array of shape (H, W, 1) representing the input 
            distances for which the weights are calculated.

    Returns:
        numpy.ndarray: A 3D array of shape (H, W, 4) containing the Lagrange 
        interpolation weights for four points.

    Notes:
        - The input array `d` is expected to have a single channel (last dimension 
          of size 1).
        - The output array contains weights for four points, calculated using 
          Lagrange interpolation formulas.
    """
    # d: (H, W, 1)
    W = np.zeros((d.shape[0], d.shape[1], 4), dtype=np.float32)
    W[:, :, 0] = -d[:, :, 0]*(d[:, :, 0]-1)*(d[:, :, 0]-2)/6
    W[:, :, 1] =  (d[:, :, 0]+1)*(d[:, :, 0]-1)*(d[:, :, 0]-2)/2
    W[:, :, 2] = -d[:, :, 0]*(d[:, :, 0]+1)*(d[:, :, 0]-2)/2
    W[:, :, 3] =  d[:, :, 0]*(d[:, :, 0]+1)*(d[:, :, 0]-1)/6
    return W

def lagrange_interpolation(image_rgb, x_rel, y_rel):
    """
    Perform Lagrange interpolation on an RGB image.

    This function applies Lagrange interpolation to resize or transform an RGB image
    based on relative coordinates provided by `x_rel` and `y_rel`. It computes interpolated
    pixel values using Lagrange weights in both horizontal and vertical directions.

    Args:
        image_rgb (numpy.ndarray): Input RGB image as a 3D NumPy array of shape (height, width, 3).
        x_rel (numpy.ndarray): Relative x-coordinates for interpolation, as a 2D NumPy array.
        y_rel (numpy.ndarray): Relative y-coordinates for interpolation, as a 2D NumPy array.

    Returns:
        numpy.ndarray: Interpolated RGB image as a 3D NumPy array of shape matching `x_rel` and `y_rel`,
        with pixel values clipped to the range [0, 255] and converted to `uint8` type.

    Notes:
        - The function assumes `x_rel` and `y_rel` are normalized coordinates within the bounds of the input image.
        - The interpolation uses a 4x4 neighborhood of pixels for each target coordinate.
        - The `lagrange_weights` function must be defined elsewhere to compute interpolation weights.
    """
    height_in, width_in = image_rgb.shape[:2]
    x0 = np.floor(x_rel).astype(int)
    y0 = np.floor(y_rel).astype(int)
    dx = x_rel - x0
    dy = y_rel - y0
    dx = dx[:, :, np.newaxis]
    dy = dy[:, :, np.newaxis]

    Wx = lagrange_weights(dx)
    Wy = lagrange_weights(dy)

    result = np.zeros((*x0.shape, 3), dtype=np.float32)

    for n in range(4):  # vertical
        Ln = np.zeros_like(result)
        for m in range(4):  # horizontal
            xm = np.clip(x0 + m - 1, 0, width_in - 1)
            yn = np.clip(y0 + n - 1, 0, height_in - 1)
            pixel = image_rgb[yn, xm].astype(np.float32)
            Ln += Wx[:, :, m:m+1] * pixel
        result += Wy[:, :, n:n+1] * Ln

    return np.clip(result, 0, 255).astype(np.uint8)

def process_image_lagrange(image_path, output_folder, scale=2.25, display=False):
    """
    Processes a single image by scaling it using Lagrange interpolation 
    and saves the result in the output folder. Optionally displays the comparison 
    between the resized original and the scaled image.

    Args:
        image_path (str): Path to the input image file.
        output_folder (str): Path to the folder where the processed image will be saved.
        scale (float, optional): Scaling factor for the image. Defaults to 2.25.
        display (bool, optional): If True, displays the comparison plot. Defaults to False.

    Returns:
        None

    Notes:
        - Supported image formats include PNG, JPG, JPEG, BMP, and TIFF.
        - The function creates a grid comparison plot showing the resized original 
          and the scaled image using Lagrange interpolation.
        - The output image is saved with a filename prefixed by "grid_lagrange_".
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

    scaled_image = lagrange_interpolation(image_rgb, x_rel, y_rel)

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
    axes[1].set_title(f"Lagrange Scaled (Scale {scale}x)")

    for ax in axes:
        ax.set_xticks(np.arange(0, W, 50))
        ax.set_yticks(np.arange(0, H, 50))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='black', linestyle=':', linewidth=0.5)
        ax.axis('on')

    image_name = os.path.basename(image_path)
    grid_output_path = os.path.join(output_folder, f"grid_lagrange_{image_name}")
    plt.savefig(grid_output_path)
    if display:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    image_path = "imgs/monalisa.png" 
    output_folder = "imgs2_output"
    scale = 3.2
    display = True
    process_image_lagrange(image_path=image_path, output_folder=output_folder, scale=scale, display=display)