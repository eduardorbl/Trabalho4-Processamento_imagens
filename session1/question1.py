import cv2
import matplotlib.pyplot as plt
import os

def process_and_display_image(input_image_path, show_on_screen=False):
    """
    Processes an input image by converting it to black-and-white,
    and optionally displays the images on the screen. Saves the processed image
    to a predefined output path.

    Args:
        input_image_path (str): Path to the input image file.
        show_on_screen (bool): Whether to display the images on the screen. Default is False.
    """
    # Read colored image
    colored_image = cv2.imread(input_image_path)  # RGB

    # Convert from BGR to RGB for correct visualization with matplotlib
    rgb_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)

    # Convert grayscale to black and white using a threshold
    _, bw_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

    # Generate output path based on input file name
    archive_original_name = input_image_path.split('/')[-1].split('.')[0]
    output_directory = "imgs1_output"
    output_image_path = f"{output_directory}/{archive_original_name}_bw.png"

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Suppress libpng warnings
    cv2.imwrite(output_image_path, bw_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # Display the images if requested
    if show_on_screen:
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_image)
        plt.title('Colored Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(bw_image, cmap='gray')
        plt.title('Monochrome Image')
        plt.axis('off')

        plt.show()

# Example usage
if __name__ == "__main__":
    input_image_path = 'imgs/objetos1.png'  # Path to the input image
    process_and_display_image(input_image_path, show_on_screen=False)
    input_image_path = 'imgs/objetos2.png' 
    process_and_display_image(input_image_path, show_on_screen=False)
    input_image_path = 'imgs/objetos3.png'
    process_and_display_image(input_image_path, show_on_screen=False)