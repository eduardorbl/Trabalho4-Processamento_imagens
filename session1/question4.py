import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import regionprops, label

def analyze_image_regions(image_path):
    """
    Analyze regions in a binary image and classify them based on their area.

    Parameters:
        image_path (str): Path to the binary image file.

    Returns:
        dict: A dictionary containing counts of small, medium, and large regions.
    """
    # Load binary image (black objects)
    img = io.imread(image_path, as_gray=True)
    binary = img < 0.5  # Invert: black objects become True

    # Label regions
    labeled = label(binary)
    props = regionprops(labeled)

    # Collect properties into a DataFrame
    df = pd.DataFrame([{
        "Region": i,
        "Area": r.area
    } for i, r in enumerate(props)])

    # Classification
    small_regions = df[df["Area"] < 1500]
    medium_regions = df[(df["Area"] >= 1500) & (df["Area"] < 3000)]
    large_regions = df[df["Area"] >= 3000]

    print(f"Number of small regions: {len(small_regions)}")
    print(f"Number of medium regions: {len(medium_regions)}")
    print(f"Number of large regions: {len(large_regions)}")

    # Manual binning according to classification ranges
    bins = [0, 1500, 3000, max(df["Area"].max(), 4000)]
    counts, _ = np.histogram(df["Area"], bins=bins)

    # Bar plot
    plt.bar(["Small", "Medium", "Large"], counts, color='blue')
    plt.title('Histogram of Object Areas by Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Regions')
    plt.tight_layout()
    plt.show()

    return

if __name__ == "__main__":
    # Example usage
    analyze_image_regions('imgs1_output/objetos1_bw.png')
    analyze_image_regions('imgs1_output/objetos2_bw.png')
    analyze_image_regions('imgs1_output/objetos3_bw.png')

