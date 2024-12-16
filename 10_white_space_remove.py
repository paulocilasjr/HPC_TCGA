import pandas as pd
from PIL import Image

def has_less_than_10_percent_white(image_path, threshold=10):
    """
    Check if the image has less than threshold% white pixels.
    :param image_path: Path to the image file.
    :param threshold: Percentage threshold for white pixels.
    :return: True if white pixel percentage < threshold, False otherwise.
    """
    try:
        # Open the image and convert to RGB
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        total_pixels = width * height

        # Get pixel data
        pixels = img.getdata()

        # Count white pixels (255, 255, 255)
        white_pixel_count = sum(1 for pixel in pixels if pixel == (255, 255, 255))
        white_percentage = (white_pixel_count / total_pixels) * 100

        return white_percentage < threshold
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False

def filter_images_by_white_pixels(input_csv, output_csv):
    """
    Filter rows where images have less than 10% white pixels.
    :param input_csv: Input CSV file with 'image_path' column.
    :param output_csv: Output CSV file to save filtered rows.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(input_csv)

        # Check if 'image_path' column exists
        if 'image_path' not in df.columns:
            raise KeyError("The input CSV must have a column named 'image_path'.")

        # Filter rows where the image has less than 10% white pixels
        filtered_rows = []
        for index, row in df.iterrows():
            image_path = row['image_path']
            if has_less_than_10_percent_white(image_path):
                filtered_rows.append(row)

        # Create a new DataFrame with filtered rows
        filtered_df = pd.DataFrame(filtered_rows)

        # Save to output CSV
        filtered_df.to_csv(output_csv, index=False)
        print(f"Filtered data saved to {output_csv}")
    except Exception as e:
        print(f"Error processing CSV: {e}")

# Define input and output CSV paths
input_csv_path = "er_status_all_data.csv"  # Replace with your input file path
output_csv_path = "er_status_no_white.csv"

# Run the script
filter_images_by_white_pixels(input_csv_path, output_csv_path)

