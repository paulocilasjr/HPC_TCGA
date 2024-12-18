import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    # Step 1: Read the CSV file
    input_file = 'er_status_no_white.csv'  # Modify this path if needed
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    # Step 2: Group by 'sample' column and count occurrences
    if 'sample' not in df.columns:
        print("Error: Column 'sample' not found in the CSV file.")
        sys.exit(1)

    sample_counts = df['sample'].value_counts()

    # Step 3: Create a scatter plot for each data point
    plt.figure(figsize=(14, 8))  # Set the size of the figure
    x = list(range(1, len(sample_counts) + 1))  # X-axis: Sample indices
    y = sample_counts.values  # Y-axis: Count of occurrences per sample

    plt.scatter(x, y, color='blue', alpha=0.6, edgecolors='black')  # Scatter plot with dots

    # Customize plot
    plt.title("Sample Tiles Distribution in Dataset")
    plt.xlabel("Samples (Ordered by Count)")
    plt.ylabel("Number of Tiles per Sample")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([])  # Remove x-axis ticks since sample names are not shown

    # Save the figure
    output_file = "sample_repeats_scatter_no_white.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graph saved as '{output_file}'.")

if __name__ == "__main__":
    main()
