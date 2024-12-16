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

    # Step 2: Group by 'sample' column and count the occurrences
    if 'sample' not in df.columns:
        print("Error: Column 'sample' not found in the CSV file.")
        sys.exit(1)

    sample_counts = df['sample'].value_counts()

    # Step 3: Plot the distribution of the total number of repeats using a horizontal bar plot
    plt.figure(figsize=(10, 6))
    repeat_counts = sample_counts.value_counts().sort_index()  # Frequency of repeats

    plt.barh(repeat_counts.index, repeat_counts.values, color='skyblue', edgecolor='black')

    # Customize plot
    plt.title("Distribution of sample tiles in dataset")
    plt.ylabel("Number of tiles")
    plt.xlabel("Frequency")
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Save the figure
    output_file = "sample_repeats_distribution_no_white.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graph saved as '{output_file}'.")

if __name__ == "__main__":
    main()
