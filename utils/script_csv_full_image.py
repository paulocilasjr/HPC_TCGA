import pandas as pd

# Input and output file paths
input_csv = "er_status_all_data.csv"  # Replace with your input CSV file
output_csv = "er_status_full_image.csv"  # Replace with the desired output file name

# Load the CSV file, assuming it uses commas as separators and the first row contains the header
df = pd.read_csv(input_csv, sep=",", header=0)

# Debugging step: Print the loaded DataFrame to ensure it's correct
print("Loaded DataFrame:")
print(df.head())

# Keep only the first occurrence of each sample
df_unique = df.drop_duplicates(subset="sample", keep="first")

# Debugging step: Print the DataFrame after dropping duplicates
print("DataFrame after dropping duplicates:")
print(df_unique.head())

# Apply transformation to the image_path column
df_unique["image_path"] = df_unique["image_path"].apply(lambda x: "images/" + x.split("/")[0] + ".svs")

# Debugging step: Print the DataFrame after applying the transformation
print("DataFrame after applying the transformation:")
print(df_unique.head())

# Save the modified data to a new CSV file
df_unique.to_csv(output_csv, index=False, header=False)

# Confirmation message
print(f"Processed file saved as {output_csv}")
