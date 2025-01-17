import pandas as pd
import os

# Load the Excel file
df = pd.read_excel('Book1.xlsx')

# Assuming the filenames are in a column named 'FileName'
file_names = df['FileName'].tolist()

# Set to store base names from rows with '.png' extension
base_names_set = set()

# Iterate over filenames to collect base names of .png files
for file in file_names:
    base_name, extension = os.path.splitext(file)
    if extension.lower() == '.png':  # Check for .png extension
        base_names_set.add(base_name)

# List to store the final filtered files
final_files = []

# Iterate over filenames to check for .svs files and compare base names
for file in file_names:
    base_name, extension = os.path.splitext(file)
    if extension.lower() == '.svs' and base_name not in base_names_set:
        final_files.append(file)

# Print the final list of filenames that meet the condition
for file in final_files:
    print(file)

# Optionally, you can also print the count of files that were added to final_files
print(f"Number of files added: {len(final_files)}")
