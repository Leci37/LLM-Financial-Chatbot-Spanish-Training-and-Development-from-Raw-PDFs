import json

# File paths for JSON files
json_files = [
    'processed/_1_w1.json',
    'processed/_1_w2.json',
    'processed/_1_w3.json',
    'processed/_1_w4.json',
    'processed/_1_w5.json'
]

# Output JSON file
output_file = 'processed/_1_combined.json'

# Initialize combined structure
combined_data = {
    "sections": [],
    "tables": []
}

# Process each file and log counts
for file_path in json_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Log the file name
            print(f"Processing file: {file_path}")

            # Count and log sections
            if "sections" in data:
                section_count = len(data["sections"])
                combined_data["sections"].extend(data["sections"])
                print(f"  - Sections added: {section_count}")
            else:
                print("  - No sections found.")

            # Count and log tables
            if "tables" in data:
                table_count = len(data["tables"])
                combined_data["tables"].extend(data["tables"])
                print(f"  - Tables added: {table_count}")
            else:
                print("  - No tables found.")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Save the combined JSON file
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    print(f"\nCombined JSON file saved as {output_file}")
except Exception as e:
    print(f"Error saving combined file: {e}")
