import pandas as pd
import json
import os
current_directory = os.getcwd()
print("Current Directory:", current_directory)

# Load JSON data from a file
with open('./malaria/test.json', 'r') as file:
    json_data_list = json.load(file)

# Extracting relevant information
# path = json_data["image"]["pathname"]
# objects = json_data.get("objects", [])

# Creating a DataFrame
df = pd.DataFrame()
for json_data in json_data_list:
    # Extracting relevant information
    path = json_data["image"]["pathname"]
    objects = json_data.get("objects", [])
    for obj in objects:
        min_r = obj["bounding_box"]["minimum"]["r"]
        min_c = obj["bounding_box"]["minimum"]["c"]
        max_r = obj["bounding_box"]["maximum"]["r"]
        max_c = obj["bounding_box"]["maximum"]["c"]
        category = obj["category"]

        df = pd.concat([df, pd.DataFrame({
            "path": [path],
            "min_r": [min_r],
            "min_c": [min_c],
            "max_r": [max_r],
            "max_c": [max_c],
            "category": [category],
            "ref": [category]  # Assuming ref should be the same as category
        })])

# Save DataFrame to CSV
df.to_csv("./malaria/test.csv", index=False)

# Display the resulting DataFrame
print(df)
