import pandas as pd
import numpy as np
import cv2

# Step 1: Load the CSV file
csv_file = "data.csv"
df = pd.read_csv(csv_file)

# Create an array to store image data
image_data = np.zeros((df.shape[0], 150), dtype=np.uint8)

# Iterate through the DataFrame and fill the image_data array
for i, row in enumerate(df.iterrows()):
    image_data[i] = row[1][1:151].to_numpy(dtype=np.uint8)

# Create a gradient representing the depth
depth = np.arange(df.shape[0], dtype=np.float32)
depth = (depth - depth.min()) / (depth.max() - depth.min())

# Normalize the depth values to a colormap (e.g., COLORMAP_JET)
depth_colormap = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_JET)

# Apply the depth colormap to the image
combined_image = cv2.vconcat([image_data])
combined_image_colored = cv2.applyColorMap(combined_image, cv2.COLORMAP_JET)

# Step 2: Save the combined image
output_filename = "output/combined_image_with_depth.png"
cv2.imwrite(output_filename, combined_image_colored)

print(f"Combined image with depth saved as {output_filename}")
