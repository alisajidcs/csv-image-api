import pandas as pd
import numpy as np
import sqlite3
import cv2
from flask import Flask, request, Response

# Step 1: Load the CSV file
csv_file = "data.csv"
df = pd.read_csv(csv_file)

# Step 2: Resize the images (width to 150 columns)
df.iloc[:, 1:151] = df.iloc[:, 1:201].values

# Step 3: Store the resized images in a SQLite database
conn = sqlite3.connect("image_data.db")
cursor = conn.cursor()
cursor.execute('CREATE TABLE images (depth INTEGER, data BLOB)')

for index, row in df.iterrows():
    depth = row['depth']
    image_data = row[1:151].to_numpy(dtype=np.uint8)
    cursor.execute("INSERT INTO images (depth, data) VALUES (?, ?)", (depth, image_data.tobytes()))

conn.commit()
conn.close()

# Step 4: Create a Flask API
app = Flask(__name__)

@app.route('/get_image', methods=['GET'])
def get_image():
    depth_min = int(request.args.get('depth_min'))
    depth_max = int(request.args.get('depth_max'))
    
    conn = sqlite3.connect("image_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT depth, data FROM images WHERE depth BETWEEN ? AND ?", (depth_min, depth_max))
    results = cursor.fetchall()
    conn.close()
    
    processed_images = []

    for depth, image_data in results:
        # Convert the image data from bytes to a NumPy array
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        
        # Reshape the array to the desired image width
        image_array = image_array.reshape(1, 150)
        
        # Apply a custom color map or any other image processing here
        # Example: You can use OpenCV to apply color mapping
        image_array_colored = cv2.applyColorMap(image_array, cv2.COLORMAP_JET)
        
        # Append the processed image to the list
        processed_images.append(image_array_colored)
    
    # Return the processed images as a JSON response
    return Response(response=processed_images, status=200, mimetype="image/jpeg")

if __name__ == '__main__':
    app.run(debug=True)
