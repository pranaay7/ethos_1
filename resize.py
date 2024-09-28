import os
from PIL import Image
import sys

# Define the source and destination folders

# Get the arguments passed from the run_script.py
output_folder = sys.argv[1]
destination_folder = os.path.join(os.path.dirname(output_folder), 'photos_out_resized')

# Make sure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)



# Iterate through all files in the source folder
for filename in os.listdir(output_folder):
    if filename.endswith('.png'):
        # Open an image file
        img_path = os.path.join(output_folder, filename)
        with Image.open(img_path) as img:
            # Check if image needs resizing
            if img.width > 2048 or img.height > 2048:
                # Calculate the new size while maintaining aspect ratio
                img.thumbnail((2048, 2048))

            # Save the resized image to the destination folder
            img.save(os.path.join(destination_folder, filename))

print("Resizing complete! Images saved to:", destination_folder)
