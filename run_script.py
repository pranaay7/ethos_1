import os
import subprocess

# Set the path for the NPZ folder
npz_folder = '/Users/pranay/Desktop/vs code/ethos/final project/test'

# Set the corresponding output folder
output_folder = '/Users/pranay/Desktop/vs code/ethos/final project/photos_out'

# Define the CSV path
csv_path = '/Users/pranay/Desktop/vs code/ethos/youtube_faces_with_keypoints_full.csv'

# Print the base folder
base_folder = os.path.dirname(npz_folder)
print(f"Base folder set to: {base_folder}")

# Run download_photos.py
print("Running download_photos.py...")
subprocess.run(['python', 'download_photos.py', csv_path, npz_folder, output_folder], check=True)

# Run resize.py
print("Running resize.py...")
subprocess.run(['python', 'resize.py', output_folder], check=True)

# Run automate.py
print("Running automate.py...")
subprocess.run(['python', 'automate.py', output_folder], check=True)

print("Processing complete. Check the output folder for results.")
