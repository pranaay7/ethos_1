import os
import subprocess
import sys

method = 'gfpgan'  # Define the method you want to use
upscale = 2  # Example upscale factor, change as necessary
# Get the arguments passed from the run_script.py
output_folder = sys.argv[1]
image_dir = os.path.join(os.path.dirname(output_folder), 'photos_out_resized')
output_dir = os.path.join(os.path.dirname(output_folder), 'photos_out_enhanced')

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)
# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get list of all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Loop through each image and run the command
for image_file in image_files:
    input_image_path = os.path.join(image_dir, image_file)
    
    # Split the file name and extension
    file_name, file_extension = os.path.splitext(image_file)
    
    # Create output image path with '_enhanced' added to the original file name
    output_image_path = os.path.join(output_dir, f'{file_name}_enhanced{file_extension}')
    
    # Run the command for each image
    command = f'python main.py --method {method} --image_path "{input_image_path}" --output_path "{output_image_path}" --upscale {upscale}'
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f'Processed {image_file} successfully as {output_image_path}.')
    except subprocess.CalledProcessError as e:
        print(f'Error processing {image_file}: {e}')
