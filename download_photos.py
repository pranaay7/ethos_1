import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import logging
import cv2
import sys  

import os
import pandas as pd

# Get the arguments passed from the run_script.py
csv_path = sys.argv[1]
npz_folder = sys.argv[2]
output_folder = sys.argv[3]

# Load the CSV file
print(f"Loading CSV file from: {csv_path}")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows from the CSV file.")

# Example processing (add your download logic here)
# Make sure to create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to calculate image blur using variance of Laplacian
def calculate_image_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


# Function to estimate head pose using landmarks
def calculate_head_pose(landmarks):
    # Use specific facial landmarks (e.g., nose and eye centers) to estimate the head pose
    # This is a simple placeholder and should be replaced with a more robust method (e.g., using OpenCV's solvePnP)
    nose_tip = landmarks[30]
    chin = landmarks[8]
    left_eye = landmarks[36]
    right_eye = landmarks[45]
    head_height = euclidean(nose_tip, chin)
    head_width = euclidean(left_eye, right_eye)
    pose_score = head_width / head_height
    return pose_score


# Function to calculate facial expression (mouth openness as a placeholder for expressions)
def calculate_facial_expression(landmarks):
    mouth_landmarks = landmarks[48:60]
    mar = calculate_mouth_aspect_ratio(mouth_landmarks)
    return mar  # Placeholder, lower mar = neutral, higher = expressive


# Functions for face centering, aspect ratios, symmetry etc.
def calculate_face_center(landmarks):
    return np.mean(landmarks, axis=0)


def calculate_eye_aspect_ratio(eye_landmarks):
    v1 = euclidean(eye_landmarks[1], eye_landmarks[5])
    v2 = euclidean(eye_landmarks[2], eye_landmarks[4])
    h = euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear


def calculate_mouth_aspect_ratio(mouth_landmarks):
    v = euclidean(mouth_landmarks[2], mouth_landmarks[6])
    h = euclidean(mouth_landmarks[0], mouth_landmarks[4])
    mar = v / h
    return mar


def calculate_face_symmetry(landmarks):
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    nose_tip = landmarks[30]
    left_mouth = landmarks[48]
    right_mouth = landmarks[54]
    
    eye_diff = abs(left_eye[1] - right_eye[1])
    mouth_diff = abs(left_mouth[1] - right_mouth[1])
    
    return 1 / (1 + eye_diff + mouth_diff)


# Normalize score with safe division by zero handling
def normalize_score(score, min_val, max_val):
    if min_val == max_val:
        return 0.5  # Neutral score if no variation
    return (score - min_val) / (max_val - min_val)


# Scoring function for a frame
def score_frame(frame, landmarks, bbox):
    scores = {}

    # Face centering
    frame_center = np.array([frame.shape[1] / 2, frame.shape[0] / 2])
    face_center = calculate_face_center(landmarks)
    center_distance = euclidean(frame_center, face_center)
    scores['centering'] = 1 / (1 + center_distance) if center_distance != 0 else 0.5

    # Eye aspect ratio
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    left_ear = calculate_eye_aspect_ratio(left_eye)
    right_ear = calculate_eye_aspect_ratio(right_eye)
    if left_ear != 0 and right_ear != 0:
        scores['eye_openness'] = (left_ear + right_ear) / 2
    else:
        scores['eye_openness'] = 0.5  # Neutral score if no variation

    # Face size
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    face_size = face_width * face_height
    size_difference = abs(face_size - average_face_size)
    scores['size'] = 1 / (1 + size_difference) if size_difference != 0 else 0.5

    # Face symmetry
    scores['symmetry'] = calculate_face_symmetry(landmarks)

    # Mouth openness
    mouth_landmarks = landmarks[48:60]
    mar = calculate_mouth_aspect_ratio(mouth_landmarks)
    scores['mouth_closed'] = 1 / (1 + mar) if mar != 0 else 0.5

    # Blur score
    scores['blur'] = calculate_image_blur(frame)

    # Expression score
    scores['expression'] = calculate_facial_expression(landmarks)

    # Head pose score
    scores['head_pose'] = calculate_head_pose(landmarks)

    return scores


# Main video processing function
def process_video(npz_path):
    try:
        video_id = os.path.basename(npz_path).replace('.npz', '')

        # Get the average face size for this video
        global average_face_size
        average_face_size = df.loc[df['videoID'] == video_id, 'averageFaceSize'].values[0]

        # Load the .npz data
        data = np.load(npz_path)
        color_images = data['colorImages']
        bounding_box = data['boundingBox']
        landmarks_2d = data['landmarks2D']

        print(f"Processing {video_id}")
        print("Color Images shape:", color_images.shape)
        print("Bounding Box shape:", bounding_box.shape)
        print("Landmarks 2D shape:", landmarks_2d.shape)

        # Evaluate all frames
        all_scores = []
        for i in range(color_images.shape[3]):
            frame = color_images[:, :, :, i]
            landmarks = landmarks_2d[:, :, i]
            bbox = bounding_box[:, 0, i]
            scores = score_frame(frame, landmarks, bbox)
            all_scores.append(scores)

        # Normalize scores
        for criterion in all_scores[0].keys():
            min_val = min(scores[criterion] for scores in all_scores)
            max_val = max(scores[criterion] for scores in all_scores)
            for scores in all_scores:
                scores[criterion] = normalize_score(scores[criterion], min_val, max_val)

        # Calculate final scores
        final_scores = []
        for scores in all_scores:
            final_score = (
                scores['centering'] * 0.2 +
                scores['eye_openness'] * 0.1 +
                scores['size'] * 0.1 +
                scores['symmetry'] * 0.1 +
                scores['mouth_closed'] * 0.1 +
                scores['blur'] * 0.15 +
                scores['expression'] * 0.15 +
                scores['head_pose'] * 0.1
            )
            final_scores.append(final_score)

        # Find the best frame
        best_frame_index = np.argmax(final_scores)
        best_frame = color_images[:, :, :, best_frame_index]

        # Save the best frame
        plt.figure(figsize=(10, 8))
        plt.imshow(best_frame)
        plt.title(f'Best Frame for {video_id} (Frame {best_frame_index})')
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, f'{video_id}_best_frame.png'), dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Processed {video_id}")
    except Exception as e:
        logging.error(f"Error processing {npz_path}: {str(e)}")


# Process all .npz files
npz_files = [os.path.join(npz_folder, f) for f in os.listdir(npz_folder) if f.endswith('.npz')]

for npz_file in npz_files:
    process_video(npz_file)

print("Processing complete. Check the output folder for results.")
