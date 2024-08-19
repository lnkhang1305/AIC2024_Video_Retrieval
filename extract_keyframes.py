import os
import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# Directory containing the input videos
input_videos_dir = 'data'
output_dir = 'keyframes'  # Directory where keyframes will be saved

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to extract keyframes from a single video
def extract_keyframes(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    
    # Create a directory for the video keyframes
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)

    # Create a video manager object
    video_manager = VideoManager([video_path])

    # Create a scene manager object
    scene_manager = SceneManager()

    # Add a content detector (which detects scene changes based on content)
    scene_manager.add_detector(ContentDetector(threshold=30.0))

    # Start the video manager
    video_manager.start()

    # Detect scenes
    scene_manager.detect_scenes(frame_source=video_manager)

    # Get the list of scenes found
    scene_list = scene_manager.get_scene_list()

    print(f"{len(scene_list)} scenes detected in {video_name}.")

    # OpenCV video capture for frame extraction
    cap = cv2.VideoCapture(video_path)

    # Extract first, middle, and last frames
    for i, scene in enumerate(scene_list):
        start_frame, end_frame = scene[0].get_frames(), scene[1].get_frames()

        # Calculate frame numbers for first, middle, and last frames
        first_frame = start_frame
        middle_frame = (start_frame + end_frame) // 2
        last_frame = end_frame - 1  # Last frame before the next scene starts

        # List of frames to extract
        frames_to_extract = [first_frame, middle_frame, last_frame]

        for j, frame_num in enumerate(frames_to_extract):
            # Set video position to the specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            ret, frame = cap.read()
            if ret:
                # Save the frame as an image
                frame_filename = f'{video_output_dir}/scene_{i + 1}_frame_{j + 1}.jpg'
                cv2.imwrite(frame_filename, frame)
            else:
                print(f"Failed to extract frame {frame_num} for scene {i + 1} in {video_name}.")

    # Release the video capture object
    cap.release()

# Process all videos in the directory
for video_file in os.listdir(input_videos_dir):
    video_path = os.path.join(input_videos_dir, video_file)
    if os.path.isfile(video_path) and video_file.lower().endswith(('.mp4', '.mkv', '.avi')):
        extract_keyframes(video_path)
