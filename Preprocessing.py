import os
import cv2


def extract_frames(video_path, output_dir, step=30):
    """
    Extract frames from a video at specified intervals.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
        step (int): Interval in seconds between frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    step_frames = int(step * fps)

    current_frame = 0
    frame_number = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        current_frame += step_frames
        frame_number += 1

    frame1, frame2 = f"frame_{frame_number -1 :04d}.jpg", f"frame_{frame_number -2:04d}.jpg"

    cap.release()
    print(f"Frames extracted from {video_path}")
    return frame1, frame2


def preprocessing(directory_path):
    """
    Preprocess the newest video in a directory by extracting frames.

    Args:
        directory_path (str): Path to the directory containing the video files.
    """
    if not os.path.isdir(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    # Find the newest .mp4 file
    newest_file = None
    newest_time = None

    for filename in os.listdir(directory_path):
        if filename.endswith(".mp4"):
            file_path = os.path.join(directory_path, filename)
            file_time = os.path.getmtime(file_path)
            if newest_time is None or file_time > newest_time:
                newest_time = file_time
                newest_file = filename

    video_output_dir,  frame1, frame2 = "", "", ""
    if newest_file is not None:
        video_path = os.path.join(directory_path, newest_file)

        video_output_dir = os.path.join(directory_path, os.path.splitext(newest_file)[0])
        os.makedirs(video_output_dir, exist_ok=True)
        frame1, frame2 = extract_frames(video_path, video_output_dir)
        print(f"Processed {newest_file}")
    else:
        print("No .mp4 files found in the directory.")

    return video_output_dir + "\\" + frame1, video_output_dir + "\\" + frame2