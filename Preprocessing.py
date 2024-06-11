import os
import cv2


def crop_video(input_video_path, output_video_path, crop_rect):
    """
    Crop each frame of a video to a specified rectangle.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the cropped video file.
        crop_rect (tuple): The cropping rectangle (x, y, width, height).
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error opening video file {input_video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Define the codec and create VideoWriter object
    x, y, w, h = crop_rect
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame
        cropped_frame = frame[y:y + h, x:x + w]

        # Write the cropped frame to the output video
        out.write(cropped_frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Cropped video saved to {output_video_path}")


def extract_frames(video_path, output_dir, step=10):
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

    cap.release()
    print(f"Frames extracted from {video_path}")


def preprocessing_old(directory_path, crop_rect):
    """
    Preprocess videos in a directory by cropping and then extracting frames.

    Args:
        directory_path (str): Path to the directory containing the video files.
        crop_rect (tuple): Rectangle coordinates (x, y, width, height) to crop.
    """
    if not os.path.isdir(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    output_dir = os.path.join(directory_path, "samples")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(directory_path):
        if filename.endswith(".mp4"):
            video_path = os.path.join(directory_path, filename)
            cropped_video_path = os.path.join(directory_path, f"cropped_{filename}")

            # Crop the video
            # crop_video(video_path, cropped_video_path, crop_rect)

            video_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            os.makedirs(video_output_dir, exist_ok=True)
            extract_frames(cropped_video_path, video_output_dir)
            print(f"Processed {filename}")


def preprocessing(directory_path):
    """
    Preprocess the newest video in a directory by extracting frames.

    Args:
        directory_path (str): Path to the directory containing the video files.
    """
    if not os.path.isdir(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    output_dir = os.path.join(directory_path, "samples")
    os.makedirs(output_dir, exist_ok=True)

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

    if newest_file is not None:
        cropped_video_path = os.path.join(directory_path, f"cropped_{newest_file}")

        video_output_dir = os.path.join(output_dir, os.path.splitext(newest_file)[0])
        os.makedirs(video_output_dir, exist_ok=True)
        extract_frames(cropped_video_path, video_output_dir)
        print(f"Processed {newest_file}")
    else:
        print("No .mp4 files found in the directory.")


# Example usage
crop_rectangle = (350, 0, 1072, 880)  # (x, y, width, height)
preprocessing("Scenes/scene1")
