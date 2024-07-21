import os
import urllib

import requests
from mjpeg import open_mjpeg_stream, read_mjpeg_frame
from mjpeg.client import Buffer


class CameraClient:
    def __init__(self, stream_url: str, resolution_url: str, save_dir: str):
        self._stream_url = stream_url
        self._resolution_url = resolution_url
        self._save_dir = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.set_resolution_to_highest()

    def set_resolution_to_highest(self):
        response = requests.get(self._resolution_url)
        response.raise_for_status()

    def _save_buffer(self, suffix: str, frame_buffer: bytearray):
        image_filename = f'image_{suffix}.jpg'
        image_path = os.path.join(self._save_dir, image_filename)
        with open(image_path, 'wb') as image_file:
            image_file.write(frame_buffer)
        print(f'Saved {image_filename}')

    def run(self, max_frames: int):
        with urllib.request.urlopen(self._stream_url) as stream:
            boundary = open_mjpeg_stream(stream)

            frame_index = 0
            # 128KB
            frame_buffer = Buffer(128 * 1024)

            while frame_index < max_frames:
                timestamp, frame_length = read_mjpeg_frame(stream, boundary, frame_buffer.data, frame_buffer.length)
                frame_actual_buffer = frame_buffer.data[:frame_length]
                if frame_length == 0:
                    # Stream is complete
                    break
                self._save_buffer(str(frame_index), frame_actual_buffer)
                self._save_buffer('latest', frame_actual_buffer)
                frame_index += 1


def main():
    stream_url = "http://192.168.1.150:81/stream"
    resolution_url = "http://192.168.1.150:80/control?var=framesize&val=13"
    save_dir = './saved_images'

    client = CameraClient(stream_url, resolution_url, save_dir)
    client.run(10000)


if __name__ == '__main__':
    main()
