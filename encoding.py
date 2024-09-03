import os
import subprocess
import numpy as np


def picturize_data(file_path, resolution):
    """
    Generator that converts a file into a sequence of images.

    This function takes a file path as input, reads the file in chunks of MxN bytes
    (with M and N being the width and height of the image, and consequently, video)
    and converts each chunk into a 3-channel image (RGB). The first image is a special metadata frame and
    contains the file size and extension encoded in the first two rows. The rest of the images
    contain the actual file data. The generator yields the images one by one,
    and is then fed into a video encoding pipeline.

    Args:
        file_path (str): The path to the file to be converted.
        resolution (str): The video resolution provided by the end user.

    Yields:
        numpy.ndarray: A 3D numpy array representing the image.

    """

    def create_metadata_frame():
        """
        Creates a metadata frame with the size and extension of the input file

        This function takes the size and extension of the input file, converts them to bytes,
        and encodes them on the first two rows of a (height, width, 1) array.
        The array is then repeated three times to create a 3D array (height, width, 3 RGB channels).
        A red pixel is placed at the end of each row to indicate the end of the data.

        Args:
            None

        Returns:
            numpy.ndarray: A 3D numpy array representing the metadata frame
        """
        size_len = len(str(file_size))
        extension_len = len(file_extension)

        # Convert individual size and extension characters into byte values
        size_as_bytes = np.array([ord(char) for char in str(file_size)])
        extension_as_bytes = np.array([ord(char) for char in file_extension])

        # Encode size on first row, extension on second row
        metadata_arr = np.zeros((height, width, 1), dtype=np.uint8)
        metadata_arr[0, :size_len] = size_as_bytes.reshape(size_len, 1)
        metadata_arr[1, :extension_len] = extension_as_bytes.reshape(extension_len, 1)
        metadata_arr = np.repeat(metadata_arr, 3, axis=2)

        # Place red pixel indicating the end (there's actually a nullbyte at the end of
        # file_size and file_extension, therefore we overwrite the final byte with the red px itself)
        metadata_arr[0, size_len, 0] = 255
        metadata_arr[1, extension_len, 0] = 255
        return metadata_arr


    width, height = list(map(int, resolution.split("x")))
    bytes_per_frame = width * height

    file_size = os.path.getsize(file_path)
    file_extension = file_path.split(".")[-1]

    metadata_arr = create_metadata_frame()
    yield metadata_arr

    with open(file_path, "rb") as f:
        while chunk := f.read(bytes_per_frame):
            # Convert chunk of data into a ndarray, pad with 0s (nullbytes) if necessary
            # then reshape it to the resolution of video and repeat it 3 times (RGB channels)
            byte_arr = np.frombuffer(chunk, dtype=np.uint8)
            byte_arr = np.pad(
                byte_arr,
                pad_width=(0, bytes_per_frame - len(byte_arr)),
                mode="constant",
            )
            byte_arr = np.reshape(byte_arr, (height, width, 1)).repeat(3, axis=2)

            yield byte_arr


def encode_video(
    image_generator, output_directory, output_name, fps, resolution, cores_utilized
):
    """
    Encodes a sequence of images into a video using FFMPEG.

    The function receives a generator of images and stitches them together into a video.
    Resolution and FPS is decided by the end user. For any given random file where there's no
    repeated pattern, it is expected to output a video with 1.5x as large as the original file.
    As resolution grows bigger, size compression gets better, although the difference
    grows exponentially smaller. The smallest size (240p) provides the least compression, but
    it is the fastest to render.


    Args:
        image_generator: A generator yielding 3D numpy arrays representing the images to be encoded.
        output_directory (str): The directory that will store the output file.
        output_name (str): The video file name.
        fps (int): The video FPS.
        resolution (str): The video resolution.
        cores_utilized (int): The number of cores utilized by FFMPEG to encode the video.

    Returns:
        None
    """
    output_path = rf"{output_directory}\{output_name}.mp4"
    # Construct the ffmpeg command
  command = [
      "ffmpeg", 
      "-y", 
      "-f", "rawvideo", 
      "-vcodec", "rawvideo", 
      "-s", resolution, 
      "-pix_fmt", "rgb24", 
      "-r", f"{fps}", 
      "-i", "-", 
      "-c:v", "png", 
      "-threads", f"{cores_utilized}", 
      output_path
  ]

    # Start the ffmpeg process
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    # Write frames to ffmpeg
    for frame in image_generator:
        process.stdin.write(frame)

    # Close pipe and wait for the process to finish
    process.stdin.close()
    process.wait()


if __name__ == "__main__":
    print("For the image_generator, pass the generator from picturize_data(). GUI is coming soon")
