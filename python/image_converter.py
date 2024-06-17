import os
import cv2 as cv
import argparse
import glob

# Finds all images in the given directory.
def find_images_in_directory(directory_path):

    image_extensions = ["jpg", "jpeg", "png", "bmp", "tif", "tiff","pnm"]
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(f"{directory_path}/*.{ext}"))
    return images

# Converts an image or all images in a directory to the specified format.
def convert_image(input_image_path, output_format):

    if os.path.isdir(input_image_path):
        # If the input path is a directory, convert all images within it.
        images = find_images_in_directory(input_image_path)
        for img_path in images:
            convert_single_image(img_path, output_format)
    else:
        # Convert a single image.
        convert_single_image(input_image_path, output_format)

# Converts a single image to the specified format.
def convert_single_image(input_image_path, output_format):
    img = cv.imread(input_image_path)
    if img is None:
        print(f"Unable to read the image from {input_image_path}")
        return
    base_name = os.path.splitext(input_image_path)[0]
    output_image_path = f"{base_name}.{output_format}"
    cv.imwrite(output_image_path, img)
    print(f"Image converted and saved as {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an image or all images in a directory to a specified format.")
    parser.add_argument("input_path", type=str, help="The path of the image or directory")
    parser.add_argument("output_format", type=str, help="The desired format for the image")
    args = parser.parse_args()

    convert_image(args.input_path, args.output_format)
