import numpy as np
import cv2
import os
import shutil

# DATASET PARAMETERS

num_images = 1 # Size of the dataset
size = (10000,10000) # Images width and height
output_dir = "HPC/dataset/images/synthetic_25m/" # Dataset directory
max_lines_per_image = 20 # Number of lines per images in range [1, max_lines_per_image]

# This function calculates the Hough transform parameters for a line.
def calculate_line_intersections(x1, y1, x2, y2, width, height):
    points = []

    # Adjust width and height to be max indexable values
    width -= 1
    height -= 1

    # Calculate line parameters
    if x2 != x1:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    else:
        # Line is vertical
        return ((x1, 0), (x1, height))

    # Check if the slope is zero to avoid division by zero
    if slope == 0:
        if 0 <= y1 <= height:
            return ((0, y1), (width, y1))

    # Intersection with left border (x=0)
    y_at_x0 = intercept
    if 0 <= y_at_x0 <= height:
        points.append((0, int(y_at_x0)))

    # Intersection with right border (x=width)
    y_at_xmax = slope * width + intercept
    if 0 <= y_at_xmax <= height:
        points.append((width, int(y_at_xmax)))

    # Intersection with bottom border (y=height)
    x_at_ymax = (height - intercept) / slope
    if 0 <= x_at_ymax <= width:
        points.append((int(x_at_ymax), height))

    # Intersection with top border (y=0)
    x_at_y0 = -intercept / slope
    if 0 <= x_at_y0 <= width:
        points.append((int(x_at_y0), 0))

    # Ensuring only two intersection points are found
    if len(points) == 2:
        return tuple(points[0]), tuple(points[1])
    else:
        points = sorted(points, key=lambda p: (p[0] ** 2 + p[1] ** 2))
        return tuple(points[0]), tuple(points[-1])


def calculate_hough_parameters(x1, y1, x2, y2, width, height):
    centerX, centerY = width / 2, height / 2
    
    # Calculate the midpoint of the line segment
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    # Calculate the angle using the line's endpoints
    theta_radians = np.arctan2(y2 - y1, x2 - x1)
    if theta_radians < 0:
        theta_radians += 2 * np.pi  # Normalize theta to be within 0 to 2*pi
    
    # Convert theta from radians to degrees
    theta_degrees = theta_radians * (180 / np.pi)

    # Calculate rho using the midpoint
    cos_theta = np.cos(theta_radians)
    sin_theta = np.sin(theta_radians)
    rho = (mid_x - centerX) * cos_theta + (mid_y - centerY) * sin_theta
    
    return rho, theta_radians, theta_degrees


# This function generates a synthetic image with lines and corresponding ground truth data.
def generate_synthetic_image_and_ground_truth(width, height, max_lines=10):

    # Generate black background
    base_background = np.zeros((height, width, 3), dtype=np.uint8)
    ground_truth = []
    line_count = max_lines #np.random.randint(1, max_lines + 1)
    
    for _ in range(line_count):
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
        cv2.line(base_background, (x1, y1), (x2, y2), (255, 255, 255), 1)
        rho, theta, theta_deg = calculate_hough_parameters(x1, y1, x2, y2, width, height)
        intersections = calculate_line_intersections(x1, y1, x2, y2, width, height)
        ix1, iy1 = intersections[0]
        ix2, iy2 = intersections[1]
        irho, itheta, itheta_deg = calculate_hough_parameters(ix1, iy1, ix2, iy2, width, height)

        ground_truth.append((x1, y1, x2, y2, rho, theta, theta_deg, line_count, ix1, iy1, ix2, iy2, irho, itheta, itheta_deg))
    
    return base_background, ground_truth

def create_or_empty_directory(path):
    # Check if the directory exists
    if os.path.exists(path):
        # Remove the directory and all its contents
        shutil.rmtree(path)

    # Create the directory
    os.makedirs(path, exist_ok=True)
    print(f"Created directory: {path}")

if __name__ == "__main__":


    create_or_empty_directory(output_dir)
    gt_path = os.path.join(output_dir, "ground_truth.csv")

    with open(gt_path, 'w') as gt_file:

        # Ground truth CSV file with the coordinates of each line of each image.
        gt_file.write("image_name,x1,y1,x2,y2,rho,theta_rad,theta_deg,lines,ix1,iy1,ix2,iy2,irho,itheta,itheta_deg\n")  # Writing header for CSV file

        for i in range(num_images):
            image, gt = generate_synthetic_image_and_ground_truth(size[0],size[1], max_lines_per_image)
            image_name = f"image_{i}.pnm"
            image_path = os.path.join(output_dir, image_name)
            cv2.imwrite(image_path, image)
            
            # Append ground truth data for each image to the CSV file
            for line in gt:
                gt_file.write(f"{image_name},{line[0]},{line[1]},{line[2]},{line[3]},{line[4]:.2f},{line[5]:.2f},{line[6]},{line[7]},{line[8]},{line[9]},{line[10]},{line[11]},{line[12]},{line[13]},{line[14]}\n")
