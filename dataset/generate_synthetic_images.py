import numpy as np
import cv2
import os
import shutil

# DATASET PARAMETERS

num_images = 2 # Number of images to generate
resolution = (100,100) # Images width and height
output_dir = "HPC/dataset/test1/" # IMages directory
ground_truth_filename = "gt.csv"
lines_per_image = 1 # Number of lines per images

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


def generate_synthetic_image_and_ground_truth(width, height, lines):
    base_background = np.zeros((height, width, 3), dtype=np.uint8)
    ground_truth = []
    
    for _ in range(lines):
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
        cv2.line(base_background, (x1, y1), (x2, y2), (255, 255, 255), 1)
        rho, theta, theta_deg = calculate_hough_parameters(x1, y1, x2, y2, width, height)
        intersections = calculate_line_intersections(x1, y1, x2, y2, width, height)
        ix1, iy1 = intersections[0]
        ix2, iy2 = intersections[1]
        irho, itheta, itheta_deg = calculate_hough_parameters(ix1, iy1, ix2, iy2, width, height)
        ground_truth.append((x1, y1, x2, y2, rho, theta, theta_deg, lines, ix1, iy1, ix2, iy2, irho, itheta, itheta_deg))
        
            
    return base_background, ground_truth

def find_csv_file(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if len(csv_files) != 1:
        raise FileNotFoundError("No CSV file found or multiple CSV files present. Please inspect the directory.")
    return os.path.join(directory, csv_files[0])

def ensure_directory_and_csv_file(output_dir):
    if os.path.exists(output_dir):
        return find_csv_file(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
        return os.path.join(output_dir, ground_truth_filename)

def get_next_image_index(directory):
    image_files = [f for f in os.listdir(directory) if f.startswith('image_') and f.endswith('.pnm')]
    if not image_files:
        return 0
    max_index = max(int(f.split('_')[1].split('.')[0]) for f in image_files)
    return max_index + 1

if __name__ == "__main__":

    gt_path = ensure_directory_and_csv_file(output_dir)

    print('Path:', os.path.exists(gt_path))
    mode = 'w' if not os.path.exists(gt_path) else 'a'
    next_image_index = get_next_image_index(output_dir)
    
    with open(gt_path, mode) as gt_file:
        if mode == 'w':
            gt_file.write("image_name,x1,y1,x2,y2,rho,theta_rad,theta_deg,lines,ix1,iy1,ix2,iy2,irho,itheta,itheta_deg\n")
        
        for i in range(next_image_index, next_image_index + num_images):
            image, gt = generate_synthetic_image_and_ground_truth(resolution[0], resolution[1], lines_per_image)
            image_name = f"image_{i}.pnm"
            image_path = os.path.join(output_dir, image_name)
            cv2.imwrite(image_path, image)
            
            for line in gt:
                print('Writin a line for ', image_name, line)
                gt_file.write(f"{image_name},{','.join(map(str, line))}\n")