import numpy as np
from perlin_noise import PerlinNoise
from PIL import Image, ImageFilter

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import svgwrite

import numpy as np
import svgwrite

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def reduce_image(image, factor=10):
    """
    Reduce the size of the image by taking every factor-th pixel in x and y directions.

    Parameters:
    - image: NumPy array representing the input image.
    - factor: The reduction factor.

    Returns:
    - Reduced image as a NumPy array.
    """
    return image[::factor, ::factor]


def generate_isometric_svg(image, svg_filename, max_height_factor=0.1, divide=10):
    image = reduce_image(image, factor=divide)
    height, width = image.shape
    scale_factor = max_height_factor / np.max(image)

    dwg = svgwrite.Drawing(svg_filename, profile='tiny')

    for i in range(height):
        for j in range(width):
            x = (j - i) * np.sqrt(2) / 2
            y = (i + j) * np.sqrt(2) / 4
            z = image[i, j] * scale_factor

            # Draw vertical line (in y direction)
            dwg.add(dwg.line(start=(x, y), end=(x, y - z), stroke="black"))

            # Draw horizontal line (in x direction)
            dwg.add(dwg.line(start=(x, y), end=(x + np.sqrt(2) / 2, y + np.sqrt(2) / 4), stroke="black"))

    dwg.save()


def generate_3d_obj(image, obj_filename, scale_factor=1.0):
    height, width = image.shape
    y, x = np.mgrid[0:height, 0:width]
    vertices = np.column_stack([x.flatten(), y.flatten(), image.flatten() * scale_factor])

    with open(obj_filename, 'w') as obj_file:
        for v in vertices:
            obj_file.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for i in range(height - 2):
            for j in range(width - 2):
                index = i * width + j
                next_index = index + 1
                below_index = index + width

                obj_file.write(f"f {index+1} {next_index+1} {below_index+1}\n")
                obj_file.write(f"f {next_index+1} {next_index+2} {below_index+1}\n")

def plot_3d_wireframe(image, scale_factor=1.0):
    height, width = image.shape
    y, x = np.mgrid[0:height, 0:width]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, image * scale_factor, color='black')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def generate_landscape(size, scale, seed, octaves=20):
    noise_gen = PerlinNoise(octaves=octaves, seed=seed)

    landscape = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            landscape[i][j] = noise_gen([i / scale, j / scale])

    return landscape

from scipy.ndimage import gaussian_filter

def apply_gaussian_blur(image, sigma):
    """
    Apply Gaussian blur to a grayscale image represented by a NumPy array.

    Parameters:
    - image: NumPy array representing a grayscale image.
    - sigma: Standard deviation of the Gaussian filter.

    Returns:
    - Modified NumPy array after applying Gaussian blur.
    """
    blurred_image = gaussian_filter(image, sigma=sigma)
    return blurred_image

def save_image(landscape, filename):
    img = Image.fromarray((landscape * 255).astype(np.uint8))
    img.save(filename)

def normalize_image(landscape):
    minval, maxval = np.min(landscape), np.max(landscape)
    return(landscape - minval)/(maxval - minval)

def display_image(normalized):
    img = Image.fromarray((normalized * 255).astype(np.uint8))
    img.show()


from scipy.ndimage import distance_transform_edt
def apply_gradient_filter(image, threshold=0.5):
    """
    Apply a gradient filter to a grayscale image represented by a NumPy array.

    Parameters:
    - image: NumPy array representing a grayscale image.
    - threshold: Intensity threshold for region identification.

    Returns:
    - Gradient filter result as a NumPy array.
    """
    # Identify regions above the threshold
    regions = image > threshold

    regions_below = image <= threshold

    # Compute the Euclidean distance transform
    distance_transform = distance_transform_edt(regions)
    distance_transform_2 = distance_transform_edt(regions_below)

    # Apply a threshold to get the final result
    gradient_result = distance_transform * (image > threshold) - distance_transform_2 * (image <= threshold)

    return gradient_result

def reduce_image(img, factor):
    # Reduce the size of the image by averaging pixel values
    return img.reshape((img.shape[0] // factor, factor, img.shape[1] // factor, factor)).mean(axis=(1, 3))

def project_3d_point(pt, view_center, look_vector, up_vector=None):
    # Ensure the look_vector is normalized
    look_vector /= np.linalg.norm(look_vector)

    if up_vector is None:
        # If up_vector is not specified, compute it aiming up (perpendicular to the XY plane)
        up_vector = np.array([0., 0., 1.0])

    # Make sure up_vector is perpendicular to the look_vector
    up_vector -= np.dot(up_vector, look_vector) * look_vector
    up_vector /= np.linalg.norm(up_vector)

    # Compute the right vector using cross product
    right_vector = np.cross(look_vector, up_vector)

    # Project the 3D point onto the 2D plane
    pt_rel = pt - view_center
    x_proj = np.dot(pt_rel, right_vector)
    y_proj = np.dot(pt_rel, up_vector)
    z_proj = np.dot(pt_rel, look_vector)

    return np.array([x_proj, y_proj, z_proj])


class IntervalGroup:
    def __init__(self, data=None):
        self.data = data or []

    def ensure_sorted(self):
        self.data.sort(key=lambda xy: xy[0])

    def clamp_to(self, x, value, delta=1e-8):

        self.ensure_sorted()

        n = len(self.data)

        if n == 0 or x < self.data[0][0]-delta:
            self.data.insert(0, [x, value])
            return value

        if n > 0 and x > self.data[-1][0]+delta:
            self.data.append([x, value])
            return value

        i = 0

        # Find the suitable pair of consecutive interval points
        while i < n and x > self.data[i][0]:
            i += 1

        if i < n and abs(x - self.data[i][0]) < delta:
            self.data[i][1] = max(self.data[i][1], value)
            return self.data[i][1]

        if i > 0 and abs(x - self.data[i - 1][0]) < delta:
            self.data[i-1][1] = max(self.data[i-1][1], value)
            return self.data[i-1][1]

        x1, y1 = self.data[i - 1]
        x2, y2 = self.data[i]

        # Check if the point should be inserted based on interpolation
        interpolated_val =  np.interp(x, [x1, x2], [y1, y2])
        if value > interpolated_val:
            self.data.insert(i, [x, value])
            return value
        else:
            return interpolated_val

    def __len__(self):
        return len(self.data)



def adjust_horizontal_isoline(line, moving_horizon):
    # print("Line:", line.shape, len(line))
    # print("Initial moving horizon:", moving_horizon.data)

    updated_line = np.array(line).copy()
    # print("Updated line:", updated_line.shape)
    for j, pt in enumerate(updated_line):
        x, y = pt
        new_y = moving_horizon.clamp_to(x, y, delta=1e-6) 
        updated_line[j][1] = new_y
 
    return updated_line

def handle_occlusion_horiz(horizontal_isolines):
    adjusted_horizontal_isolines = []

    moving_horizon = None
    for i, line in enumerate(reversed(horizontal_isolines)):
        line = np.array(line)
        if moving_horizon is None:
            moving_horizon = IntervalGroup([[xy[0], xy[1]] for xy in line])
        adjusted_horizontal_isoline = adjust_horizontal_isoline(line, moving_horizon)
        adjusted_horizontal_isolines.append(adjusted_horizontal_isoline)

    return np.array(adjusted_horizontal_isolines)


def handle_occlusion(horizontal_isolines):
    adjusted_horizontal_isolines = handle_occlusion_horiz(horizontal_isolines)

    # Create vertical isolines by 'cutting' horizontal isolines
    vertical_isolines = []

    for i in range(len(adjusted_horizontal_isolines[0])):
        # generating ith vertical isoline
        vertical_isoline = [adjusted_horizontal_isolines[i][j] for j in range(len(adjusted_horizontal_isolines))]

        #vertical_isoline = [adjusted_horizontal_isolines[j][i] for j in range(len(adjusted_horizontal_isolines))]
        vertical_isolines.append(vertical_isoline)

    return adjusted_horizontal_isolines, np.array(vertical_isolines).T


def generate_occlusion_grid(img, factor=10, plot=False, tilt_angle=30, yaw_angle=30, view_point=(0, 0, 10)):
    img_reduced = reduce_image(img, factor=factor)

    # Compute look vector from tilt and yaw angles
    tilt_angle_rad = np.radians(tilt_angle)
    yaw_angle_rad = np.radians(yaw_angle)
    look_vector = np.array([np.cos(tilt_angle_rad) * np.sin(yaw_angle_rad),
                           np.cos(tilt_angle_rad) * np.cos(yaw_angle_rad),
                           np.sin(tilt_angle_rad)])

    # Compute 3D coordinates of the landscape points
    x, y = np.meshgrid(np.linspace(0, 1, img_reduced.shape[1]), np.linspace(0, 1, img_reduced.shape[0]))
    z = img_reduced

    # Flatten the coordinates and values
    points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))

    # Project the 3D points onto the viewing plane
    projected_points = np.array([project_3d_point(pt, view_point, look_vector) for pt in points])

    # Extract x, y, z coordinates of the projected points
    x_proj, y_proj, z_proj = projected_points[:, 0], projected_points[:, 1], projected_points[:, 2]

    # Generate horizontal and vertical isolines
    isolines_horizontal = []
    isolines_vertical = []

    for i in range(img_reduced.shape[0]):
        isolines_horizontal.append(list(zip(x_proj[i * img_reduced.shape[1]:(i + 1) * img_reduced.shape[1]], y_proj[i * img_reduced.shape[1]:(i + 1) * img_reduced.shape[1]])))

    for j in range(img_reduced.shape[1]):
        isolines_vertical.append(list(zip(x_proj[j::img_reduced.shape[1]], y_proj[j::img_reduced.shape[1]])))

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    isolines_horizontal = np.array(isolines_horizontal)
    isolines_vertical = np.array(isolines_vertical)
    print(f"isolines_horizontal.shape={isolines_horizontal.shape} isolines_vertical.shape={isolines_vertical.shape}")
    return isolines_horizontal, isolines_vertical, look_vector

