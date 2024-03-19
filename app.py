import sys
import numpy as np
from perlin_noise import PerlinNoise
from PIL import Image, ImageFilter
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import svgwrite
from datetime import datetime
from landscaper import generate_landscape, apply_gaussian_blur, apply_gradient_filter, normalize_image, display_image, save_image
from landscaper import generate_occlusion_grid, handle_occlusion, generate_3d_obj, plot_3d_wireframe, generate_isometric_svg


def save_isolines_svg(adjusted_horizontal_isolines, output_path='output.svg', svg_size =(1600, 800)):
    # Create an SVG drawing
    dwg = svgwrite.Drawing(output_path, profile='tiny', size=svg_size)

    # Calculate the range of points in adjusted_horizontal_isolines
    min_x = np.min(adjusted_horizontal_isolines[:, :, 0])
    max_x = np.max(adjusted_horizontal_isolines[:, :, 0])
    min_y = np.min(adjusted_horizontal_isolines[:, :, 1])
    max_y = np.max(adjusted_horizontal_isolines[:, :, 1])

    # Set the viewbox based on the range of points
    viewbox = (min_x, min_y, max_x - min_x, max_y - min_y)
    dwg.viewbox(*viewbox)

    # Create a group (<g>) to contain individual polylines
    g = dwg.g()

    # Iterate through the rows of polylines and add them to the group
    for row in adjusted_horizontal_isolines:
        # print(row)
        polyline = dwg.polyline(points=row.tolist(), fill='none', stroke='black', stroke_width=0.002)
        g.add(polyline)

    # Add the group to the SVG drawing
    dwg.add(g)

    # Save the SVG file
    dwg.save()

def main():

    today = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")

    config_path = sys.argv[1] if len(sys.argv) >= 2 else "config/test2.yaml"
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    size = config["size"]
    scale = config["scale"]
    seed = config["seed"]
    radius = config["radius"]

    image_output_path = f'target/{today}.png'
    isolines_output_path = f'target/{today}_iso.svg'
    obj_output_path = f'target/{today}.obj'

    landscape = generate_landscape(size, scale, seed)
    blurred_landscape = apply_gaussian_blur(landscape, radius)
    gradient_landscape = apply_gradient_filter(blurred_landscape, threshold=0.0)
    blurred_landscape_2 = apply_gaussian_blur(gradient_landscape, 2*radius)
    #blurred_landscape = landscape
 
    normalized = normalize_image(blurred_landscape_2)
    save_image(normalized, image_output_path)


    # Example usage:
    # Assuming you have a raster image represented by a NumPy array img of shape (M, M)
    # and you want to generate a 3D OBJ file and visualize it with wireframe grid:

    img = normalized

    # Example usage:
    # Assuming you have a 2D numpy array 'image' with values between 0 and 1
    isolines_horizontal, isolines_vertical, look_vector = generate_occlusion_grid(
        img, 
        factor=config["occ_grid_factor"], 
        plot=config["plot"], 
        tilt_angle=config["occ_grid_tilt_angle"],
        yaw_angle=config["occ_grid_yaw_angle"], 
        view_point=(0, 0, config["occ_grid_view_point_z"]))
    # fig = plt.figure()
    # for il in isolines_horizontal:
    #     il = np.array(il)
    #     plt.plot(il[:,0], il[:,1], color="r")
    # fig = plt.figure()
    # for il in isolines_vertical:
    #     il = np.array(il)
    #     plt.plot(il[:,0], il[:,1], color="b")

    adjusted_horizontal_isolines, adjusted_vertical_isolines = handle_occlusion(isolines_horizontal)
    print(adjusted_horizontal_isolines.shape, adjusted_vertical_isolines.shape)
    fig = plt.figure()
    for il in adjusted_horizontal_isolines:
        il = np.array(il)
        plt.plot(il[:,0], il[:,1], color="r")

    plt.show()

    save_isolines_svg(adjusted_horizontal_isolines, output_path=isolines_output_path, svg_size=tuple(config["svg_size"]))

    # Generate 3D OBJ file
    scale_factor = config["scale_factor"]
    generate_3d_obj(img, obj_output_path, scale_factor=scale_factor)

    # Plot 3D wireframe
    # plot_3d_wireframe(img, scale_factor=scale_factor)

    # Example usage:
    # Assuming you have a raster image represented by a NumPy array img of shape (M, M)
    # and you want to generate an isometric SVG:

    # Choose an appropriate max_height_factor to achieve the desired maximum height
    max_height_factor = 10.0
    # Generate isometric SVG
    # generate_isometric_svg(img, 'target/landscape_isometric.svg', max_height_factor=max_height_factor)


if __name__ == "__main__":
    main()