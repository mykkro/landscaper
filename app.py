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
from landscaper import generate_landscape, apply_gaussian_blur, apply_gradient_filter, normalize_image, display_image, save_image
from landscaper import generate_occlusion_grid, handle_occlusion, generate_3d_obj, plot_3d_wireframe, generate_isometric_svg

def main():
    size = 200
    scale = 100.0
    seed = 0
    radius = 6.0

    landscape = generate_landscape(size, scale, seed)
    blurred_landscape = apply_gaussian_blur(landscape, radius)
    print(blurred_landscape.min())
    gradient_landscape = apply_gradient_filter(blurred_landscape, threshold=0.0)
    blurred_landscape_2 = apply_gaussian_blur(gradient_landscape, 2*radius)
    #blurred_landscape = landscape
 
    normalized = normalize_image(blurred_landscape_2)
    save_image(normalized, 'target/landscape.png')


    # Example usage:
    # Assuming you have a raster image represented by a NumPy array img of shape (M, M)
    # and you want to generate a 3D OBJ file and visualize it with wireframe grid:

    img = normalized

    # Example usage:
    # Assuming you have a 2D numpy array 'image' with values between 0 and 1
    isolines_horizontal, isolines_vertical, look_vector = generate_occlusion_grid(img, factor=2, plot=False, tilt_angle=30, yaw_angle=45, view_point=(0, 0, 10))
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

    # Generate 3D OBJ file
    scale_factor = 10
    generate_3d_obj(img, 'target/landscape.obj', scale_factor=scale_factor)

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