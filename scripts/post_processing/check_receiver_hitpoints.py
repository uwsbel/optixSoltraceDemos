import numpy as np
import matplotlib.pyplot as plt
import csv

# extract receiver points in global reference frame from the input file
def extract_receiver_points(filename, solver="optix"):
    receiver_pts_global = []
    if solver == "optix":
        loc_x, loc_y, loc_z, stage, number = [], [], [], [], []
        with open(filename, mode='r') as file:
            csvFile = csv.reader(file)
            header = next(csvFile)
            for lines in csvFile:
                number.append(int(lines[0]))
                stage.append(int(lines[1]))
                loc_x.append(float(lines[2]))
                loc_y.append(float(lines[3]))
                loc_z.append(float(lines[4]))
        
        for i in range(len(stage)):
            if stage[i] == 2:
                receiver_pts_global.append(np.array([loc_x[i], loc_y[i], loc_z[i]]))

    elif solver == "solTrace":
        loc_x = []
        loc_y = []
        loc_z = []
        Element = []

        with open(filename, mode='r') as file:
            csvFile = csv.reader(file)
            header = next(csvFile)
            for lines in csvFile:
                loc_x.append(float(lines[0]))
                loc_y.append(float(lines[1]))
                loc_z.append(float(lines[2]))
                Element.append(lines[6])

        for i in range(len(Element)):
            if Element[i] == "-1":
                receiver_pts_global.append(np.array([loc_x[i], loc_y[i], loc_z[i]]))

    return np.array(receiver_pts_global)


if __name__ == "__main__":
    # SOLVER = "optix"
    # SURFACE = "FLAT"
    # folder_dir = "C:/optixSoltraceDemos_build/bin/Release/"
    # filename = folder_dir + "1700_elements_1000000_rays.csv"
    # # v1, v2 and anchor points of the receiver
    # v1 = np.array([9.0, 0.0, 0.0])
    # v2 = np.array([0.0, 0, 7])
    # anchor = np.array([-4.5, 0.0, 76.5])

    # SOLVER = "solTrace"
    # SURFACE = "FLAT"
    # folder = "C:/Users/allie/Documents/SolTrace/"
    # filename = folder + "small-system-soltrace-raydata-flat.csv"
    # v1 = np.array([9.0, 0.0, 0.0])
    # v2 = np.array([0.0, 0, 7])
    # anchor = np.array([-4.5, 0.0, 76.5])

    # SOLVER = "optix"
    # SURFACE = "CYLINDRICAL"
    # # folder_dir = "C:/optixSoltraceDemos_build/bin/Release/"
    # # folder_dir = "C:/Users/fang/Documents/NREL_SOLAR/optix/build_debug/bin/Release/"
    # folder_dir = "C:/Users/allie/Documents/SolTrace/hit_point_data/"
    # filename = folder_dir + "cyl_receiver_vertical_capped_gpu_15384138_rays_raydata_cleaned.csv"
    # # R, H, C radius, height, and center of receiver
    # R = 1.0
    # H = 2.4
    # C = np.array([0.0, 0.0, 10.0])

    # R, H, C radius, height, and center of receiver
    # R = 9.0
    # H = 22
    # C = np.array([0.0, 0.0, 195.0])

    # BASE_X, BASE_Z local x-z (Circle plane)
    BASE_X = np.array([1.0, 0.0, 0.0])
    BASE_Z = np.array([0.0, -1.0, 0.0])

    SOLVER = "solTrace"
    SURFACE = "CYLINDRICAL"
    # folder = "C:/Users/fang/Documents/NREL_SOLAR/optix/optixSoltraceDemos/data/stinputs/"
    # filename = folder + "raydata_0_slope_error.csv"
    folder = "C:/Users/allie/Documents/SolTrace/hit_point_data/"
    filename = folder + "cyl_receiver_vertical_capped_cpu_15384138_rays_raydata.csv"
    R = 1.0
    H = 2.4
    C = np.array([0.0, 0.0, 10.0])
    # BASE_X, BASE_Z local x-z (Circle plane)
    BASE_X = np.array([1.0, 0.0, 0.0])
    BASE_Z = np.array([0.0, -1.0, 0.0])

    receiver_pts_global = extract_receiver_points(filename, SOLVER)

    fig = plt.figure(figsize=(9, 6))

    if SURFACE == "FLAT":
        dim_x = np.linalg.norm(v1)
        dim_y = np.linalg.norm(v2)
        print("Receiver size: ", dim_x, dim_y)

        receiver_center = anchor + 0.5 * v1 + 0.5 * v2
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        v3_norm = np.cross(v2_norm, v1_norm)
        R = np.array([v1_norm, v2_norm, v3_norm]).T

        receiver_pts_local = np.array([np.dot(R.T, pt - receiver_center) for pt in receiver_pts_global])
 
        x_local = receiver_pts_local[:, 0]
        y_local = receiver_pts_local[:, 1]

        title_scatter = f"Receiver Hit Points in Local Coordinates \n Total # of Hits: {len(x_local)}"
        title_heatmap = "Binned Hit Counts"

    if SURFACE == "CYLINDRICAL":
        dim_x = 2 * np.pi * R   # Full circumference 
        dim_y = H               # Cylinder height

        translated_pts = receiver_pts_global - C
        cylinder_axis = np.cross(BASE_Z, BASE_X)

        # Project onto the plane perpendicular to the cylinder's axis
        projection = translated_pts - np.outer(np.dot(translated_pts, cylinder_axis), cylinder_axis)

        # Compute cylindrical coordinates
        theta = np.arctan2(translated_pts[:, 1], translated_pts[:, 0])
        # Compute height (z) along the cylinder's axis
        z = np.dot(translated_pts, cylinder_axis)

        # Filter out points that are on the caps
        cap_tolerance = 1e-6  # Adjust tolerance as needed
        is_on_cap = np.logical_or(
            np.abs(z - np.min(z)) < cap_tolerance,  # Points on bottom cap
            np.abs(z - np.max(z)) < cap_tolerance   # Points on top cap
        )
        curved_surface_points = ~is_on_cap  # Invert to get points on the curved surface

        # Apply the filter
        theta = theta[curved_surface_points]
        z = z[curved_surface_points]

        # Map cylindrical coordinates to x-y plane 
        x_local = R * theta
        y_local = z

        title_scatter = f"Receiver Hit Points in X-Y Projected Space \n Total # of Hits: {len(x_local)}"
        title_heatmap = "Binned Hit Counts in X-Y Projected Space"

    # Scatter plot in unwrapped x-y space
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(x_local, y_local, s=1)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title(title_scatter)
    ax1.set_xlim([-dim_x / 2, dim_x / 2])
    ax1.set_ylim([-dim_y / 2, dim_y / 2])
    ax1.set_aspect('equal', adjustable='box')

    bin_size = 0.05
    bins_x = np.arange(-dim_x / 2, dim_x / 2 + bin_size, bin_size)
    bins_y = np.arange(-dim_y / 2, dim_y / 2 + bin_size, bin_size)
    H, xedges, yedges = np.histogram2d(x_local, y_local, bins=[bins_x, bins_y])

    # Heatmap of binned hit counts
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(H.T, origin='lower',
                    extent=[-dim_x / 2, dim_x / 2, -dim_y / 2, dim_y / 2],
                    aspect='auto', cmap=plt.cm.YlOrRd_r)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Counts')
    ax2.set_title(title_heatmap)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_xlim([-dim_x / 2, dim_x / 2])
    ax2.set_ylim([-dim_y / 2, dim_y / 2])
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()
