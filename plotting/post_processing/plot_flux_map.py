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
    nx = 50
    ny = 50

    # SOLVER = "optix"
    # SURFACE = "FLAT"
    # folder_dir = "C:/optixSoltraceDemos_build/bin/Release/"
    # filename = folder_dir + "1700_elements_1000000_rays.csv"
    # # v1, v2 and anchor points of the receiver
    # v1 = np.array([9.0, 0.0, 0.0])
    # v2 = np.array([0.0, 0, 7])
    # anchor = np.array([-4.5, 0.0, 76.5])

    SOLVER = "optix"
    SURFACE = "FLAT"
    folder_dir = "C:/Users/allie/Documents/SolTrace/hit_point_data/"
    filename = folder_dir + "toy_problem_parabolic.csv"
    # v1, v2 and anchor points of the receiver
    v1 = np.array([2.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.788854, 0.894428])
    anchor = np.array([-1.0, -0.894427, 9.552786])

    # SOLVER = "solTrace"
    # SURFACE = "FLAT"
    # folder_dir = "C:/Users/allie/Documents/SolTrace/hit_point_data/"
    # filename = folder_dir + "cpu_toy_problem_sun_offset_sun_shape_on-1_cpu.csv"
    # # v1, v2 and anchor points of the receiver
    # v1 = np.array([2.0, 0.0, 0.0])
    # v2 = np.array([0.0, 1.788854, 0.894428])
    # anchor = np.array([-1.0, -0.894427, 9.552786])

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
    # filename = folder_dir + "cyl_receiver_uniform_rdm_sun_sample.csv"
    # # R, H, C radius, height, and center of receiver
    # R = 1.0
    # H = 2.4
    # C = np.array([0.0, 0.0, 10.0])

    # R, H, C radius, height, and center of receiver
    # R = 9.0
    # H = 22
    # C = np.array([0.0, 0.0, 195.0])
    # BASE_X, BASE_Z local x-z (Circle plane)
    # BASE_X = np.array([1.0, 0.0, 0.0])
    # BASE_Z = np.array([0.0, -1.0, 0.0])

    # SOLVER = "solTrace"
    # SURFACE = "CYLINDRICAL"
    # # folder = "C:/Users/fang/Documents/NREL_SOLAR/optix/optixSoltraceDemos/data/stinputs/"
    # # filename = folder + "raydata_0_slope_error.csv"
    # folder = "C:/Users/allie/Documents/SolTrace/hit_point_data/"
    # filename = folder + "cyl_receiver_uniform_rdm_sun_sample_soltrace_cpu.csv"
    # R = 1.0
    # H = 2.4
    # C = np.array([0.0, 0.0, 10.0])
    # # BASE_X, BASE_Z local x-z (Circle plane)
    # BASE_X = np.array([1.0, 0.0, 0.0])
    # BASE_Z = np.array([0.0, -1.0, 0.0])

    receiver_pts_global = extract_receiver_points(filename, SOLVER)
    flux_2D = np.zeros((ny,nx))

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
 
        x_local = receiver_pts_local[:, 0].T
        y_local = receiver_pts_local[:, 1].T

        raybins_x = np.floor((x_local + dim_x/2)/dim_x*nx).astype(int)
        raybins_y = np.floor((y_local + dim_y/2)/dim_y*ny).astype(int)

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

        raybins_x = np.floor((x_local.T + np.pi)*nx/(2 * np.pi)).astype(int)
        raybins_y = np.floor((y_local.T + H/2)/H*ny).astype(int)

        title_scatter = f"Receiver Hit Points in X-Y Projected Space \n Total # of Hits: {len(x_local)}"
        title_heatmap = "Binned Hit Counts in X-Y Projected Space"


    dx = dim_x / nx
    dy = dim_y / ny
    x_rec = np.arange(0, dim_x, dim_x/nx)
    y_rec = np.arange(0, dim_y, dim_y/ny)
    # Calculate flux metrics
    # TODO - Output sun stats as well and read data, manual right now
    # sun shape off
    # sun_xmax = 0.519808
    # sun_xmin = -5.94168
    # sun_ymax = 5.94892
    # sun_ymin = -5.94892
    # sun shape on
    sun_xmax = 0.520613
    sun_xmin = -5.9434
    sun_ymax = 5.95019
    sun_ymin = -5.95019      
    nsunrays = 1000000

    dni = 1000.0  # Direct normal irradiance in W/m^2
    power_per_ray = (sun_xmax - sun_xmin) * (sun_ymax - sun_ymin) / nsunrays * dni
    # Compute power per ray (ppr) based on node area
    anode = dx * dy
    ppr = power_per_ray / anode

    for r in range(len(raybins_x)):
        flux_2D[raybins_x[r], raybins_y[r]] += ppr 

    peak_flux = np.max(flux_2D)
    min_flux = np.min(flux_2D[flux_2D > 0])  # Ignore empty bins
    #avg_flux = np.mean(flux_2D[flux_2D > 0])
    avg_flux = np.mean(flux_2D)
    sigma_flux = np.std(flux_2D[flux_2D > 0])
    uniformity = sigma_flux / avg_flux

    # Placeholder uncertainty model (TODO: UPDATE)
    # peak_flux_uncertainty = peak_flux * 0.05
    # avg_flux_uncertainty = avg_flux * 0.05

    print(f"Total # of Hits: {len(x_local)}")
    print(f"Peak flux: {peak_flux:.2f}") # ± {peak_flux_uncertainty:.2f}")
    print(f"Min flux: {min_flux:.2f}")
    print(f"Avg flux: {avg_flux:.2f}") # ± {avg_flux_uncertainty:.2f}")
    #print(f"Sigma: {sigma_flux:.2f}")
    #print(f"Uniformity: {uniformity:.3f}")

    plt.title(f"Flux intensity")
    Xr,Yr = np.meshgrid(y_rec, x_rec)
    plt.contourf(Yr, Xr, flux_2D, levels=nx)
    plt.colorbar()
    plt.title(f"Receiver | Max {peak_flux:.0f} | Mean {avg_flux:.0f}")
    plt.xlabel("X-axis position")
    plt.ylabel("Y-axis position")
    plt.tight_layout()
    plt.show()