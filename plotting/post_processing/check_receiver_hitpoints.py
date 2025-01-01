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
    # folder_dir = "C:/Users/fang/Documents/NREL_SOLAR/optix/build_Allie/bin/Release/"
    # filename = folder_dir + "toyproblem-hit_counts-1000000_rays_with_buffer.csv"
    # # v1, v2 and anchor points of the receiver
    # v1 = np.array([2.0, 0.0, 0.0])
    # v2 = np.array([0.0, 1.788854, 0.894428])
    # anchor = np.array([-1.0, -0.894427, 9.552786])


    SOLVER = "solTrace"
    folder = "C:/Users/fang/Documents/NREL_SOLAR/large_scene/"
    filename = folder + "raydata_from_soltrace.csv"
    v1 = np.array([9.0, 0.0, 0.0])
    v2 = np.array([0.0, 0.0, 7.0])
    anchor = np.array([-4.5, 0.0, 76.5])



    receiver_pts_global = extract_receiver_points(filename, SOLVER)
    dim_x = np.linalg.norm(v1)
    dim_y = np.linalg.norm(v2)
    print("Receiver size: ", dim_x, dim_y)

    receiver_center = anchor + 0.5 * v1 + 0.5 * v2
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    v3_norm = np.cross(v2_norm, v1_norm)
    R = np.array([v1_norm, v2_norm, v3_norm]).T

    fig = plt.figure(figsize=(6, 12))
    receiver_pts_local = np.array([np.dot(R.T, pt - receiver_center) for pt in receiver_pts_global])

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(receiver_pts_local[:, 0], receiver_pts_local[:, 1], s=1)
    ax1.set_ylabel('Y (m)')
    ax1.set_xlabel('X (m)')
    ax1.set_title("Receiver Hit Points in Local Coordinates")
    ax1.set_xlim([-dim_x / 2, dim_x / 2])
    ax1.set_ylim([-dim_y / 2, dim_y / 2])
    ax1.set_aspect('equal', adjustable='box')

    bin_size = 0.05
    bins_x = np.arange(-dim_x / 2, dim_x / 2 + bin_size, bin_size)
    bins_y = np.arange(-dim_y / 2, dim_y / 2 + bin_size, bin_size)
    H, xedges, yedges = np.histogram2d(receiver_pts_local[:, 0], receiver_pts_local[:, 1], bins=[bins_x, bins_y])

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(H.T, origin='lower',
                    extent=[-dim_x / 2, dim_x / 2, -dim_y / 2, dim_y / 2],
                    aspect='auto', cmap=plt.cm.YlOrRd_r)
    plt.colorbar(im, ax=ax2, label='Counts')
    ax2.set_title("Binned Hit Counts")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_xlim([-dim_x / 2, dim_x / 2])
    ax2.set_ylim([-dim_y / 2, dim_y / 2])
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()
