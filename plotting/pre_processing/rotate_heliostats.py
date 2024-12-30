import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import io

# writes the rotated corners of the heliostats to a csv file, given the aim points and location of the center of the heliostats
# the input file can be either a .csv or a .stinput, generated from solar pilot
# each row is a heliostat element
# the columns are the global coordinates of the 3 corners of the heliostat, 0, 1 and 2 (the 4th corner can be calculated)
# rectangle is along x and y axis, where x is the long edge of the rectangle
# edge 12 is the bottom edge that is parallel to the global xy plane
# surface normal to the aim point is +z direction

#
#           ^ +y
#           |
#    0 ----------- 3
#    |      |      |
# -----------------------> +x
#    |      |      |
#    1 ----------- 2
#           |



parallelogram_dim_x = 9
parallelogram_dim_y = 6

def read_input(filename):
    loc_x = []
    loc_y = []
    loc_z = []
    aim_pt_x = []
    aim_pt_y = []
    aim_pt_z = []

    # check the extension of the file to see if it's csv or stinput
    if filename.endswith('.csv'):
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            next(csvreader)
            for row in csvreader:
                loc_x.append(float(row[3]))
                loc_y.append(float(row[4]))
                loc_z.append(float(row[5]))
                aim_pt_x.append(float(row[11]))
                aim_pt_y.append(float(row[12]))
                aim_pt_z.append(float(row[13]))

    if filename.endswith('.stinput'):
        with open(filename, 'r') as csvfile:
            # skip until we find the line that starts with "Heliostat field"
            for row in csvfile:
                if row.startswith("Heliostat field"):
                    break

            # now we want to extract the data
            for row in csvfile:
                if row.startswith("STAGE"):
                    break
                my_list = row.split()
                loc_x.append(float(my_list[1]))
                loc_y.append(float(my_list[2]))
                loc_z.append(float(my_list[3]))
                aim_pt_x.append(float(my_list[4]))
                aim_pt_y.append(float(my_list[5]))
                aim_pt_z.append(float(my_list[6]))

    return loc_x, loc_y, loc_z, aim_pt_x, aim_pt_y, aim_pt_z

def calculate_rotated_corners(position, surface_normal, width, height):
    """
    Calculate the four corners of a rotated rectangle representing the heliostat aperture.
    
    Parameters:
    - position: (x, y, z) coordinates of the heliostat
    - surface_normal: (x, y, z) aim vector (normal to the surface)
    - width: Width of the aperture
    - height: Height of the aperture
    
    Returns:
    - List of (x, y, z) corner coordinates in global space
    """    
    # Normalize the surface normal
    nx, ny, nz = surface_normal

    theta = np.arccos(nz)
    phi = np.arctan2(nx, ny)

    # Rotation about z by -phi
    Rz_phi = np.array([[np.cos(-phi), -np.sin(-phi), 0],
                    [np.sin(-phi),  np.cos(-phi), 0],
                    [0, 0, 1]])

    # Rotation about x by -theta 
    Rx_theta = np.array([[1, 0, 0],
                        [0, np.cos(-theta), -np.sin(-theta)],
                        [0, np.sin(-theta), np.cos(-theta)]])

    # Rotation matrix, local to global, align surface normal, edge parallel to xy plane
    R_LTG = Rz_phi @ Rx_theta

    # define 4 corners of the rectangle, centered at the origin
    half_width = width / 2
    half_height = height / 2

    # sequence: from top left corner, counter clockwise, +x is the long edge of the rectangle
    corners = np.array([[-half_width,  half_height, 0],
                    [-half_width, -half_height, 0],
                    [ half_width, -half_height, 0],
                    [ half_width,  half_height, 0]])

    # Rotate the rectangle in the global frame note that surface normal is now aligned with n 
    rotated_corners = [R_LTG @ corner + position for corner in corners]

    # check if the "bottom edge" is indeed on the bottom! 
    # if not, rotate around z by pi 
    if (rotated_corners[1][2] > rotated_corners[0][2]):
        Rz_pi = np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
                        [np.sin(np.pi),  np.cos(np.pi), 0],
                        [0, 0, 1]])
        R_LTG = R_LTG @ Rz_pi
        rotated_corners = [R_LTG @ corner + position for corner in corners]



    edge1 = rotated_corners[2] - rotated_corners[1]   # bottom edge +x direction
    edge2 = rotated_corners[0] - rotated_corners[1]   # left edge +y direction
    normal_rect = np.cross(edge1, edge2)
    normal_rect /= np.linalg.norm(normal_rect)
    # print("input and output normal diff: ", np.linalg.norm(np.array([nx, ny, nz]) - normal_rect))
    # print("z of 2 vertices bottom diff : ", rotated_corners[1][2] - rotated_corners[2][2])

    # check that normal_unit and normal_rect are parallel (and same direction! )
    assert np.allclose(np.array([nx, ny, nz]), normal_rect)
    # check that bottom edge is parallel to xy plane
    assert np.allclose(edge1[2] , 0)
    
    return rotated_corners

# read csv file
folder_dir = "C:/Users/fang/Documents/NREL_SOLAR/large_scene/"
# filename = folder_dir + "small-system-coordinates.csv"
# filename = folder_dir + "small-system.stinput"
filename = folder_dir + "large-system-coordinates.csv"
loc_x, loc_y, loc_z, aim_pt_x, aim_pt_y, aim_pt_z = read_input(filename)

PLOT = False

# Now plot aim points in 3D space
if PLOT:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(aim_pt_x, aim_pt_y, aim_pt_z, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

# now also plot loc points, for now use scatter, blue colored, marker should be a square
test_id = len(loc_x)

# create a csv file to store the results
# header is x,y,z,nx,ny,nz,R0_x,R0_y,R0_z,R1_x,R1_y,R1_z,R2_x,R2_y,R2_z,R3_x,R3_y,R3_z
output_stream = io.StringIO()
csvwriter = csv.writer(output_stream)
csvwriter.writerow(['R0_x', 'R0_y', 'R0_z', 'R1_x', 'R1_y', 'R1_z', 'R2_x', 'R2_y', 'R2_z'])

for i in range(test_id):
    print("element: " + str(i))

    normal = np.array([aim_pt_x[i] - loc_x[i], aim_pt_y[i] - loc_y[i], aim_pt_z[i] - loc_z[i]])
    normalized = normal / np.linalg.norm(normal)

    position = np.array([loc_x[i], loc_y[i], loc_z[i]])
    rotated_corners = calculate_rotated_corners(position, normalized, parallelogram_dim_x, parallelogram_dim_y)

    # write the results to the csv file
    csvwriter.writerow([rotated_corners[0][0], rotated_corners[0][1], rotated_corners[0][2],
                        rotated_corners[1][0], rotated_corners[1][1], rotated_corners[1][2],
                        rotated_corners[2][0], rotated_corners[2][1], rotated_corners[2][2]])


    if PLOT:
        # Plot the rotated corners
        for corner in rotated_corners:
            ax.scatter(corner[0], corner[1], corner[2], c='b', marker='*')

        # grab the direction of the edges of the rectangle, cross those two directions,
        # i should get the normal of the rectangle, then plot the normal vector
        edge1 = np.array([rotated_corners[1][0] - rotated_corners[0][0], rotated_corners[1][1] - rotated_corners[0][1], rotated_corners[1][2] - rotated_corners[0][2]])
        edge2 = np.array([rotated_corners[2][0] - rotated_corners[1][0], rotated_corners[2][1] - rotated_corners[1][1], rotated_corners[2][2] - rotated_corners[1][2]])

        rect_normal = np.cross(edge1, edge2)
        rect_normal = rect_normal / np.linalg.norm(rect_normal)

        # find start of quiver, which is the mid point of the rectangle
        rect_normal_loc = 0.25 * (rotated_corners[0] + rotated_corners[1] + rotated_corners[2] + rotated_corners[3])
        ax.quiver(rect_normal_loc[0], rect_normal_loc[1], rect_normal_loc[2], rect_normal[0], rect_normal[1], rect_normal[2], length=20, color='g')

        # also plot the given normal vector
        ax.quiver(loc_x[i], loc_y[i], loc_z[i], normalized[0], normalized[1], normalized[2], length=20, color='r')

if filename.endswith('.csv'):
    output_filename = filename.replace('.csv', '_rotated.csv')
elif filename.endswith('.stinput'):
    output_filename = filename.replace('.stinput', '_rotated.csv')

# Write the in-memory stream to the file once
with open(output_filename, mode='w', newline='') as csvfile:
    csvfile.write(output_stream.getvalue())



if PLOT:
    ax.scatter(loc_x[0:test_id], loc_y[0:test_id], loc_z[0:test_id], c='b', marker='s')
    plt.show()