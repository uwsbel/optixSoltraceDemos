import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import io

# writes the rotated corners of the heliostats to a csv file, given the aim points and location of the center of the heliostats
# the input file can be either a .csv or a .stinput, generated from solar pilot
# each row is a heliostat element
# the columns are the global coordinates v1,v2 and the anchor point
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



    edge1 = rotated_corners[2] - rotated_corners[1]   # bottom edge -x direction
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





#v1_x,v1_y,v1_z,v2_x,v2_y,v2_z,anchor_x,anchor_y,anchor_z
#-1.1353348133236523,2.776871416118439,4.440892098500626e-16,1.8368626048375063,0.7510085092377778,2.24987154966721,1406.5692361042431,-235.7159399626781,1.1228542251663949



# input x, y, z, aimpoint_x, aimpoint_y, aimpoint_z ( i don't know what zrot is for )
loc_x = 1406.92
loc_y = -233.952
loc_z = 2.24779

aim_pt_x = 712.743
aim_pt_y = -517.769
aim_pt_z = 663.733

dim_x = 3
dim_y = 3

parallelogram_dim_x = 3
parallelogram_dim_y = 3


normal = np.array([aim_pt_x - loc_x, aim_pt_y - loc_y, aim_pt_z - loc_z])
normalized = normal / np.linalg.norm(normal)

position = np.array([loc_x, loc_y, loc_z])
rotated_corners = calculate_rotated_corners(position, normalized, parallelogram_dim_x, parallelogram_dim_y)

# vector v1 points from 2 to 1, vector v2 points from 2 to 3, anchor is point 2 
v1 = rotated_corners[1] - rotated_corners[2]
v2 = rotated_corners[3] - rotated_corners[2]
anchor = rotated_corners[2]


print("receiver results: ")
print("v1: ", v1)
print("v2: ", v2)
print("anchor: ", anchor)


# print out some test scenarios
v1 = np.array([2, 0, 0])
v2 = np.array([0, 1.788854, 0.894428])

normal = np.cross(v2, v1)

anchor = np.array([-1.0, -0.894427, 9.552786])

origin = anchor + v1/2 + v2/2
print("origin: ", origin)   

aim_point = origin + normal 

print("aim point: ", aim_point)

print("dim x: " + str(np.linalg.norm(v1))) 
print("dim y: " + str(np.linalg.norm(v2)))