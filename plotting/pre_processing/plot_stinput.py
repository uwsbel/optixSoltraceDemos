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


# read csv file
folder_dir = "C:/Users/fang/Documents/NREL_SOLAR/large_scene/"
# filename = folder_dir + "small-system-coordinates.csv"
filename = folder_dir + "small-system.stinput"
# filename = folder_dir + "large-system-coordinates.csv"
loc_x = []
loc_y = []
loc_z = []
aim_pt_x = []
aim_pt_y = []
aim_pt_z = []

radius = 160

# we want to create a smaller file that has the same info but less heliostats, only the one that are close to the receiver
# create a new file
filename_out = folder_dir + "debug-system_test.stinput"
# everytime we read something from the input file, we write it to the output file
# we only write the heliostats that are close to the receiver 



with open(filename, 'r') as csvfile:
    # skip until we find the line that starts with "Heliostat field"
    for row in csvfile:
        # write the row to the output file
        with open(filename_out, 'a') as csvfile_out:
            csvfile_out.write(row)
        if row.startswith("Heliostat field"):
            break

    # now we want to extract the data
    count = 0
    for row in csvfile:
        if row.startswith("STAGE"):
            with open(filename_out, 'a') as csvfile_out:
                csvfile_out.write(row)
            break
        my_list = row.split()

        # only have the heliostats that are close the the receiver
        x_pos = float(my_list[1])
        y_pos = float(my_list[2])

        if x_pos**2 + y_pos**2 < radius**2:

            loc_x.append(float(my_list[1]))
            loc_y.append(float(my_list[2]))
            loc_z.append(float(my_list[3]))
            aim_pt_x.append(float(my_list[4]))
            aim_pt_y.append(float(my_list[5]))
            aim_pt_z.append(float(my_list[6]))

            print("element number: " + str(count))
            # write the row to the output file
            with open(filename_out, 'a') as csvfile_out:
                csvfile_out.write(row)
                print(row)

        count = count + 1

    # write the rest of the data to the output file
    for row in csvfile:
        with open(filename_out, 'a') as csvfile_out:
            csvfile_out.write(row)
            print(row)

# print number of heliostats
print("Number of heliostats: " + str(count))
