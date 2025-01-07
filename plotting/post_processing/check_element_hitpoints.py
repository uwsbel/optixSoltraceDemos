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
            if Element[i] == "1":
                receiver_pts_global.append(np.array([loc_x[i], loc_y[i], loc_z[i]]))

    return np.array(receiver_pts_global)


if __name__ == "__main__":
    SOLVER = "solTrace"
    folder_dir = "C:/Users/fang/Documents/NREL_SOLAR/large_scene/"
    filename = folder_dir + "raydata_from_soltrace.csv"


    pts_global = extract_receiver_points(filename, SOLVER)

    # i need to figure out the normal of the plane that covers all those points
    # best way is to find random points, have two vectors, and perform cross product 
    # to get the normal vector

    # find any three point from pts_global
    p1 = pts_global[0]
    p2 = pts_global[1]
    p3 = pts_global[2]

    # two vectors
    v1 = p2 - p1
    v2 = p3 - p1

    # normal vector
    normal = np.cross(v1, v2)

    # normalize it
    normal = normal / np.linalg.norm(normal)

    print("Normal vector of the plane: ", normal)

    # now test a different point to see if it is on the plane
    p4 = pts_global[3]
    # calculate the dot product of the normal vector and the vector from p1 to p4

    # if the dot product is zero, then the point is on the plane
    print("dot product: ", np.dot(normal, p4 - p1)) 
    if np.dot(normal, p4 - p1) == 0:
        print("Point is on the plane")
    else:
        print("Point is not on the plane")