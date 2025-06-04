# demo program for ray cylinder intersection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for now ray is in the same coordinate system as the cylinder 
# TODO: check if left handed or right handed coordinate system affect the result ... it should not be for z aligned system ... although it affect normal vector? 

# Define the ray
# first intersection outside bounds, second intersection inside bounds, hitting inside of the cylinder
# ray_origin = np.array([2.0, 3.2, 0.5])   
# ray_direction = np.array([-1.0, -0.8, -0.1])

# first intersection outside bounds, second intersection outside bounds (this should be taken care of by aabb check)
# ray_origin = np.array([2.0, 3.2, 0.5])   
# ray_direction = np.array([-1.0, 0.0, -0.1])

# # ray origin at the surface 
# ray_origin = np.array([1.0, 1.9, 1.0])   
# ray_direction = np.array([-1.0, 0.0, 0.0])

# ray origin inside the cylinder within the bounds
# ray_origin = np.array([0.0, 0.0, 0.5])   
# ray_direction = np.array([-0.5, 0.0, -1.0])

# ray origin inside the cylinder outside the bounds
# ray_origin = np.array([0.0, -3.0, 0.5])   
# ray_direction = np.array([-0.5, 0.0, -1.0])

# ray origin inside the cylinder outside the bounds, but hit points within the bounds
ray_origin = np.array([0.0, -3.0, 0.5])   
ray_direction = np.array([-0.5, 3.0, -1.0])


ray_direction = ray_direction/np.linalg.norm(ray_direction)
intersection_point = np.array([0.0, 0.0, 0.0])

r = 1.2

# Define the cylinder
cylinder_height = 4.0

Xdelta = ray_origin[0]
Ydelta = ray_origin[1]
Zdelta = ray_origin[2]

cos_x = ray_direction[0]
cos_y = ray_direction[1]
cos_z = ray_direction[2]
			
A = cos_x * cos_x  + cos_z * cos_z
B = 2.0 * (Xdelta*cos_x + Zdelta*cos_z)
C = Xdelta*Xdelta + Zdelta*Zdelta - r*r

# now see if there will be a solution
determinant = B*B - 4*A*C

print("determinant = ", determinant)

intersect_flag = False

if determinant > 0:
    # t1 > t2
    t1 = (-B + np.sqrt(determinant))/(2*A)
    t2 = (-B - np.sqrt(determinant))/(2*A)

    print("t1 = ", t1)
    print("t2 = ", t2)

    # ray location outside the cylinder
    if t2 > 0:

        print("initial ray location outside the cylinder")
        t = t2
        intersection_point[0] = ray_origin[0] + ray_direction[0] * t
        intersection_point[1] = ray_origin[1] + ray_direction[1] * t
        intersection_point[2] = ray_origin[2] + ray_direction[2] * t

        if intersection_point[1] > cylinder_height/2 or intersection_point[1] < -cylinder_height/2:
            t = t1
            intersection_point[0] = ray_origin[0] + ray_direction[0] * t
            intersection_point[1] = ray_origin[1] + ray_direction[1] * t
            intersection_point[2] = ray_origin[2] + ray_direction[2] * t
        
    if np.abs(t2) < 1e-8:  # initial ray location at the cylinder surface (could be infinite)
        print("initial ray location at the cylinder surface")
        t = t1
        intersection_point[0] = ray_origin[0] + ray_direction[0] * t
        intersection_point[1] = ray_origin[1] + ray_direction[1] * t
        intersection_point[2] = ray_origin[2] + ray_direction[2] * t

    if t2 < 0 and t1 > 0:  # ray origin inside the cylinder
        print("ray origin inside the cylinder")
        t = t1
        intersection_point[0] = ray_origin[0] + ray_direction[0] * t
        intersection_point[1] = ray_origin[1] + ray_direction[1] * t
        intersection_point[2] = ray_origin[2] + ray_direction[2] * t
    if t1 < 0 :  # ray heading away from the cylinder 
        print("Ray is heading away from the cylinder")

elif np.abs(determinant) < 1e-8:
    print("Ray is tangent to the cylinder")
    t = -B/(2*A)
    intersection_point[0] = ray_origin[0] + ray_direction[0] * t
    intersection_point[1] = ray_origin[1] + ray_direction[1] * t
    intersection_point[2] = ray_origin[2] + ray_direction[2] * t
else: # no intersection
    print("No intersection")

# now let's draw the cylinder in 3D, with the height
# draw ray origin, arrow and intersection point 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# draw cylinder
u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(-cylinder_height/2, cylinder_height/2, 100)
U, V = np.meshgrid(u, v)
X = r * np.cos(U)
Z = r * np.sin(U)

Y = V
ax.plot_surface(X, Y, Z, alpha=0.5)

# draw ray origin
ax.scatter(ray_origin[0], ray_origin[1], ray_origin[2], color='r')

# draw ray arrow
ax.quiver(ray_origin[0], ray_origin[1], ray_origin[2], ray_direction[0], ray_direction[1], ray_direction[2], length=1, color='r')

# draw intersection point
ax.scatter(intersection_point[0], intersection_point[1], intersection_point[2], color='g')


# make sure axis is equal
ax.set_aspect('equal')

# set axis labels, x y and z
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()