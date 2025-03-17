import numpy as np
import matplotlib.pyplot as plt


# GPU heliostat geometry
rect1_3d = np.array([
    [-4.05108223553502, 0.5, -0.224009], 
    [-4.05108223553502, -0.5, -0.224009],
    [-5.94891776446498, -0.5, 0.2240091973214239],
    [-5.94891776446498, 0.5, 0.2240091973214239],
])

n = np.cross(rect1_3d[1] - rect1_3d[0], rect1_3d[2] - rect1_3d[1])
n /= np.linalg.norm(n)
print("GPU heliostat normal:", n)

# CPU heliostat geometry, computed by printing max/min x, y, z hit point values
# Minimum x: -5.958820461685046, Maximum x: -4.041069770525555
# Minimum y: -0.5208086614526061, Maximum y: 0.5206490038702324
# Minimum z: -0.2263727198354415, Maximum z: 0.22634680717540345

rect2_3d = np.array([
    [-4.041069770525555, 0.5206490038702324, -0.2263727198354415], 
    [-4.041069770525555, -0.5208086614526061, -0.2263727198354415],
    [-5.958820461685046, -0.5208086614526061, 0.22634680717540345],
    [-5.958820461685046, 0.5206490038702324, 0.22634680717540345],
])

n = np.cross(rect1_3d[1] - rect1_3d[0], rect1_3d[2] - rect1_3d[1])
n /= np.linalg.norm(n)
print("CPU heliostat normal:", n)

# CPU: Same setup, but sun position changed before heliostat created, so aim is towards receiver
# Minimum x: -6.0360249272580795, Maximum x: -3.9655048626598592
# Minimum y: -0.7018219867217613, Maximum y: 0.7034613371243075
# Minimum z: -0.23118048365033503, Maximum z: 0.23123336050404472

# rect2_3d = np.array([
#     [-3.9655048626598592, 0.7034613371243075, -0.23118048365033503], 
#     [-3.9655048626598592, -0.7018219867217613, -0.23118048365033503],
#     [-6.0360249272580795, -0.7018219867217613, 0.23123336050404472],
#     [-6.0360249272580795, 0.7034613371243075, 0.23123336050404472],
# ])

# plot each of the rectangles defined by each of their 4 corners in 3d space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# draw line given two points in 3d space
def draw_line(p1, p2, color):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color)

# draw the GPU heliostat rectangle
draw_line(rect1_3d[0], rect1_3d[1], 'b')
draw_line(rect1_3d[1], rect1_3d[2], 'b')
draw_line(rect1_3d[2], rect1_3d[3], 'b')
draw_line(rect1_3d[3], rect1_3d[0], 'b')

# draw the CPU heliostat rectangle
draw_line(rect2_3d[0], rect2_3d[1], 'r')
draw_line(rect2_3d[1], rect2_3d[2], 'r')
draw_line(rect2_3d[2], rect2_3d[3], 'r')
draw_line(rect2_3d[3], rect2_3d[0], 'r')

plt.show()


# Now compute local x, y dimensions for each rectangle:
gpu_x_dim = np.linalg.norm(rect1_3d[1] - rect1_3d[0])
gpu_y_dim = np.linalg.norm(rect1_3d[3] - rect1_3d[0])

cpu_x_dim = np.linalg.norm(rect2_3d[1] - rect2_3d[0])
cpu_y_dim = np.linalg.norm(rect2_3d[3] - rect2_3d[0])

print("GPU heliostat local X dimension:", gpu_x_dim)
print("GPU heliostat local Y dimension:", gpu_y_dim)

print("CPU heliostat local X dimension:", cpu_x_dim)
print("CPU heliostat local Y dimension:", cpu_y_dim)

print("ΔX =", cpu_x_dim - gpu_x_dim)
print("ΔY =", cpu_y_dim - gpu_y_dim)

