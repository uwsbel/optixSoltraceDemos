import numpy as np
import matplotlib.pyplot as plt

def create_square_points(size=1.0):
    """Create points for a square in the XZ plane"""
    points = np.array([
        [-size/2, 0, -size/2],  # Bottom left
        [size/2, 0, -size/2],   # Bottom right
        [size/2, 0, size/2],    # Top right
        [-size/2, 0, size/2],   # Top left
        [-size/2, 0, -size/2]   # Back to start to close the square
    ])
    return points

def normal_to_euler(normal, zrot=0):
    """Convert normal vector to Euler angles using right-handed system (SolTrace convention)
    
    Args:
        normal: Normalized direction vector (x, y, z)
        zrot: Rotation around Z axis (degrees)
    
    Returns:
        yaw, pitch, roll in degrees
    """
    # Normalize the vector
    normal = normal / np.linalg.norm(normal)
    dx, dy, dz = normal
    
    # SolTrace's conversion (right-handed system)
    yaw = np.degrees(np.arctan2(dx, dz))    # Rotation about Y axis
    pitch = np.degrees(np.arcsin(dy))        # Rotation about X axis
    roll = zrot                              # Rotation about Z axis
    
    return yaw, pitch, roll

def rotate_points(points, yaw, pitch, roll):
    """Rotate points using Euler angles (Tait-Bryan) with Spencer-Murty convention
    
    This implements the Spencer-Murty rotation convention (used by SolTrace):
    - Right-handed coordinate system
    - Left-handed rotations for all angles
    - Yaw (α): rotation about Y axis
    - Pitch (β): rotation about X axis
    - Roll (γ): rotation about Z axis
    
    The transformation sequence:
    1. Start with points in local coordinates
    2. Build global-to-local matrix: R = Rz @ Rx @ Ry
    3. Transpose to get local-to-global matrix: R.T
    4. Transform points from local to global: R.T @ points
    
    Args:
        points: Array of points in local coordinates
        yaw: Rotation about Y axis (degrees)
        pitch: Rotation about X axis (degrees)
        roll: Rotation about Z axis (degrees)
    
    Returns:
        Points transformed to global coordinates
    """
    # Convert to radians
    alpha, beta, gamma = np.radians([yaw, pitch, roll])
    
    # Left-handed rotation matrices (Spencer-Murty convention)
    Ry = np.array([
        [np.cos(alpha), 0, -np.sin(alpha)],  # Y rotation (left-handed)
        [0, 1, 0],
        [np.sin(alpha), 0, np.cos(alpha)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(beta), -np.sin(beta)],    # X rotation (left-handed)
        [0, np.sin(beta), np.cos(beta)]
    ])
    
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],   # Z rotation (left-handed)
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    
    # Build transformation matrix
    # Global-to-local transformation: R = Rz @ Rx @ Ry
    # This matrix transforms points from global coordinates to local coordinates
    R = Rz @ Rx @ Ry
    
    # Local-to-global transformation: R.T
    # Transpose to get the matrix that transforms from local coordinates to global coordinates
    # This is what we want since our points start in local coordinates
    R = R.T
    
    # Print rotation matrix for debugging
    print("Local-to-global rotation matrix:")
    print(R)
    
    # Transform points from local coordinates to global coordinates
    rotated_points = R @ points.T
    return rotated_points.T

def plot_coordinate_systems():
    """Create a visualization of the right-handed coordinate system"""
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create square
    square = create_square_points()
    
    # Example from SolTrace
    # aim_point = np.array([40.7463, -303.375, 295.871])
    # position = np.array([995.877, -316.992, 0])

    aim_point = np.array([17.36068, 0, 94.721360])
    position = np.array([-5, 0, 0])

    # Calculate normal vector (pointing from position to aim point)
    normal = aim_point - position
    normal = normal / np.linalg.norm(normal)
    
    # SolTrace's z-rotation value
    # zrot = -88.75721138871927
    zrot = -90
    
    # Convert normal to Euler angles
    yaw, pitch, roll = normal_to_euler(normal, zrot)
    print(f"Euler angles (degrees): Yaw={yaw:.2f}, Pitch={pitch:.2f}, Roll={roll:.2f}")
    
    # Rotate square
    rotated_square = rotate_points(square, yaw, pitch, roll)
    
    # Plot rotated square
    ax.plot(rotated_square[:, 0], rotated_square[:, 1], rotated_square[:, 2], 'b-', linewidth=2)
    ax.scatter(rotated_square[:, 0], rotated_square[:, 1], rotated_square[:, 2], c='b', marker='*')
    
    # Plot normal vector
    ax.quiver(0, 0, 0, normal[0], normal[1], normal[2], 
             color='g', alpha=0.5, label='Normal')
    
    # Plot coordinate axes
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', alpha=0.5)  # X axis
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.5)  # Y axis
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', alpha=0.5)  # Z axis

    # print rotated square points for debugging
    print("Rotated square points:")
    print(rotated_square)

    
    ax.set_title('Right-Handed Coordinate System\n(SolTrace Convention)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_coordinate_systems() 