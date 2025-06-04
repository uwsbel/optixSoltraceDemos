import numpy
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, textwrap
import math
import plotly.io as pio
import time

def normal_to_euler(normal, zrot):
    """Convert normal vector to Euler angles"""
    # Ensure normal is normalized
    n = normal / np.linalg.norm(normal)
    
    # Calculate Euler angles
    yaw = math.atan2(n[0], n[2])    # Rotation about Y axis (alpha)
    pitch = math.asin(n[1])         # Rotation about X axis (beta)
    roll = zrot * math.pi/180.0     # Rotation about Z axis (gamma)
    
    return np.array([yaw, pitch, roll])

def get_rotation_matrix_G2L(euler):
    """Get rotation matrix from global to local coordinates"""
    # Convert to radians
    alpha = euler[0]  # yaw 
    beta  = euler[1]  # pitch
    gamma = euler[2]  # zrot
    
    # Precompute sines and cosines
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    cb = math.cos(beta)
    sb = math.sin(beta)
    cg = math.cos(gamma)
    sg = math.sin(gamma)
    
    # Fill in elements of the transformation matrix
    return np.array([
        [ca*cg + sa*sb*sg, -cb*sg, -sa*cg + ca*sb*sg],
        [ca*sg - sa*sb*cg,  cb*cg, -sa*sg - ca*sb*cg],
        [sa*cb,             sb,     ca*cb]
    ])

def local_to_global(point, matrix, origin):
    """Convert point from local to global coordinates"""
    rotated = np.array([
        matrix[0,0]*point[0] + matrix[1,0]*point[1] + matrix[2,0]*point[2],
        matrix[0,1]*point[0] + matrix[1,1]*point[1] + matrix[2,1]*point[2],
        matrix[0,2]*point[0] + matrix[1,2]*point[1] + matrix[2,2]*point[2]
    ])
    
    return rotated + origin

def draw_rectangle(fig, center, normal, width, height, color, zrot=0):
    """Draw a rectangle on the plot"""
    # Create local rectangle points
    local_points = np.array([
        [-width/2, -height/2, 0],
        [width/2, -height/2, 0],
        [width/2, height/2, 0],
        [-width/2, height/2, 0],
        [-width/2, -height/2, 0]  # Close the rectangle
    ])
    
    # Get transformation matrix
    euler = normal_to_euler(normal, zrot)
    matrix = get_rotation_matrix_G2L(euler)
    
    # Transform points to global coordinates
    global_points = np.array([local_to_global(p, matrix, center) for p in local_points])
    
    # Add rectangle to plot
    fig.add_trace(go.Scatter3d(
        x=global_points[:, 0],
        y=global_points[:, 1],
        z=global_points[:, 2],
        mode='lines',
        line=dict(color=color, width=2),
        showlegend=False
    ))

def plot_trace(df, nrays:int = 100000, ntrace:int=100, show_sun_vector:bool=False, save_image:bool=False, image_filename:str=None, display:bool=False):
        """
        Creates and displays a 3D scatter and trace plot.
        
        Parameters
        ------------
        save_image : bool
            Whether to save the plot as an image file
        image_filename : str
            Name of the file to save the image to. If None, will use a default name.
        display : bool
            Whether to display the plot in browser. Set to False when processing multiple frames.
        """
        print("Generating 3D trace plots")
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
        except:
            raise RuntimeError("Missing library: plotly. \n Trace plotting requires the Plotly library to be installed. [$ pip install plotly]")

        # Choose how many points to plot. 
        nn = min(nrays, len(df))
        inds = numpy.random.choice(range(len(df)), size=nn, replace=False)
        
        # Data for a three-dimensional line
        loc_x = df.loc_x.values[inds]
        loc_y = df.loc_y.values[inds]
        loc_z = df.loc_z.values[inds]
        stage = df.stage.values[inds]
        raynum = df.number.values[inds]

        print(f"Plotting {nn} points")
        
        # Basic scene setup with fixed camera
        scene = dict(
            aspectmode='data',
            camera=dict(
                eye=dict(x=3, y=-2, z=0.75),  # Try different values here
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        )
        
        # Generate the 3D scatter plot
        layout = go.Layout(
            scene=scene,
            width=1000,
            height=800,
            margin=dict(l=0, r=0, b=0, t=0)
        )

        if len(list(set(stage))) > 1:
            md = dict( size=0.75, color=stage, colorscale='jet', opacity=0.7, ) 
        else:
            md = dict( size=0.75, color='black', opacity=0.7, ) 

        fig = go.Figure(data=go.Scatter3d(x=loc_x, y=loc_y, z=loc_z, mode='markers', marker=md ), layout=layout )

        # Generate line traces for a subset of randomly selected rays
        print(f"Adding {ntrace} ray traces")
        for i in numpy.random.choice(raynum, size=min(ntrace, len(raynum)), replace=False):
            dfr = df[df.number == i]    #find all rays numbered 'i'
            ray_x = dfr.loc_x 
            ray_y = dfr.loc_y
            ray_z = dfr.loc_z
            fig.add_trace(go.Scatter3d(x=ray_x, y=ray_y, z=ray_z, mode='lines', line=dict(color='black', width=0.5)))

        # Add a trace for the sun vector
        if show_sun_vector:
            tmp = df[df.stage==1].iloc[0]
            sun_vec = numpy.array([-tmp.cos_x,-tmp.cos_y,-tmp.cos_z])
            sunrange = numpy.array([df.loc_x.max()-df.loc_x.min(), df.loc_y.max()-df.loc_y.min(), df.loc_z.max()-df.loc_z.min()])
            sun_vec *= (sunrange*sun_vec).max()
            fig.add_trace(go.Scatter3d(x=[0,sun_vec[0]], y=[0,sun_vec[1]], z=[0,sun_vec[2]], mode='lines', line=dict(color='orange', width=3)))
            fig.add_trace(go.Scatter3d(x=[0,sun_vec[0]], y=[0,sun_vec[1]], z=[0,0], mode='lines', line=dict(color='gray', width=2)))

        fig.update_layout(showlegend=False)
        
        # draw rectangle heliostat
        heliostat_center = np.array([0,5,0])
        heliostat_aimpoint = np.array([0, -17.360680, 94.721360])
        heliostat_normal = heliostat_aimpoint - heliostat_center
        heliostat_normal = heliostat_normal / np.linalg.norm(heliostat_normal)
        width = 1
        height = 1.95
        draw_rectangle(fig, center=heliostat_center, normal=heliostat_normal, width=width, height=height, color='green')

        # draw receiver
        receiver_center = np.array([0, 0, 10])
        receiver_aimpoint = np.array([0, 5, 0])
        receiver_normal = receiver_aimpoint - receiver_center
        receiver_normal = receiver_normal / np.linalg.norm(receiver_normal)
        width = 4
        height = 4
        draw_rectangle(fig, center=receiver_center, normal=receiver_normal, width=width, height=height, color='red')

        # Save image if requested
        if save_image:
            if image_filename is None:
                image_filename = "plot_3d.png"
            print(f"Saving image to {image_filename}")
            pio.write_image(fig, image_filename, width=900, height=900, scale=2)
        
        # Display the figure only if requested
        if display:
            fig.show()
            
        return fig


if __name__ == "__main__":
    # folder directory
    folder_dir = "C:/Users/fang/Documents/NREL_SOLAR/optix/build/bin/Release/"

    for i in range(0, 2):
        print(f"\nProcessing frame {i}...")
        filename = folder_dir + f"output_dynamic_{i}.csv"
        df = pd.read_csv(filename)
        print(f"Loaded data: {len(df)} points")
        
        # Create and save the plot without displaying it
        fig = plot_trace(df, nrays=5000, ntrace=20, save_image=True, image_filename=f"plot_frame_{i}.png", display=False)
        
        # Close the figure to free memory
        fig = None
        
        print(f"Completed frame {i}")
        time.sleep(0.5)  # Small delay between frames