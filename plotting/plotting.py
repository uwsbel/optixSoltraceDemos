import numpy
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint

def plot_trace(df, nrays:int = 100000, ntrace:int=100, show_sun_vector:bool=False):
        """
        Creates and (optionally) displays a 3D scatter and trace plot. This
        function requires that the Python package `plotly` be installed. 

        Parameters
        ------------
        nrays : int
            Number of individual rays to include in the scatter plot. Very 
            large values may render slowly.
        ntrace : int 
            Number of rays for which traces will be displayed. Large values
            may render slowly
        show_sun_vector : bool
            Flag indicating whether the sun vector should be rendered on the plot
        """

        print("Generating 3D trace plots")
        # Plotting with plotly
        try:
            import plotly.graph_objects as go
        except:
            raise RuntimeError("Missing library: plotly. \n Trace plotting requires the Plotly library to be installed. [$ pip install plotly]")

        # Choose how many points to plot. 
        nn = min(nrays, len(df))
        inds = numpy.random.choice(range(len(df)), size=nn, replace=False)
        
        # Data for a three-dimensional line. Randomly choose points if fewer than the full amount are desired.
        loc_x = df.loc_x.values[inds]
        loc_y = df.loc_y.values[inds]
        loc_z = df.loc_z.values[inds]
        stage = df.stage.values[inds]
        raynum = df.number.values[inds]

        # Generate the 3D scatter plot
        layout = go.Layout(scene=dict(aspectmode='data'))

        if len(list(set(stage))) > 1:
            md = dict( size=0.75, color=stage, colorscale='jet', opacity=0.7, ) 
        else:
            md = dict( size=0.75, color='black', opacity=0.7, ) 

        fig = go.Figure(data=go.Scatter3d(x=loc_x, y=loc_y, z=loc_z, mode='markers', marker=md ), layout=layout )

        # Generate line traces for a subset of randomly selected rays
        for i in numpy.random.choice(raynum, size=200, replace=False):
            dfr = df[df.number == i]    #find all rays numbered 'i'
            ray_x = dfr.loc_x 
            ray_y = dfr.loc_y
            ray_z = dfr.loc_z
            fig.add_trace(go.Scatter3d(x=ray_x, y=ray_y, z=ray_z, mode='lines', line=dict(color='black', width=0.5)))
        # Add a trace for the sun vector
        if show_sun_vector:
            tmp = df[df.stage==1].iloc[0]  #sun is coming from the cos vector of the elements in the first stage. Just take the first.
            sun_vec = numpy.array([-tmp.cos_x,-tmp.cos_y,-tmp.cos_z])  #negative of the vector
            # scale the vector based on the overall size of the sun bounding box
            sunrange = numpy.array([df.loc_x.max()-df.loc_x.min(), df.loc_y.max()-df.loc_y.min(), df.loc_z.max()-df.loc_z.min()])
            # sun_scale = ((self.sunstats['xmax']-self.sunstats['xmin'])**2 + (self.sunstats['ymax']-self.sunstats['ymin'])**2)**.5 *0.75
            # sun_scale = min([sun_scale, df.loc_x.max()])
            sun_vec *= (sunrange*sun_vec).max()
            fig.add_trace(go.Scatter3d(x=[0,sun_vec[0]], y=[0,sun_vec[1]], z=[0,sun_vec[2]], mode='lines', line=dict(color='orange', width=3)))
            fig.add_trace(go.Scatter3d(x=[0,sun_vec[0]], y=[0,sun_vec[1]], z=[0,0], mode='lines', line=dict(color='gray', width=2)))

        fig.update_layout(showlegend=False)
        fig.show()

def plot_yz_view(df, df_gpu):
    """
    Plots hit points in the Y–Z plane.
    """
    plt.figure()
    plt.scatter(df['loc_y'], df['loc_z'], s=2, alpha=0.6)
    plt.scatter(df_gpu['loc_y'], df_gpu['loc_z'], s=2, alpha=0.1, color='red')
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.title("Hit Points in Y–Z Plane")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

def best_fit_plane(points_3d):
    """
    Returns a point on the plane (p0) and a normal vector (n) for the best-fit plane
    through the Nx3 array 'points_3d'.
    """
    centroid = numpy.mean(points_3d, axis=0)
    # Center the points at the centroid
    centered = points_3d - centroid
    # Perform SVD on the centered coordinate matrix
    u, s, vh = numpy.linalg.svd(centered, full_matrices=False)
    # Normal is the last singular vector
    normal = vh[-1,:]
    return centroid, normal

def project_points_onto_plane(points_3d, plane_point, plane_normal):
    """
    Projects the Nx3 array 'points_3d' onto a plane defined by 'plane_point' and 'plane_normal'
    and returns the 3D coordinates of those projected points.
    """
    plane_normal = plane_normal / numpy.linalg.norm(plane_normal)
    projected = []
    for p in points_3d:
        v = p - plane_point
        dist = numpy.dot(v, plane_normal)  # distance along normal
        p_proj = p - dist * plane_normal
        projected.append(p_proj)
    return numpy.array(projected)

def to_local_2d(points_3d, plane_point, plane_normal):
    """
    Creates a local 2D coordinate system (e1, e2) in the plane and returns Nx2 coords.
    """
    plane_normal = plane_normal / numpy.linalg.norm(plane_normal)
    # Pick an arbitrary vector not parallel to plane_normal
    temp = numpy.array([1, 0, 0]) if abs(numpy.dot(plane_normal, [1,0,0])) < 0.99 else numpy.array([0,1,0])
    e1 = numpy.cross(plane_normal, temp)
    e1 /= numpy.linalg.norm(e1)
    e2 = numpy.cross(plane_normal, e1)
    out_2d = []
    for p in points_3d:
        r = p - plane_point
        x_local = numpy.dot(r, e1)
        y_local = numpy.dot(r, e2)
        out_2d.append([x_local, y_local])
    return numpy.array(out_2d)

def compute_area_best_fit_plane(df):
    """
    Finds a best-fit plane for the points in 'df', projects them,
    and returns the 2D convex-hull area on that plane.
    """
    points_3d = df[['loc_x','loc_y','loc_z']].values
    plane_point, plane_normal = best_fit_plane(points_3d)
    proj_3d = project_points_onto_plane(points_3d, plane_point, plane_normal)
    coords_2d = to_local_2d(proj_3d, plane_point, plane_normal)
    
    from shapely.geometry import MultiPoint
    hull = MultiPoint(coords_2d).convex_hull
    return hull.area

if __name__ == "__main__":
    # folder directory
    folder_dir = "C:/Users/allie/Documents/SolTrace/hit_point_data/"

    # filename = folder_dir + "cpu_one_heliostat_aim_at_receiver_heliostat_only.csv"
    # filename = folder_dir + "toyproblem-hit_counts-1000000_rays_with_buffer.csv"
    #filename = folder_dir + "toy_problem_parabolic.csv"
    filename = folder_dir + "cpu_one_heliostat_aim_at_receiver_heliostat_only.csv"
    #filename = folder_dir + "cpu_one_heliostat_heliostat_only.csv"
    filename_gpu = folder_dir + "gpu_one_heliostat_aim_at_receiver_heliostat_only.csv"

    df = pd.read_csv(filename)
    df_gpu = pd.read_csv(filename_gpu)

    area_CPU = compute_area_best_fit_plane(df)
    print("Best-fit plane area of CPU convex hull:", area_CPU)

    area_GPU = compute_area_best_fit_plane(df_gpu)
    print("Best-fit plane area of GPU convex hull:", area_GPU)

    # Split the dataframe into two dataframes based on a loc_y bound
    df_left = df[df['loc_y'] < -0.7]
    print("CPU left z min:", df_left['loc_z'].min())
    df_right = df[df['loc_y'] > 0.265]
    #df_left = df[df['loc_y'] < -0.5]
    #print("CPU left z min:", df_left['loc_z'].min())
    #df_right = df[df['loc_y'] > 0.46]
    print("CPU right z min:", df_right['loc_z'].min())
    print("CPU difference:", df_left['loc_z'].min() - df_right['loc_z'].min())
    df_gpu_left = df_gpu[df_gpu['loc_y'] < -0.69]
    print("GPU left z min:", df_gpu_left['loc_z'].min())
    df_gpu_right = df_gpu[df_gpu['loc_y'] > 0.28]
    print("GPU right z min:", df_gpu_right['loc_z'].min())
    print("GPU difference:", df_gpu_left['loc_z'].min() - df_gpu_right['loc_z'].min())

    # print(df)
    plot_yz_view(df, df_gpu)
    #plot_trace(df, nrays=100000, ntrace=200)