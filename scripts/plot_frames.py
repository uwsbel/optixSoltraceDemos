# plotting_matplotlib.py  -------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                  # noqa: F401
import math, pathlib, time

# ---------- rectangle helpers (unchanged) --------------------------
def normal_to_euler(normal, zrot):
    n = normal / np.linalg.norm(normal)
    yaw   = math.atan2(n[0], n[2])
    pitch = math.asin(n[1])
    roll  = zrot * math.pi / 180.0
    return np.array([yaw, pitch, roll])

def get_rotation_matrix_G2L(euler):
    a, b, g = euler
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cg, sg = math.cos(g), math.sin(g)
    return np.array([
        [ca*cg + sa*sb*sg, -cb*sg, -sa*cg + ca*sb*sg],
        [ca*sg - sa*sb*cg,  cb*cg, -sa*sg - ca*sb*cg],
        [sa*cb,             sb,     ca*cb]
    ])

def local_to_global(p, m, origin):       # p: (x,y,z)
    return origin + m.T @ p

def draw_rectangle(ax, center, normal, width, height, color='k', zrot=0):
    localsq = np.array([
        [-width/2, -height/2, 0],
        [ width/2, -height/2, 0],
        [ width/2,  height/2, 0],
        [-width/2,  height/2, 0],
        [-width/2, -height/2, 0]
    ])
    R = get_rotation_matrix_G2L(normal_to_euler(normal, zrot))
    globalsq = np.array([local_to_global(p, R, center) for p in localsq])
    ax.plot(globalsq[:,0], globalsq[:,1], globalsq[:,2], lw=2, c=color)

# ----------------- plotting routine --------------------------------
N_RAYS   = 100000
N_TRACE  = 200
CAM_ELEV = 11.8            # degrees  (from eye -> elev  = asin(z/r))
CAM_AZIM = -15.7           # degrees  (from eye -> azim = atan2(y,x))
FIGSIZE  = (8, 8)
DPI      = 200

def plot_trace(df: pd.DataFrame, png_name: str,
               nrays=N_RAYS, ntrace=N_TRACE, show_sun=False):
    nn   = min(nrays, len(df))
    inds = np.random.choice(len(df), nn, replace=False)

    x, y, z = df.loc_x.values[inds], df.loc_y.values[inds], df.loc_z.values[inds]
    stage   = df.stage.values[inds]
    ray_id  = df.number.values[inds]

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax  = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=CAM_ELEV, azim=CAM_AZIM)
    ax.set_box_aspect([1,1,1])

    # scatter -------------------------------------------------------
    if len(np.unique(stage)) > 1:
        cmap_vals = stage / stage.max()
        colors = plt.cm.jet(cmap_vals)
    else:
        colors = 'k'
    ax.scatter(x, y, z, s=1, c=colors, alpha=0.7)

    # ray traces ----------------------------------------------------
    chosen = np.random.choice(np.unique(ray_id), size=min(ntrace, len(np.unique(ray_id))), replace=False)
    # print out num of rays to be plotted
    print(f"Plotting {len(chosen)} traces out of {len(np.unique(ray_id))} total rays")
    for rid in chosen:
        seg = df[df.number == rid]
        ax.plot(seg.loc_x, seg.loc_y, seg.loc_z, lw=0.4, c='k')


    # sun vector ----------------------------------------------------
    if show_sun:
        tmp = df[df.stage == 1].iloc[0]
        sun = -np.array([tmp.cos_x, tmp.cos_y, tmp.cos_z])
        extent = max(df.loc_x.max()-df.loc_x.min(),
                     df.loc_y.max()-df.loc_y.min(),
                     df.loc_z.max()-df.loc_z.min())
        sun *= extent * 0.8
        ax.plot([0, sun[0]], [0, sun[1]], [0, sun[2]],
                lw=3, color='orange')

    # rectangles ----------------------------------------------------
    heliostat_center  = np.array([0, 5, 0])
    heliostat_aim     = np.array([0, -17.360680, 94.721360])
    heliostat_normal  = (heliostat_aim - heliostat_center)
    heliostat_normal /= np.linalg.norm(heliostat_normal)
    draw_rectangle(ax, heliostat_center, heliostat_normal,
                   width=1, height=1.95, color='green')

    receiver_center  = np.array([0, 0, 10.])
    receiver_aim     = np.array([0, 5., 0])
    receiver_normal  = receiver_aim - receiver_center
    receiver_normal /= np.linalg.norm(receiver_normal)
    draw_rectangle(ax, receiver_center, receiver_normal,
                   width=2, height=4, color='red')

    # labels / save -------------------------------------------------
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.tight_layout()
    
    
    plt.title(f"Frame {png_name.split('_')[1].split('.')[0]}")
    # specifcy x y z limits
    ax.set_ylim([-2.5, 7])
    ax.set_xlim([-2, 2])
    ax.set_zlim([-1.5, 12])
    
    # plt.show()
    # plt.savefig(png_name, dpi=DPI, bbox_inches='tight')
    # need to make sure figure pixel size are divisible by 2
    fig.savefig(png_name, dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {png_name}")



# ------------- batch driver ----------------------------------------
if __name__ == "__main__":
    folder = pathlib.Path(r"C:/Users/fang/Documents/NREL_SOLAR/optix/cmake_refactor/build/bin/Release")
    for idx in range(42):
        print(f"Processing frame {idx} …")
        df = pd.read_csv(folder / f"output_dynamic_{idx}.csv")
        plot_trace(df, f"frame_{idx}.png")

    # use ffmpeg to create a video 
    import subprocess
    video_name = "output_video.mp4"
    cmd = [
        "ffmpeg", "-y", "-framerate", "4", "-i",
        str("frame_%d.png"), "-c:v", "libx264",
        "-pix_fmt", "yuv420p", video_name
    ]

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
