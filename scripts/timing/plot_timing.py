import matplotlib.pyplot as plt
import numpy as np

# # CPU data, 1700 elements, point-focus ON, [ms]
# # array_<desired num of receiver hits>_<approx. num rays launched>
# array_15500_100000 = [347, 352, 358, 358, 371, 375, 351, 383, 354, 359]
# array_155000_1000000 = [3504, 3571, 3573, 3634, 3621, 3606, 3597, 3622, 3601, 3587]
# array_1550000_10000000 = [34866, 35571, 35616, 35809, 40088, 35753, 37705, 35661, 35815, 35670]

# # GPU data, 30 elements, [ms]
# # 10000 100000 1000000 10000000 100000000 rays launched
# array_10000 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# array_100000 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# array_1000000 = [6, 5, 5, 6, 6, 6, 6, 6, 6, 6]
# array_10000000 = [56, 56, 56, 56, 56, 56, 56, 56, 57, 56, 56]
# array_100000000 = [1303, 1116, 1152, 1169, 1183, 1155, 1197, 1154, 1182, 1161]

# # GPU data, 200 elements, [ms]
# # 10000 100000 1000000 10000000 100000000 rays launched
# # array_10000 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# # array_100000 = [1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
# # array_1000000 = [19, 14, 15, 19, 19, 19, 19, 19, 19, 19]
# # array_10000000 = [182, 181, 183, 183, 183, 183, 183, 183, 183, 183]
# # array_100000000 = [2079, 2167, 2376, 2405, 2413, 2427, 2433, 2421, 2419, 2403]

# array_10000 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# array_100000 = [2, 1, 1, 1, 1, 1, 1, 1, 1, 2]
# array_1000000 = [20, 14, 19, 14, 14, 14, 19, 15, 14, 14]
# array_10000000 = [186, 186, 168, 170, 175, 176, 175, 180, 185, 185]
# array_100000000 = [2262, 2086, 2064, 2102, 2062, 2053, 2075, 2058, 2070, 2101]

# # GPU data, 1700 elements, [ms]
# # 10000 100000 1000000 10000000 100000000 rays launched
# array_10000 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# array_100000 = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
# array_1000000 = [8, 7, 8, 8, 8, 8, 8, 8, 8, 8]
# array_10000000 = [78, 78, 78, 78, 78, 78, 78, 78, 78, 78]
# array_100000000 = [1249, 1281, 1346, 1341, 1316, 1370, 1379, 1373, 1372, 1387]

def plot_cpu_vs_gpu(cpu_data, gpu_data, num_rays, title):
    """
    Plots the comparison between CPU and GPU data for the same number of elements.

    Parameters:
        cpu_data (list): List of CPU timing data.
        gpu_data (list): List of GPU timing data.
        num_rays (list): List of number of rays launched.
        title (str): Title of the plot.
    """
    # Calculate means
    cpu_mean = [np.mean(cpu_data[key]) for key in cpu_data]
    gpu_mean = [np.mean(gpu_data[key]) for key in gpu_data]

    plt.figure(figsize=(10, 6))
    plt.plot(num_rays, cpu_mean, marker='o', label='CPU (1700 elements)')
    plt.plot(num_rays, gpu_mean, marker='s', label='GPU (1700 elements)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Rays Launched')
    plt.ylabel('Ray Trace Execution Time (ms)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.show()

def plot_gpu_comparison(gpu_data_sets, num_rays, labels, title):
    """
    Plots the comparison between GPU data for different element counts.

    Parameters:
        gpu_data_sets (list of dict): List of GPU timing data dictionaries.
        num_rays (list): List of number of rays launched.
        labels (list): List of labels for each GPU data set.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))

    # for gpu_data, label in zip(gpu_data_sets, labels):
    #     # Calculate means
    #     gpu_mean = [np.mean(gpu_data[key]) for key in gpu_data]
    #     plt.plot(num_rays, gpu_mean, marker='o', label=label)

    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Number of Rays Launched')
    # plt.ylabel('Execution Time (ms)')
    # plt.title(title)
    # plt.legend()
    # plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    # plt.show()

    for ray_count in num_rays:
        mean_times = [np.mean(gpu_data[ray_count]) for gpu_data in gpu_data_sets]
        plt.plot(labels, mean_times, marker='o', label=f'{ray_count} Rays')

    plt.yscale('log')
    plt.xlabel('Element Count')
    plt.ylabel('Ray Trace Execution Time (ms)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.show()

# Data Preparation
num_rays = [100000, 1000000, 10000000, 100000000]
num_rays_cpu_compare = [100000, 1000000, 10000000]

# CPU data (1700 elements)
cpu_data_1700 = {
    100000: [347, 352, 358, 358, 371, 375, 351, 383, 354, 359],
    1000000: [3504, 3571, 3573, 3634, 3621, 3606, 3597, 3622, 3601, 3587],
    10000000: [34866, 35571, 35616, 35809, 40088, 35753, 37705, 35661, 35815, 35670]
}

# GPU data (1700 elements)
gpu_data_1700_cpu_compare = {
    100000: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    1000000: [8, 7, 8, 8, 8, 8, 8, 8, 8, 8],
    10000000: [78, 78, 78, 78, 78, 78, 78, 78, 78, 78],
}
gpu_data_1700 = {
    100000: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    1000000: [8, 7, 8, 8, 8, 8, 8, 8, 8, 8],
    10000000: [78, 78, 78, 78, 78, 78, 78, 78, 78, 78],
    100000000: [1249, 1281, 1346, 1341, 1316, 1370, 1379, 1373, 1372, 1387]
}

# GPU data for different element counts
gpu_data_30 = {
    100000: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    1000000: [6, 5, 5, 6, 6, 6, 6, 6, 6, 6],
    10000000: [56, 56, 56, 56, 56, 56, 56, 56, 57, 56, 56],
    100000000: [1303, 1116, 1152, 1169, 1183, 1155, 1197, 1154, 1182, 1161]
}

gpu_data_200 = {
    100000: [2, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    1000000: [20, 14, 19, 14, 14, 14, 19, 15, 14, 14],
    10000000: [186, 186, 168, 170, 175, 176, 175, 180, 185, 185],
    100000000: [2262, 2086, 2064, 2102, 2062, 2053, 2075, 2058, 2070, 2101]
}

# Plotting
plot_cpu_vs_gpu(cpu_data_1700, gpu_data_1700_cpu_compare, num_rays_cpu_compare, "CPU vs GPU (1700 Heliostats)")

plot_gpu_comparison(
    [gpu_data_30, gpu_data_200, gpu_data_1700],
    num_rays,
    ["30", "200", "1700"],
    "GPU Comparison Across Heliostat Field Size"
)