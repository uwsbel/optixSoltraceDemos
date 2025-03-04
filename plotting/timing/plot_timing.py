import matplotlib.pyplot as plt
import numpy as np

# # CPU data, 1700 elements, point-focus ON, [ms]
# # array_<desired num of receiver hits>_<approx. num rays launched>
# array_15500_100000 = [347, 352, 358, 358, 371, 375, 351, 383, 354, 359]
# array_155000_1000000 = [3504, 3571, 3573, 3634, 3621, 3606, 3597, 3622, 3601, 3587]
# array_1550000_10000000 = [34866, 35571, 35616, 35809, 40088, 35753, 37705, 35661, 35815, 35670]

# # GPU data, 30 elements, [ms], GTX 1650
# # 10000 100000 1000000 10000000 100000000 rays launched
# array_10000 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# array_100000 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# array_1000000 = [6, 5, 5, 6, 6, 6, 6, 6, 6, 6]
# array_10000000 = [56, 56, 56, 56, 56, 56, 56, 56, 57, 56, 56]
# array_100000000 = [1303, 1116, 1152, 1169, 1183, 1155, 1197, 1154, 1182, 1161]

# # GPU data, 250 elements, [ms], GTX 1650
# # 10000 100000 1000000 10000000 100000000 rays launched
# array_10000 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# array_100000 = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
# array_1000000 = [8, 6, 6, 6, 6, 6, 6, 6, 6, 6]
# array_10000000 = [75, 74, 74, 74, 74, 75, 75, 75, 75, 75]
# array_100000000 = [1383, 1227, 1185, 1189, 1187, 1171, 1179, 1171, 1188, 1195]

# # GPU data, 1700 elements, [ms], GTX 1650
# # 10000 100000 1000000 10000000 100000000 rays launched
# array_10000 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# array_100000 = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
# array_1000000 = [8, 7, 8, 8, 8, 8, 8, 8, 8, 8]
# array_10000000 = [78, 78, 78, 78, 78, 78, 78, 78, 78, 78]
# array_100000000 = [1249, 1281, 1346, 1341, 1316, 1370, 1379, 1373, 1372, 1387]

# # GPU data, 30 elements, [ms], RTX 4070
# # 10000 100000 1000000 10000000 100000000 rays launched
# array_10000 = [1, 0, 1, 1, 1, 1, 1, 1, 0, 1]
# array_100000 = [1, 0, 1, 0, 1, 1, 1, 0, 1, 1]
# array_1000000 = [2, 2, 2, 2, 2, 2, 2, 3, 2, 2]
# array_10000000 = [19, 18, 18, 18, 18, 18, 17, 18, 18, 18]
# array_100000000 = [185, 179, 179, 179, 179, 178, 178, 180, 178, 181]

# # GPU data, 250 elements, [ms], RTX 4070
# # 10000 100000 1000000 10000000 100000000 rays launched
# array_10000 = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
# array_100000 = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
# array_1000000 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
# array_10000000 = [19, 19, 19, 19, 19, 19, 19, 19, 19, 19]
# array_100000000 = [188, 186, 187, 187, 188, 187, 187, 188, 186, 187]

# # GPU data, 1700 elements, [ms], RTX 4070
# # 10000 100000 1000000 10000000 100000000 rays launched
# array_10000 = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
# array_100000 = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
# array_1000000 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
# array_10000000 = [18, 18, 18, 18, 18, 18, 18, 18, 19, 18]
# array_100000000 = [180, 181, 181, 181, 182, 181, 182, 181, 187, 181]

def plot_cpu_vs_gpu(cpu_data_no_point_focus, cpu_data_point_focus, gpu_data_1650, gpu_data_4070, num_rays, title):
    """
    Plots the comparison between CPU and GPU data for the same number of elements.

    Parameters:
        cpu_data_no_point_focus (list): List of CPU timing data with point focus disabled.
        cpu_data_point_focus (list): List of CPU timing data with point focus enabled.
        gpu_data_1650 (list): List of GTX 1650 timing data.
        gpu_data_4070 (list): List of RTX 4070 timing data.
        num_rays (list): List of number of rays launched.
        title (str): Title of the plot.
    """
    # Calculate means
    cpu_mean_no_point_focus = [np.mean(cpu_data_no_point_focus[key]) for key in cpu_data_no_point_focus]
    cpu_mean_point_focus = [np.mean(cpu_data_point_focus[key]) for key in cpu_data_point_focus]
    gpu_1650_mean = [np.mean(gpu_data_1650[key]) for key in gpu_data_1650]
    gpu_4070_mean = [np.mean(gpu_data_4070[key]) for key in gpu_data_4070]

    print([cpu_mean_no_point_focus[idx]/cpu_mean_point_focus[idx] for idx, x in enumerate(num_rays)])


    # ​​100 * (old - new) / old
    # print([100 * (cpu_mean_point_focus[idx] - gpu_1650_mean[idx])/cpu_mean_point_focus[idx] for idx, x in enumerate(num_rays)])
    print([cpu_mean_no_point_focus[idx]/gpu_1650_mean[idx] for idx, x in enumerate(num_rays)])


    # print([100 * (cpu_mean_point_focus[idx] - gpu_4070_mean[idx])/cpu_mean_point_focus[idx] for idx, x in enumerate(num_rays)])
    print([cpu_mean_no_point_focus[idx]/gpu_4070_mean[idx] for idx, x in enumerate(num_rays)])

    cpu_std_no_point_focus = [np.std(cpu_data_no_point_focus[key]) for key in cpu_data_no_point_focus]
    cpu_std_point_focus = [np.std(cpu_data_point_focus[key]) for key in cpu_data_point_focus]
    gpu_std_1650 = [np.std(gpu_data_1650[key]) for key in gpu_data_1650]
    gpu_std_4070 = [np.std(gpu_data_4070[key]) for key in gpu_data_4070]

    plt.figure(figsize=(10, 6))
    plt.plot(num_rays, cpu_mean_no_point_focus, marker='*', label='CPU i7-9750H (point-focus off)')
    plt.plot(num_rays, cpu_mean_point_focus, marker='o', label='CPU i7-9750H (point-focus on)')
    plt.plot(num_rays, gpu_1650_mean, marker='s', label='GPU GTX 1650')
    plt.plot(num_rays, gpu_4070_mean, marker='^', label='GPU RTX 4070')

    # plt.errorbar(num_rays, cpu_mean_no_point_focus, yerr=cpu_std_no_point_focus, marker='*', label='CPU i7-9750H (point-focus off)', capsize=3)
    # plt.errorbar(num_rays, cpu_mean_point_focus, yerr=cpu_std_point_focus, marker='o', label='CPU i7-9750H (point-focus on)', capsize=3)
    # plt.errorbar(num_rays, gpu_1650_mean, yerr=gpu_std_1650, marker='s', label='GPU GTX 1650', capsize=3)
    # plt.errorbar(num_rays, gpu_4070_mean, yerr=gpu_std_4070, marker='^', label='GPU RTX 4070', capsize=3)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Rays Launched', fontsize=14)
    plt.ylabel('Ray Trace Execution Time (ms)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='best',fontsize=12)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.show()

    # Calculate means
    # cpu_mean = [np.mean(cpu_data[key]) for key in cpu_data]
    # gpu_mean = [np.mean(gpu_data[key]) for key in gpu_data]

    # plt.figure(figsize=(10, 6))
    # plt.plot(num_rays, cpu_mean, marker='o', label='SolTrace CPU')
    # plt.plot(num_rays, gpu_mean, marker='s', label='SolTrace GPU')

    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Number of Rays Launched', fontsize=14)
    # plt.ylabel('Ray Trace Execution Time (ms)', fontsize=14)
    # plt.title(title, fontsize=16)
    # plt.legend(fontsize=14)
    # plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    # plt.show()

def plot_gpu_comparison(gpu_data_sets, num_rays, labels, title):
    """
    Plots the comparison between GPU data for different element counts.

    Parameters:
        gpu_data_sets (dict): Dictionary of timing data.
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
    # plt.xlabel('Number of Rays Launched', fontsize=12)
    # plt.ylabel('Execution Time (ms)', fontsize=12)
    # plt.title(title, fontsize=14)
    # plt.legend(fontsize=12)
    # plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    # plt.show()

    legend_labels = [r"$10^5$ Rays", r"$10^6$ Rays", r"$10^7$ Rays", r"$10^8$ Rays"]
    for i, ray_count in enumerate(num_rays):
        mean_times = [np.mean(gpu_data[ray_count]) for gpu_data in gpu_data_sets]
        plt.plot(labels, mean_times, marker='o', label=legend_labels[i])

    plt.yscale('log')
    plt.xlabel('Number of Heliostats', fontsize=14)
    plt.ylabel('Ray Trace Execution Time (ms)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.show()

#### ============ FLAT RECEIVER DATA ==============================================
# Data Preparation
num_rays = [10**5, 10**6, 10**7, 10**8]
num_rays_cpu_compare = [100000, 1000000, 10000000]

# CPU data (1700 elements)
cpu_data_1700 = {
    100000: [347, 352, 358, 358, 371, 375, 351, 383, 354, 359],
    1000000: [3504, 3571, 3573, 3634, 3621, 3606, 3597, 3622, 3601, 3587],
    10000000: [34866, 35571, 35616, 35809, 40088, 35753, 37705, 35661, 35815, 35670]
}

cpu_data_1700_no_point_focus = {
    100000: [1962, 1976, 1992, 2028, 1988, 2122, 2003, 1984, 2009, 1982],
    1000000: [19720, 20008, 19770, 20053, 19713, 19942, 19740, 19740, 19823, 19741],
    10000000: [196476, 198549, 197752, 197491, 197291, 197184, 198081, 196381, 198350, 197166]
}

# GPU data (1700 elements)
gpu_data_1700_cpu_compare_gtx1650 = {
    100000: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    1000000: [8, 7, 8, 8, 8, 8, 8, 8, 8, 8],
    10000000: [78, 78, 78, 78, 78, 78, 78, 78, 78, 78],
}

gpu_data_1700_cpu_compare_rtx4070 = {
    100000: [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    1000000: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    10000000: [18, 18, 18, 18, 18, 18, 18, 18, 19, 18]
}

# GPU data for different element counts
gpu_data_30_gtx1650 = {
    100000: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    1000000: [6, 5, 5, 6, 6, 6, 6, 6, 6, 6],
    10000000: [56, 56, 56, 56, 56, 56, 56, 56, 57, 56, 56],
    100000000: [1303, 1116, 1152, 1169, 1183, 1155, 1197, 1154, 1182, 1161]
}

gpu_data_250_gtx1650 = {
    100000: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    1000000: [8, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    10000000: [75, 74, 74, 74, 74, 75, 75, 75, 75, 75],
    100000000: [1383, 1227, 1185, 1189, 1187, 1171, 1179, 1171, 1188, 1195]
}

gpu_data_1700_gtx1650 = {
    100000: [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    1000000: [8, 7, 8, 8, 8, 8, 8, 8, 8, 8],
    10000000: [78, 78, 78, 78, 78, 78, 78, 78, 78, 78],
    100000000: [1249, 1281, 1346, 1341, 1316, 1370, 1379, 1373, 1372, 1387]
}

gpu_data_30_rtx4070 = {
    100000: [1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    1000000: [2, 2, 2, 2, 2, 2, 2, 3, 2, 2],
    10000000: [19, 18, 18, 18, 18, 18, 17, 18, 18, 18],
    100000000: [185, 179, 179, 179, 179, 178, 178, 180, 178, 181]
}

gpu_data_250_rtx4070 = {
    100000: [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    1000000: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    10000000: [19, 19, 19, 19, 19, 19, 19, 19, 19, 19],
    100000000: [188, 186, 187, 187, 188, 187, 187, 188, 186, 187]
}

gpu_data_1700_rtx4070 = {
    100000: [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    1000000: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    10000000: [18, 18, 18, 18, 18, 18, 18, 18, 19, 18],
    100000000: [180, 181, 181, 181, 182, 181, 182, 181, 187, 181]
}

#### ============ CYL RECEIVER DATA ==============================================
# Data Preparation
num_rays = [100000, 500000, 1000000, 10000000]
num_rays_cpu_compare = [100000, 1000000, 10000000]

# CPU data (large cylinder scene, 140000 elements)
cpu_data_cyl_large = {
    100000: [546, 528, 536, 585, 541, 526, 544, 522, 537, 538],
    500000: [2593, 2640, 2716, 2652, 2573, 2622, 2532, 2555, 2632, 2621], 
    1000000: [5755, 5864, 5735, 5797, 5713, 5698, 5638, 5674, 5674, 5826, 5831],
    10000000: [58239, 57550, 59084, 57863, 57714, 56868, 56497, 57001, 57326, 58066]
}

cpu_data_cyl_large_no_point_focus = {
    100000: [381802, 295044],
    500000: [1673119, 1437203], 
    1000000: [3088856, 3179246],
    10000000: [32847081]
}

# GPU data (large cylinder scene, 140000 elements)
gpu_data_cyl_large_gtx1650 = {
    100000: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    500000: [7, 7, 7, 7, 7, 7, 7, 7, 7, 7], 
    1000000: [13, 13, 13, 13, 13, 13, 13, 13, 13, 13],
    10000000: [128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
}

gpu_data_cyl_large_rtx4070 = {
    100000: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    500000: [1, 1, 1, 1, 1, 1, 2, 1, 1, 1], 
    1000000: [3, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    10000000: [21, 20, 20, 22, 21, 22, 21, 21, 21, 21]
}



# Plotting
# plot_cpu_vs_gpu(cpu_data_1700_no_point_focus, cpu_data_1700, gpu_data_1700_cpu_compare_gtx1650 , gpu_data_1700_cpu_compare_rtx4070, num_rays_cpu_compare, "SolTrace CPU vs. SolTrace GPU (1700 Elements)")
# plot_cpu_vs_gpu(cpu_data_1700, gpu_data_1700_cpu_compare, num_rays_cpu_compare, "SolTrace CPU vs. SolTrace GPU (1700 Heliostats)")
plot_cpu_vs_gpu(cpu_data_cyl_large_no_point_focus, cpu_data_cyl_large, gpu_data_cyl_large_gtx1650, gpu_data_cyl_large_rtx4070, num_rays, "SolTrace CPU vs. SolTrace GPU (140,000 heliostats, Cylindrical Receiver)")

# plot_gpu_comparison(
#     [gpu_data_30_rtx4070, gpu_data_250_rtx4070, gpu_data_1700_rtx4070],
#     num_rays,
#     ["30", "250", "1700"],
#     "Heliostat Field Scaling (RTX 4070 GPU)"
# )