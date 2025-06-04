import re
from collections import defaultdict

def parse_log_file(log_file_path):
    results = defaultdict(lambda: {"ray_launch_times": [], "full_sim_times": []})
    current_sun_rays = None

    with open(log_file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Match the number of sun rays
            sun_rays_match = re.search(r"number of sun rays:\s*(\d+)", line)
            if sun_rays_match:
                current_sun_rays = int(sun_rays_match.group(1))

            # Match execution time for ray launch
            ray_launch_match = re.search(r"Execution time ray launch:\s*(\d+)\s*milliseconds", line)
            if ray_launch_match and current_sun_rays is not None:
                results[current_sun_rays]["ray_launch_times"].append(int(ray_launch_match.group(1)))

            # Match execution time for full sim
            full_sim_match = re.search(r"Execution time full sim:\s*(\d+)\s*milliseconds", line)
            if full_sim_match and current_sun_rays is not None:
                results[current_sun_rays]["full_sim_times"].append(int(full_sim_match.group(1)))

    return results

def calculate_averages(results):
    averages = {}
    for sun_rays, timings in results.items():
        avg_ray_launch = sum(timings["ray_launch_times"]) / len(timings["ray_launch_times"]) if timings["ray_launch_times"] else 0
        avg_full_sim = sum(timings["full_sim_times"]) / len(timings["full_sim_times"]) if timings["full_sim_times"] else 0
        averages[sun_rays] = {"avg_ray_launch_time_ms": avg_ray_launch, "avg_full_sim_time_ms": avg_full_sim}

    sun_rays = sorted(averages.keys())
    avg_ray_launch_times = [averages[r]["avg_ray_launch_time_ms"] for r in sun_rays]
    avg_full_sim_times = [averages[r]["avg_full_sim_time_ms"] for r in sun_rays]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    
    # Plot data
    plt.plot(sun_rays, avg_ray_launch_times, marker='o', linestyle='-', label="Ray Launch Time (ms)")
    plt.plot(sun_rays, avg_full_sim_times, marker='s', linestyle='--', label="Full Sim Time (ms)")

    # Log scale for timing
    plt.yscale("log")
    plt.xscale("log")

    # Labels & Title
    plt.xlabel("Number of Sun Rays")
    plt.ylabel("Execution Time (ms) [Log Scale]")
    plt.title("Execution Time vs Number of Sun Rays")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Show plot
    plt.show()        
    return averages

def main():
    log_file_path = "C:/Users/fang/Documents/NREL_SOLAR/optix/build_debug/bin/Release/timing_results.log"  # Update with your actual log file path
    results = parse_log_file(log_file_path)
    averages = calculate_averages(results)
    
    # Print results
    for sun_rays, avg_timings in averages.items():
        print(f"\nNumber of Sun Rays: {sun_rays}")
        print(f"  Average Ray Launch Time: {avg_timings['avg_ray_launch_time_ms']:.2f} ms")
        print(f"  Average Full Sim Time: {avg_timings['avg_full_sim_time_ms']:.2f} ms")

if __name__ == "__main__":
    main()

