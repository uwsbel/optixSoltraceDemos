#include <optix.h>
#include <curand_kernel.h>
#include <vector_types.h>

#include "helpers.h"
#include "random.h"
#include "Soltrace.h"

// Launch parameters for soltrace
extern "C" {
    __constant__ soltrace::LaunchParams params;
}

// Halton sequence generator, used for quasi-random sampling
// Generates a Halton sequence value for a given index and base
__device__ float halton(int index, int base) {
    float f = 1.0f, result = 0.0f;
    while (index > 0) {
        f = f / base;
        result = result + f * (index % base);
        index = index / base;
    }
    return result;
}

// Generate a sample point within a parallelogram defined by the AABB (Axis-Aligned Bounding Box)
// Uses the Halton sequence for sampling
__device__ float2 haltonSampleInParallelogram(OptixAabb aabb, int sample_index) {
    // Generate Halton sequence values
    float u = halton(sample_index, 2); // Base 2 for x
    float v = halton(sample_index, 3); // Base 3 for y

    // Map the Halton values to the parallelogram
    float sampled_x = aabb.minX + u * (aabb.maxX - aabb.minX);
    float sampled_y = aabb.minY + v * (aabb.maxY - aabb.minY);

    return make_float2(sampled_x, sampled_y);
}

// Generate a random sample point within a parallelogram using a random number generator (RNG)
// The parallelogram is defined by an AABB (Axis-Aligned Bounding Box)
__device__ float2 samplePointInParallelogram(OptixAabb aabb, unsigned int seed) {
    curandState rng_state;
    curand_init(seed, 0, 0, &rng_state);

    // Generate random values between 0 and 1
    float u = curand_uniform(&rng_state);
    float v = curand_uniform(&rng_state);

    // Interpolate between the bounds of the parallelogram
    float x = aabb.minX + u * (aabb.maxX - aabb.minX);
    float y = aabb.minY + v * (aabb.maxY - aabb.minY);

    // Return the sampled point
    return make_float2(x, y);
}

// Generate a random point within a disk with a given radius
// Uses polar coordinates (r, theta) for sampling
__device__ float2 samplePointInDisk(float radius, unsigned int seed) {
    curandState rng_state;
    curand_init(seed, 0, 0, &rng_state);

    // Generate random radius and angle values
    float r = radius * sqrtf(curand_uniform(&rng_state));   
    float theta = 2.0f * M_PIf * curand_uniform(&rng_state);

    // Convert to Cartesian coordinates
    return make_float2(r * cosf(theta), r * sinf(theta));
}

// Sample a random ray direction within a cone defined by a maximum angle
__device__ float3 sampleRayDirection(float max_angle, unsigned int seed) {
    curandState rng_state;
    curand_init(seed, 0, 0, &rng_state);

    // Sample a random angle within the cone's angular spread
    float angle = max_angle * curand_normal(&rng_state);   // Random angle within max angular spread
    float phi = 2.0f * M_PIf * curand_normal(&rng_state);  // Random azimuthal angle

    // Convert spherical coordinaes to Cartesian for ray direction
    float x = sinf(angle) * cosf(phi);
    float y = sinf(angle) * sinf(phi);
    float z = -cosf(angle); // Z is negative to ensure the cone is pointing downward (towards the scene)

    return normalize(make_float3(x, y, z));
}

// == Ray Generation Program - Sun Source (Parallelogram Sampling)
extern "C" __global__ void __raygen__sun_source()
{
    // Lookup location in launch grid
    const uint3 launch_idx = optixGetLaunchIndex();         // Index of the current launch thread
    const uint3 launch_dims = optixGetLaunchDimensions();   // Dimensions of the launch grid
    const unsigned int ray_number = launch_idx.y * launch_dims.x + launch_idx.x;  // Unique ray ID

    // TODO: add a buffer around smallest AABB
    float2 sun_sample_pos = haltonSampleInParallelogram(params.scene_aabb, ray_number);

    // Sample emission angle here - capturing sun distribution
    // TODO: this is assuming the sun is directly above the scene (sun vector (0, 0, height)) to avoid projections for now
    const float3 ray_gen_pos = params.sun_center + make_float3(sun_sample_pos.x, sun_sample_pos.y, 0.0f);
    float3 initial_ray_dir = normalize(make_float3(sun_sample_pos.x, sun_sample_pos.y, 0.0f) - ray_gen_pos);
    // Add some angular variation to the ray direction to simulate the sun's spread
    float3 ray_dir = initial_ray_dir + sampleRayDirection(params.max_sun_angle, launch_idx.x);

    // Create the PerRayData structure to track ray state (e.g., path index and recursion depth)
    soltrace::PerRayData prd;
    prd.ray_path_index = ray_number;
    prd.depth = 0;

    /*
    params.hit_point_buffer[params.max_depth * prd.ray_path_index] = make_float4(0.0f, ray_gen_pos);
    */

    // Cast and trace the ray through the scene
    optixTrace(
        params.handle,               // Acceleration structure handle
        ray_gen_pos,                 // Ray origin
        ray_dir,                     // Ray direction
        0.001f,                      // Minimum ray distance (near hit distance)
        1e16f,                       // Maximum ray distance (far hit distance)
        0.0f,                        // Time parameter (static for now)
        OptixVisibilityMask(1),      // Visibility mask (e.g., to restrict ray interactions)
        OPTIX_RAY_FLAG_NONE,         // Ray flags (no special flags)
        soltrace::RAY_TYPE_RADIANCE, // Ray type (radiance for sunlight)
        soltrace::RAY_TYPE_COUNT,    // Number of ray types
        soltrace::RAY_TYPE_RADIANCE, // SBT offset (ray type to launch)
        reinterpret_cast<unsigned int&>(prd.ray_path_index),
        reinterpret_cast<unsigned int&>(prd.depth)  
    );
}

/*
// == Ray Generation Program - Sun disk
extern "C" __global__ void __raygen__sun_source()
{
    // Lookup location in launch grid here
    const uint3 launch_idx = optixGetLaunchIndex();
    const uint3 launch_dims = optixGetLaunchDimensions();
    const unsigned int ray_number = launch_idx.y * launch_dims.x + launch_idx.x;
    const unsigned int seed = launch_idx.x; // Use launch index to seed RNG for unique sampling

    float2 sun_sample_pos = samplePointInDisk(params.sun_radius, seed);

    // Sample emission angle here - capturing sun distribution
    // TODO need to update for sun's position angle
    const float3 ray_gen_pos = params.sun_center + make_float3(sun_sample_pos.x, sun_sample_pos.y, 0.0f);

    //float3 initial_ray_dir = normalize(params.scene_position - ray_gen_pos);
    float3 initial_ray_dir = normalize(make_float3(sun_sample_pos.x, sun_sample_pos.y, 0.0f) - ray_gen_pos);
    float3 ray_dir = initial_ray_dir + sampleRayDirection(params.max_sun_angle, seed);

    soltrace::PerRayData prd;
    prd.ray_path_index = ray_number;
    prd.depth = 0;

    params.hit_point_buffer[params.max_depth * prd.ray_path_index] = make_float4(1.0f, ray_gen_pos);

    // Cast and trace the ray through the scene
    optixTrace(
        params.handle,
        ray_gen_pos,
        ray_dir,
        0.001f,
        1e16f,
        0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        soltrace::RAY_TYPE_RADIANCE,
        soltrace::RAY_TYPE_COUNT,
        soltrace::RAY_TYPE_RADIANCE,
        reinterpret_cast<unsigned int&>(prd.ray_path_index),
        reinterpret_cast<unsigned int&>(prd.depth)
    );
}
*/


