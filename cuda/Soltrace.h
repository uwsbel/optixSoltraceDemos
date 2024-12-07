#pragma once

#include <vector_types.h>

#include <cuda/BufferView.h>

#include <cuda/GeometryDataST.h>
#include <cuda/MaterialDataST.h>

namespace soltrace
{
const unsigned int NUM_ATTRIBUTE_VALUES = 4u;
const unsigned int NUM_PAYLOAD_VALUES   = 2u;
const unsigned int MAX_TRACE_DEPTH      = 5u;
    
struct HitGroupData
{
    GeometryData geometry_data;
    MaterialData material_data;
};

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_COUNT = 1         // not using occlusion/shadow rays atm
};

struct LaunchParams
{
    unsigned int                width;
    unsigned int                height;
    int                         max_depth;

    float4*                     hit_point_buffer;
    float4*                     reflected_dir_buffer;
    OptixTraversableHandle      handle;

    float3                      sun_center;
    float                       sun_radius;
    float                       max_sun_angle;
    int                         num_sun_points;
    float3                      scene_position;
    OptixAabb                   scene_aabb;
};

struct PerRayData
{
    unsigned int ray_path_index;  // Index of the ray in the ray path buffer
    unsigned int depth;           // Trace depth
};

} // end namespace soltrace