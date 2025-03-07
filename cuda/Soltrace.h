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

struct BoundingBoxVertex
{
    float       distance;
    float3      point;
};

struct ProjectedPoint
{
    float       buffer;
    float2      point;
};

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_COUNT = 1         // not using occlusion/shadow rays atm
};

struct LaunchParams
{
    unsigned int                width;   // essentially number of rays launched and sun points 
    unsigned int                height;
    int                         max_depth;

    float4*                     hit_point_buffer;
    OptixTraversableHandle      handle;

    float3                      sun_vector;
    float                       max_sun_angle;

    float3                      sun_v0;
    float3                      sun_v1;
    float3                      sun_v2;
    float3                      sun_v3;
};

struct PerRayData
{
    unsigned int ray_path_index;  // Index of the ray in the ray path buffer
    unsigned int depth;           // Trace depth
};

} // end namespace soltrace