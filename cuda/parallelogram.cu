#include <optix.h>
#include <cuda/helpers.h>
#include "Soltrace.h"

extern "C" __global__ void __intersection__parallelogram()
{
    // Load shader binding table (SBT) and access data specific to this hit group
    const soltrace::HitGroupData* sbt_data = reinterpret_cast<soltrace::HitGroupData*>(optixGetSbtDataPointer());
    const GeometryData::Parallelogram& parallelogram = sbt_data->geometry_data.getParallelogram();

    // Get ray information: origin, direction, and min/max distances over which ray should be tested
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();

    // Compute ray intersection point
    float3 n  = make_float3( parallelogram.plane );
    float  dt = dot( ray_dir, n );
    // Compute distance t (point of intersection) along ray direction from ray origin
    float  t  = ( parallelogram.plane.w - dot( n, ray_orig ) ) / dt;

    // Verify intersection distance and Report ray intersection point
    if( t > ray_tmin && t < ray_tmax )
    {
        float3 p  = ray_orig + ray_dir * t;
        float3 vi = p - parallelogram.anchor;
        float  a1 = dot( parallelogram.v1, vi );
        if( a1 >= 0 && a1 <= 1 )
        {
            float a2 = dot( parallelogram.v2, vi );
            if( a2 >= 0 && a2 <= 1 )
            {
                optixReportIntersection( t, 0, float3_as_args( n ), __float_as_uint( a1 ), __float_as_uint( a2 ) );
            }
        }
    }
}