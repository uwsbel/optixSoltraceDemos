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
                // float as uint? is this a boolean? 
                optixReportIntersection( t, 0, float3_as_args( n ), __float_as_uint( a1 ), __float_as_uint( a2 ) );
            }
        }
    }
}

extern "C" __global__ void __intersection__cylinder_y()
{
    // Load shader binding table (SBT) and access data specific to this hit group
    const soltrace::HitGroupData* sbt_data = reinterpret_cast<soltrace::HitGroupData*>(optixGetSbtDataPointer());
    const GeometryData::Cylinder_Y& cyl = sbt_data->geometry_data.getCylinder_Y();

    // Get ray information: origin, direction, and min/max distances over which ray should be tested
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = normalize(optixGetWorldRayDirection());
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    // Transform ray to the cylinder's local coordinate system
    float3 local_ray_orig = ray_orig - cyl.center;
    float3 local_ray_dir = ray_dir;

	// TODO: check how to optimize this, there should be a way in optix to rotate coordinates 
    float3 local_x = cyl.base_x;
    float3 local_z = cyl.base_z;
    float3 local_y = cross(local_z, local_x);

    local_ray_orig = make_float3(
        dot(local_ray_orig, local_x),
        dot(local_ray_orig, local_y),
        dot(local_ray_orig, local_z)
    );
    local_ray_dir = make_float3(
        dot(local_ray_dir, local_x),
        dot(local_ray_dir, local_y),
        dot(local_ray_dir, local_z)
    );

	// solve quadratic equation for intersection
    float A = local_ray_dir.x * local_ray_dir.x + local_ray_dir.z * local_ray_dir.z;
    float B = 2.0f * (local_ray_orig.x * local_ray_dir.x + local_ray_orig.z * local_ray_dir.z);
    float C = local_ray_orig.x * local_ray_orig.x + local_ray_orig.z * local_ray_orig.z - cyl.radius * cyl.radius;

    float determinant = B * B - 4.0f * A * C;

    if (determinant < 0.0f)
    {
        // No intersection
        return;
    }

    // Compute intersection distances
    float t1 = (-B - sqrtf(determinant)) / (2.0f * A);
    float t2 = (-B + sqrtf(determinant)) / (2.0f * A);

    float t = t1 > 0.0f ? t1 : t2; // Use the closer valid intersection
    if (t < ray_tmin || t > ray_tmax)
    {
        // Intersection is out of bounds
        return;
    }

    // Compute intersection point in local space
    float3 local_hit_point = local_ray_orig + t * local_ray_dir;

    // Check if the hit point is within the cylinder's height bounds
    if (fabsf(local_hit_point.y) > cyl.half_height)
    {
        // If t1 is invalid, try t2
        t = t2;
        local_hit_point = local_ray_orig + t * local_ray_dir;
        if (t < ray_tmin || t > ray_tmax || fabsf(local_hit_point.y) > cyl.half_height)
        {
            return; // Both intersections are out of bounds
        }
    }

    // Compute normal in local coordinates
    float3 local_normal = normalize(make_float3(local_hit_point.x, 0.0f, local_hit_point.z));

    // Transform normal back to world coordinates
    float3 world_normal = local_normal.x * local_x + local_normal.y * local_y + local_normal.z * local_z;

    // Compute the hit point in world space
    float3 world_hit_point = ray_orig + t * ray_dir;

    // Report intersection to OptiX
    optixReportIntersection(
        t,
        0,
        float3_as_args(world_normal),
        __float_as_uint(world_hit_point.x),
        __float_as_uint(world_hit_point.y)
    );
}

// ray cylinder intersection with top and bottom caps 
// it can also be modeled as cylinder with two disks. 
extern "C" __global__ void __intersection__cylinder_y_capped()
{
    // Load shader binding table (SBT) and access data specific to this hit group
    const soltrace::HitGroupData* sbt_data = reinterpret_cast<soltrace::HitGroupData*>(optixGetSbtDataPointer());
    const GeometryData::Cylinder_Y& cyl = sbt_data->geometry_data.getCylinder_Y();

    // Get ray information: origin, direction, and min/max distances over which ray should be tested
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = normalize(optixGetWorldRayDirection());
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    // Transform ray to the cylinder's local coordinate system
    float3 local_ray_orig = ray_orig - cyl.center;
    float3 local_ray_dir = ray_dir;

    // Transform using the cylinder's local basis
    float3 local_x = cyl.base_x;
    float3 local_z = cyl.base_z;
    float3 local_y = cross(local_z, local_x);

    local_ray_orig = make_float3(
        dot(local_ray_orig, local_x),
        dot(local_ray_orig, local_y),
        dot(local_ray_orig, local_z)
    );
    local_ray_dir = make_float3(
        dot(local_ray_dir, local_x),
        dot(local_ray_dir, local_y),
        dot(local_ray_dir, local_z)
    );

    // Solve quadratic equation for intersection with curved surface
    float A = local_ray_dir.x * local_ray_dir.x + local_ray_dir.z * local_ray_dir.z;
    float B = 2.0f * (local_ray_orig.x * local_ray_dir.x + local_ray_orig.z * local_ray_dir.z);
    float C = local_ray_orig.x * local_ray_orig.x + local_ray_orig.z * local_ray_orig.z - cyl.radius * cyl.radius;

    float determinant = B * B - 4.0f * A * C;

    float t_curved = ray_tmax + 1.0f; // Initialize to invalid
    if (determinant >= 0.0f)
    {
        // Compute intersection distances
        float t1 = (-B - sqrtf(determinant)) / (2.0f * A);
        float t2 = (-B + sqrtf(determinant)) / (2.0f * A);

        // Select the closest valid intersection within bounds
        if (t1 > ray_tmin && t1 < ray_tmax && fabsf(local_ray_orig.y + t1 * local_ray_dir.y) <= cyl.half_height)
        {
            t_curved = t1;
        }
        else if (t2 > ray_tmin && t2 < ray_tmax && fabsf(local_ray_orig.y + t2 * local_ray_dir.y) <= cyl.half_height)
        {
            t_curved = t2;
        }
    }

    // Check intersection with top and bottom caps
    float t_caps = ray_tmax + 1.0f;
    {
        // Bottom cap: y = -half_height
        if (fabsf(local_ray_dir.y) > 1e-6f) // Avoid division by zero
        {
            float t = (-cyl.half_height - local_ray_orig.y) / local_ray_dir.y;
            float2 hit_point = make_float2(local_ray_orig.x + t * local_ray_dir.x,
                local_ray_orig.z + t * local_ray_dir.z);
            if (t > ray_tmin && t < ray_tmax && dot(hit_point, hit_point) <= cyl.radius * cyl.radius)
            {
                t_caps = t;
            }
        }

        // Top cap: y = +half_height
        if (fabsf(local_ray_dir.y) > 1e-6f)
        {
            float t = (cyl.half_height - local_ray_orig.y) / local_ray_dir.y;
            float2 hit_point = make_float2(local_ray_orig.x + t * local_ray_dir.x,
                local_ray_orig.z + t * local_ray_dir.z);
            if (t > ray_tmin && t < ray_tmax && dot(hit_point, hit_point) <= cyl.radius * cyl.radius)
            {
                t_caps = fminf(t_caps, t);
            }
        }
    }

    // Use the closest valid intersection
    float t = fminf(t_curved, t_caps);
    if (t >= ray_tmax || t <= ray_tmin)
    {
        return; // No valid intersection
    }

    // Compute intersection point and normal
    float3 local_hit_point = local_ray_orig + t * local_ray_dir;
    float3 local_normal;

    if (t == t_curved)
    {
        // Hit on the curved surface
        local_normal = normalize(make_float3(local_hit_point.x, 0.0f, local_hit_point.z));
    }
    else
    {
        // Hit on one of the caps
        local_normal = make_float3(0.0f, signbit(local_hit_point.y) ? -1.0f : 1.0f, 0.0f);
    }

    // Transform normal back to world coordinates
    float3 world_normal = local_normal.x * local_x + local_normal.y * local_y + local_normal.z * local_z;

    // Compute world-space hit point
    float3 world_hit_point = ray_orig + t * ray_dir;

    // Report intersection to OptiX
    optixReportIntersection(
        t,
        0, // User-defined instance ID or custom data
        float3_as_args(world_normal),
        __float_as_uint(world_hit_point.x),
        __float_as_uint(world_hit_point.y)
    );
}
