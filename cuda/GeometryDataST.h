#pragma once

#include <cuda/BufferView.h>

#include <sutil/vec_math.h>

#ifndef __CUDACC_RTC__
#include <cassert>
#else
#define assert(x) /*nop*/
#endif

// unaligned equivalent of float2
struct Vec2f
{
    __host__ __device__ operator float2() const { return { x, y }; }

    float x, y;
};

struct Vec4f
{
    __host__ __device__ operator float4() const { return { x, y, z, w }; }

    float x, y, z, w;
};

struct GeometryData
{
    enum Type
    {
        PARALLELOGRAM         = 0,
        UNKNOWN_TYPE          = 1
    };

    struct Parallelogram
    {
        Parallelogram() = default;
        Parallelogram( float3 v1, float3 v2, float3 anchor )
            : v1( v1 )
            , v2( v2 )
            , anchor( anchor )
        {
            float3 normal = normalize( cross( v1, v2 ) );
            float  d      = dot( normal, anchor );
            this->v1 *= 1.0f / dot( v1, v1 );
            this->v2 *= 1.0f / dot( v2, v2 );
            plane = make_float4( normal, d );
        }
        float4 plane;
        float3 v1;
        float3 v2;
        float3 anchor;
    };

    GeometryData() {};

    void setParallelogram( const Parallelogram& p )
    {
        assert( type == UNKNOWN_TYPE );
        type          = PARALLELOGRAM;
        parallelogram = p;
    }

    __host__ __device__ const Parallelogram& getParallelogram() const
    {
        assert( type == PARALLELOGRAM );
        return parallelogram;
    }

    Type type = UNKNOWN_TYPE;

    private:
    union
    {
        Parallelogram parallelogram;
    };
};