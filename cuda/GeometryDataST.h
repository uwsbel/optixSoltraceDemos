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
		CYLINDER_Y            = 1,  
		RECTANGLE_PARABOLIC   = 2,
        UNKNOWN_TYPE          = 3
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

    
	struct Cylinder_Y {
		Cylinder_Y() = default;
        Cylinder_Y(float3 center, float radius, float half_height, float3 base_x, float3 base_z)
            : center(center)
            , radius(radius)
            , half_height(half_height)
            , base_x(base_x)
            , base_z(base_z) {
			assert(dot(base_x, base_z) < 1e-3f);
        }


        float3 center;
        float radius;
		float half_height;
		float3 base_x;   // x axis of the cylinder
		float3 base_z;   // z axis of the cylinder
	};

    struct Rectangle_Parabolic {

		Rectangle_Parabolic() = default;
		Rectangle_Parabolic(float3 v1, float3 v2, float3 anchor, float curv_x, float curv_y)
			: v1(v1)
			, v2(v2)
			, anchor(anchor)
			, curv_x(curv_x)
			, curv_y(curv_y)
		{
			float3 normal = normalize(cross(v1, v2));
			float d = dot(normal, anchor);
			this->v1 *= 1.0f / dot(v1, v1);
			this->v2 *= 1.0f / dot(v2, v2);
			plane = make_float4(normal, d);
		}

		float4 plane;
		float3 v1;
		float3 v2;
		float3 anchor;
		//float3 focus;
        float curv_x;
        float curv_y;
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

    void setCylinder_Y(const Cylinder_Y& c)
    {
        assert(type == UNKNOWN_TYPE);
        type = CYLINDER_Y;
        cylinder_y = c;
    }

    __host__ __device__ const Cylinder_Y& getCylinder_Y() const
    {
        assert(type == CYLINDER_Y);
        return cylinder_y;
    }

	void setRectangleParabolic(const Rectangle_Parabolic& r)
	{
		assert(type == UNKNOWN_TYPE);
		type = RECTANGLE_PARABOLIC;
		rectangle_parabolic = r;
	}

	__host__ __device__ const Rectangle_Parabolic& getRectangleParabolic() const
	{
		assert(type == RECTANGLE_PARABOLIC);
		return rectangle_parabolic;
	}


    Type type = UNKNOWN_TYPE;

    private:
    union
    {
        Parallelogram parallelogram;
		Cylinder_Y cylinder_y;
        Rectangle_Parabolic rectangle_parabolic;
    };
};