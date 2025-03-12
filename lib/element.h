#ifndef SOLTRACE_ELEMENT_H
#define SOLTRACE_ELEMENT_H

#include <cstdint>
#include <string>

#include "vector3d.hpp"
#include "SoltraceType.h"
// TODO: get rid of base

class Element
{
public:

    Element() {};
    virtual ~Element() {};

    // Accessors
    virtual const Vector3d& get_origin() const = 0;
    virtual void set_origin(const Vector3d&) = 0;
    virtual const Vector3d& get_aim_point() const = 0;
    virtual void set_aim_point(const Vector3d&) = 0;
    virtual const Vector3d& get_euler_angles() const = 0;
    virtual void set_euler_angles(const Vector3d&) = 0;

    virtual const Vector3d& get_upper_bounding_box() const = 0;
    virtual const Vector3d& get_lower_bounding_box() const = 0;

    virtual const Shape& get_shape() const = 0;

    //virtual const OpticalProperties& get_optical_properties() const = 0;
    //virtual void set_optical_properties(const OpticalProperties&) = 0;

    //// Other routines
    //virtual int update_orientation(const DateTime&, const RaySource&) = 0;

protected:

    virtual int set_bounding_box() = 0;

private:
};

class ElementBase : public Element
{
public:
    ElementBase();
    virtual ~ElementBase();

    virtual const Vector3d& get_origin() const
    {
        return this->origin;
    }
    virtual void set_origin(const Vector3d& point)
    {
        this->origin = point;
        return;
    }
    virtual const Vector3d& get_aim_point() const
    {
        return this->aim;
    }
    virtual void set_aim_point(const Vector3d& direction)
    {
        this->aim = direction;
        return;
    }
    virtual const Vector3d& get_euler_angles() const
    {
        return this->euler_angles;
    }
    virtual void set_euler_angles(const Vector3d& angles)
    {
        this->euler_angles = angles;
    }

    ////////////////////// optix related ////////////////////////
    // set and get method for aperture and surface type
	void set_aperture_type(ApertureType type)
	{
		this->aperture_type = type;
	}

	ApertureType get_aperture_type() const
	{
		return this->aperture_type;
	}

	void set_surface_type(SurfaceType type)
	{
		this->surface_type = type;
	}

	SurfaceType get_surface_type() const
	{
		return this->surface_type;
	}

private:



    //////////////// from John /////////////////////////////
    Vector3d origin;
    Vector3d aim;
    Vector3d euler_angles;

    //Matrix3d reference_to_local;
    //Matrix3d local_to_reference;

    Vector3d upper_box_bound;
    Vector3d lower_box_bound;

    //OpticalProperties optics;
    //Shape shape;

	ApertureType aperture_type;
	SurfaceType surface_type;

	//////////////////////////////////////////////////////


};

#endif
