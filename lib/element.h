#ifndef SOLTRACE_ELEMENT_H
#define SOLTRACE_ELEMENT_H

#include <cstdint>
#include <string>

#include "Vector3d.h"
#include "SoltraceType.h"
#include "Surface.h"
#include "Aperture.h"
#include "cuda/GeometryDataST.h"

// Ask John, why element needs a base? difference between ElementBase and Element?
class ElementBase {
public:
    ElementBase() = default;
    virtual ~ElementBase() = default;

    // Positioning and orientation.
    virtual const Vector3d& get_origin() const = 0;
    virtual void set_origin(const Vector3d&) = 0;
    virtual const Vector3d& get_aim_point() const = 0;
    virtual void set_aim_point(const Vector3d&) = 0;
    //virtual const Vector3d& get_euler_angles() const = 0;
    //virtual void set_euler_angles(const Vector3d&) = 0;

    // Bounding box accessors.
    //virtual const Vector3d& get_upper_bounding_box() const = 0;
    //virtual const Vector3d& get_lower_bounding_box() const = 0;


	virtual GeometryData toDeviceGeometryData() const = 0;

protected:
    // Derived classes must implement bounding box computation.
    //virtual int set_bounding_box() = 0;
};

// A concrete implementation of Element that stores data in member variables.
class Element : public ElementBase {
public:
    Element() {

		m_origin = Vector3d(0.0, 0.0, 0.0);
		m_aim_point = Vector3d(0.0, 0.0, 1.0); // Default aim direction
		m_euler_angles = Vector3d(0.0, 0.0, 0.0); // Default orientation

		m_surface = nullptr;
		m_aperture = nullptr;

    }
    ~Element() {}



    // set and get origin 
	const Vector3d& get_origin() const override
	{
		return m_origin;
	}

    void set_origin(const Vector3d& o) override {
        m_origin = o;
    }

    void set_aim_point(const Vector3d& a) override {
		m_aim_point = a;
    }

    const Vector3d& get_aim_point() const override {
		return m_aim_point;
    }


    std::shared_ptr<Aperture> get_aperture() const
    {
        return m_aperture;
    }
    std::shared_ptr<Surface> get_surface() const 
    {
        return m_surface;
    }
    ApertureType get_aperture_type() const 
    {
		return m_aperture->get_aperture_type();
    }
    SurfaceType get_surface_type() const
    {
		return m_surface->get_surface_type();
    }

    // Optical elements setters.
    void set_aperture(const std::shared_ptr<Aperture>& aperture)
    {
        m_aperture = aperture;
    }
    void set_surface(const std::shared_ptr<Surface>& surface)
    {
        m_surface = surface;
    }

	GeometryData toDeviceGeometryData() const
	{
        // get center 
        Vector3d center = m_origin;

		// get aim point
		Vector3d aim = m_aim_point;

		// get surface and aperture
		GeometryData geometry_data;

		SurfaceType surface_type = m_surface->get_surface_type();
		ApertureType aperture_type = m_aperture->get_aperture_type();

		if (aperture_type == ApertureType::RECTANGLE) {

			// compute anchor point, v1 and v2 for rectangle aperture
            double x_dim = m_aperture->get_width();
			double y_dim = m_aperture->get_height();

			m_aperture->compute_device_aperture(m_origin, m_aim_point); // Compute the device aperture based on the origin and aim point

			float3 anchor = m_aperture->get_anchor(); // anchor point
			float3 v1 = m_aperture->get_v1();
			float3 v2 = m_aperture->get_v2();

            // rectangle flat
			if (surface_type == SurfaceType::FLAT) {
				// cast to GeometryData::Parallelogram Type
				GeometryData::Parallelogram heliostat(v1, v2, anchor);
				geometry_data.setParallelogram(heliostat);
            }

            if (surface_type == SurfaceType::PARABOLIC) {
				// create GeometryData::Rectangle_Parabolic 
                GeometryData::Rectangle_Parabolic heliostat(v1, v2, anchor, 
                                                            m_surface->get_curvature_1(),
                                                            m_surface->get_curvature_2());
				geometry_data.setRectangleParabolic(heliostat);
            }
		}
        return geometry_data;
        
	}

private:
    Vector3d m_origin;
    Vector3d m_aim_point;
    Vector3d m_euler_angles;

    Vector3d m_upper_box_bound;
    Vector3d m_lower_box_bound;

    std::shared_ptr<Surface> m_surface;
    std::shared_ptr<Aperture> m_aperture;
};

#endif // SOLTRACE_ELEMENT_H