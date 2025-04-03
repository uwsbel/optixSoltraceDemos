#ifndef SOLTRACE_ELEMENT_H
#define SOLTRACE_ELEMENT_H

#include <cstdint>
#include <string>

#include "Vector3d.h"
#include "SoltraceType.h"
#include "Surface.h"
#include "Aperture.h"
#include "mathUtil.h"
#include "cuda/GeometryDataST.h"

class Aperture;

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


	virtual GeometryDataST toDeviceGeometryData() const = 0;

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
        m_zrot = 0.0;    
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

    void set_zrot(double zrot) {
        m_zrot = zrot;
    }

    double get_zrot() const {
        return m_zrot;
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

    // set orientation based on aimpoint and zrot
    void update_euler_angles(const Vector3d& aim_point, const double zrot) {
        // compute euler angles from aim point, origin and zror 
        Vector3d normal = aim_point - m_origin;
        normal.normalized();
        // compute euler angles from normal vector
        m_euler_angles = mathUtil::normal_to_euler(normal, zrot);
    }

    void update_euler_angles(){
        Vector3d normal = m_aim_point - m_origin;
        normal.normalized();
        m_euler_angles = mathUtil::normal_to_euler(normal, m_zrot);
    }

    void update_element(const Vector3d& aim_point, const double zrot){
        m_aim_point = aim_point;
        m_zrot = zrot;
        update_euler_angles();
    }
    // return L2G rotation matrix
    Matrix33d get_rotation_matrix() const {
        // get G2L rotation matrix from euler angles 
        // TODO: need to think about if we store this or not
        Matrix33d mat_G2L = mathUtil::get_rotation_matrix_G2L(m_euler_angles);
        return mat_G2L.transpose();
    }


    // return upper bounding box
	Vector3d get_upper_bounding_box() const {
        printf("upper bounding box: %f, %f, %f\n", m_upper_box_bound[0], m_upper_box_bound[1], m_upper_box_bound[2]);
		return m_upper_box_bound;
	}

	// return lower bounding box
	Vector3d get_lower_bounding_box() const {
        printf("lower bounding box: %f, %f, %f\n", m_lower_box_bound[0], m_lower_box_bound[1], m_lower_box_bound[2]);
		return m_lower_box_bound;
	}


	GeometryDataST toDeviceGeometryData() const override
	{
        GeometryDataST geometry_data;

        SurfaceType surface_type = m_surface->get_surface_type();
        ApertureType aperture_type = m_aperture->get_aperture_type();

        if (aperture_type == ApertureType::RECTANGLE) {
            m_aperture->compute_device_aperture(m_origin, m_aim_point);

            float3 anchor = m_aperture->get_anchor();
            float3 v1 = m_aperture->get_v1();
            float3 v2 = m_aperture->get_v2();

            if (surface_type == SurfaceType::FLAT) {
                GeometryDataST::Parallelogram heliostat(v1, v2, anchor);
				geometry_data.setParallelogram(heliostat);

            }

            if (surface_type == SurfaceType::PARABOLIC) {
                GeometryDataST::Rectangle_Parabolic heliostat(v1, v2, anchor, 
                    (float)m_surface->get_curvature_1(),
                    (float)m_surface->get_curvature_2());
                geometry_data.setRectangleParabolic(heliostat);
            }
        }
        
        
        
        if (aperture_type == ApertureType::EASY_RECTANGLE) {
            
            double width = m_aperture->get_width();
            double height = m_aperture->get_height();

			Matrix33d rotation_matrix = get_rotation_matrix();  // L2G rotation matrix

            Vector3d v1 = rotation_matrix.get_x_basis();
			Vector3d v2 = rotation_matrix.get_y_basis();
            
			std::cout << "x basis of heliostat : " << v1[0] << ", " << v1[1] << ", " << v1[2] << std::endl;
			std::cout << "y basis of heliostat : " << v2[0] << ", " << v2[1] << ", " << v2[2] << std::endl;

            GeometryDataST::Rectangle_Flat heliostat(mathUtil::toFloat3(m_origin), mathUtil::toFloat3(v1), mathUtil::toFloat3(v2), (float)width, (float)height);
            geometry_data.setRectangle_Flat(heliostat);

        }
        
        return geometry_data;
	}


    // we also need to implement the bounding box computation
    // for a case like a rectangle aperture,
    // once we have the origin, euler angles, rotatioin matrix
    // and the aperture size, we can compute the bounding box
    // this can be called when adding an element to the system
    void compute_bounding_box(){
        // this can also be called while "initializing" the element
        // get the rotation matrix first
        Matrix33d rotation_matrix = get_rotation_matrix();  // L2G rotation matrix

        // now check the type of the aperture
        ApertureType aperture_type = m_aperture->get_aperture_type();

        if (aperture_type == ApertureType::EASY_RECTANGLE) {
            // get the width and height of the aperture
            double width = m_aperture->get_width();
            double height = m_aperture->get_height();

            // compute the four corners of the rectangle locally
            Vector3d corner1 = Vector3d(-width/2, -height/2, 0.0);
            Vector3d corner2 = Vector3d( width/2, -height/2, 0.0);
            Vector3d corner3 = Vector3d( width/2,  height/2, 0.0);
            Vector3d corner4 = Vector3d(-width/2,  height/2, 0.0);

            // transform the corners to the global frame
            Vector3d corner1_global = rotation_matrix * corner1 + m_origin;
            Vector3d corner2_global = rotation_matrix * corner2 + m_origin;
            Vector3d corner3_global = rotation_matrix * corner3 + m_origin;
            Vector3d corner4_global = rotation_matrix * corner4 + m_origin;

            // now update the bounding box, need to find the min and max x, y, z
            m_lower_box_bound[0] = fmin(fmin(corner1_global[0], corner2_global[0]), fmin(corner3_global[0], corner4_global[0]));
            m_lower_box_bound[1] = fmin(fmin(corner1_global[1], corner2_global[1]), fmin(corner3_global[1], corner4_global[1]));
            m_lower_box_bound[2] = fmin(fmin(corner1_global[2], corner2_global[2]), fmin(corner3_global[2], corner4_global[2]));

            m_upper_box_bound[0] = fmax(fmax(corner1_global[0], corner2_global[0]), fmax(corner3_global[0], corner4_global[0]));
            m_upper_box_bound[1] = fmax(fmax(corner1_global[1], corner2_global[1]), fmax(corner3_global[1], corner4_global[1]));
            m_upper_box_bound[2] = fmax(fmax(corner1_global[2], corner2_global[2]), fmax(corner3_global[2], corner4_global[2]));

            
            
    }
}


private:
    Vector3d m_origin;
    Vector3d m_aim_point;
    Vector3d m_euler_angles;  // euler angles, need to be computed from aim point and zrot
    double m_zrot; // zrot from the stinput file, user provided value, in degrees

    Vector3d m_upper_box_bound;
    Vector3d m_lower_box_bound;

    std::shared_ptr<Surface> m_surface;
    std::shared_ptr<Aperture> m_aperture;

};

#endif // SOLTRACE_ELEMENT_H