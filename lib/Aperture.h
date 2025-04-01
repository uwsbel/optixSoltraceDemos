#pragma once
#ifndef SOLTRACE_APERTURE_H
#define SOLTRACE_APERTURE_H

#include "SoltraceType.h"  // For ApertureType enum


// Abstract base class for apertures (e.g., rectangle, circle)
class Aperture {
public:
    Aperture() = default;
    virtual ~Aperture() = default;

    virtual ApertureType get_aperture_type() const = 0;

	virtual double get_width() const { return 0.0;} 
	virtual double get_height() const { return 0.0;}
	virtual double get_radius() const { return 0.0;} 

    // interface for defining the size of the aperture for device data
    // TODO, need to not use input here ... 
    // need to think about if multiple element can point to the same aperture. 
    // input is origin and unit vector of the surface normal
    virtual void compute_device_aperture(Vector3d pos, Vector3d normal) = 0; 

    // get anchor, v1 and v2
    virtual float3 get_anchor() = 0;
	virtual float3 get_v1() = 0;
	virtual float3 get_v2() = 0;


};

// Concrete class for a rectangular aperture.
class ApertureRectangle : public Aperture {
public:

    ApertureRectangle() : x_dim(1.0), y_dim(1.0) {
		m_anchor = make_float3(-0.5, -0.5, 0.0); // anchor at the center of the rectangle
		m_v1 = make_float3(1.0f, 0.0f, 0.0f);   // width vector
		m_v2 = make_float3(0.0f, 1.0f, 0.0f);   // height vector
    
    }

	// or initialize with x and y dimensions
	ApertureRectangle(double xDim, double yDim) : x_dim(xDim), y_dim(yDim) {
		m_anchor = make_float3(-xDim / 2.0f, -yDim / 2.0f, 0.0f); // anchor at the center of the rectangle
		m_v1 = make_float3(x_dim, 0.0f, 0.0f);   // width vector
		m_v2 = make_float3(0.0f, y_dim, 0.0f);   // height vector
    
    }
    ~ApertureRectangle() {};

    virtual ApertureType get_aperture_type() const override {
        return ApertureType::RECTANGLE;
    }

    void set_size(double x, double y) {
        x_dim = x;
        y_dim = y;
    }

    double get_width()  const override { return x_dim; }
    double get_height() const override { return y_dim; }
	virtual float3 get_anchor() override{ return m_anchor; } // TODO: should i return float3 instead?   
    virtual float3 get_v1() override { return m_v1; }
    virtual float3 get_v2() override { return m_v2; }

    // TODO: shall i move this to initialization? 
    // called after setting the element position and aimponit
    // rectangle is along x and y axis, where x is the long edge of the rectangle
    // edge 12 is the bottom edge that is parallel to the global xy plane
    // surface normal to the aim point is + z direction
    // anchor point is 2
    // v1 is the vector from 2 to 1 (bottom edge)
    // v2 is the vector from 2 to 3 (left edge)
    // 
    //            ^ +y
    //            |
    //     0 ----------- 3
    //     |      |      |
    // -----------------------> +x
    //     |      |      |
    //     1 ----------- 2
    //            |
	void compute_device_aperture(Vector3d pos, Vector3d aim_point) override {

		Vector3d normal = (aim_point - pos).normalized(); // surface normal is the vector from the origin to the aim point

        double nx = normal[0];
        double ny = normal[1];
        double nz = normal[2];

        // TODO: need utility that deals with all the rotation matrices
        // compete angles 
        double theta = std::acos(nz);
        double phi = std::atan2(nx, ny);

        // Rotation about z by -phi.
        double cos_phi = std::cos(-phi);
        double sin_phi = std::sin(-phi);
        Matrix33d Rz_phi({ cos_phi, -sin_phi, 0.0,
                         sin_phi,  cos_phi, 0.0,
                         0.0,      0.0,     1.0 });


        // Rotation about x by -theta.
        double cos_theta = std::cos(-theta);
        double sin_theta = std::sin(-theta);
        Matrix33d Rx_theta({ 1.0,      0.0,       0.0,
                           0.0, cos_theta, -sin_theta,
                           0.0, sin_theta,  cos_theta });

        Matrix33d R_LTG = Rz_phi * Rx_theta;

		// location of all corners of the rectangle in local coordinates
		Vector3d corner0{ -x_dim / 2.0,  y_dim / 2.0, 0.0 }; // top left
		Vector3d corner1{ -x_dim / 2.0, -y_dim / 2.0, 0.0 }; // bottom left
		Vector3d corner2{  x_dim / 2.0, -y_dim / 2.0, 0.0 }; // bottom right
		Vector3d corner3{  x_dim / 2.0,  y_dim / 2.0, 0.0 }; // top right

        // rotate corners to global coordinates given R_LTG
		Vector3d corner0_global = R_LTG * corner0 + pos;
		Vector3d corner1_global = R_LTG * corner1 + pos;
		Vector3d corner2_global = R_LTG * corner2 + pos;
		Vector3d corner3_global = R_LTG * corner3 + pos;

		// print out global coordinates of corners for debugging 
		std::cout << "corner0_global: " << corner0_global[0] << ", " << corner0_global[1] << ", " << corner0_global[2] << std::endl;
		std::cout << "corner1_global: " << corner1_global[0] << ", " << corner1_global[1] << ", " << corner1_global[2] << std::endl;
     


        // check if bottom edge is indeed at the bottom of the rectangle, if not flip. 
		if (corner2_global[2] > corner3_global[2]) {

            // we need to flip now! 
			std::cout << "FLIP RECTANGLE!!!" << std::endl;

			m_anchor = make_float3(corner0_global[0], corner0_global[1], corner0_global[2]); // anchor at the top right
			m_v1 = make_float3(corner3_global[0] - corner0_global[0],
				               corner3_global[1] - corner0_global[1],
				               corner3_global[2] - corner0_global[2]);   // width vector

			m_v2 = make_float3(corner1_global[0] - corner0_global[0], 
                			   corner1_global[1] - corner0_global[1],
				               corner1_global[2] - corner0_global[2]);   // height vector
		}
        else {
            // no worries, keep with the order
			// TODO: add api for convert vector3d to float3
			m_anchor = make_float3(corner2_global[0], corner2_global[1], corner2_global[2]); // anchor at the bottom left
			m_v1 = make_float3(corner1_global[0] - corner2_global[0],
				               corner1_global[1] - corner2_global[1],
				               corner1_global[2] - corner2_global[2]);   // width vector

			m_v2 = make_float3(corner3_global[0] - corner2_global[0],
				               corner3_global[1] - corner2_global[1],
				               corner3_global[2] - corner2_global[2]);   // height vector

        }

        // print out values 
		std::cout << "Aperture anchor: " << m_anchor.x << ", " << m_anchor.y << ", " << m_anchor.z << std::endl;
		std::cout << "Aperture v1: " << m_v1.x << ", " << m_v1.y << ", " << m_v1.z << std::endl;
		std::cout << "Aperture v2: " << m_v2.x << ", " << m_v2.y << ", " << m_v2.z << std::endl;
        


        // sanity check to make sure crossing m_v1 and m_v2 gives us the normal
		float3 normal_check = normalize(cross(m_v2, m_v1));
        float val = fabs(normal_check.x - nx);
        assert(fabs(normal_check.x - nx) < 1e-6);
		assert(fabs(normal_check.y - ny) < 1e-6);
		assert(fabs(normal_check.z - nz) < 1e-6);



	}

private:
    double x_dim;
    double y_dim;

	// data type for device geometry
    float3 m_anchor;
    float3 m_v1;    // vector along the width
	float3 m_v2;    // vector along the height
};

// Concrete class for a circular aperture.
class ApertureCircle : public Aperture {
public:
    ApertureCircle() : radius(1.0) {}
	// or initialize with radius
	ApertureCircle(double r) : radius(r) {}
    virtual ~ApertureCircle() = default;

    virtual ApertureType get_aperture_type() const override {
        return ApertureType::CIRCLE;
    }

    void set_size(double r) { radius = r; }

    virtual double get_radius() const override { return radius; }
	virtual double get_width() const override { return 2.0 * radius; }
	virtual double get_height() const override { return 2.0 * radius; }

private:
    double radius;
};

#endif // SOLTRACE_APERTURE_H
