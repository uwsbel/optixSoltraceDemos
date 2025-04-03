#include "Aperture.h"
#include "mathUtil.h"
#include "element.h"
#include <optix.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <iostream>
#include <cmath>

class Element;

// Aperture base class implementations
Aperture::Aperture() = default;

double Aperture::get_width() const { return 0.0; }
double Aperture::get_height() const { return 0.0; }
double Aperture::get_radius() const { return 0.0; }

// ApertureRectangle implementations
ApertureRectangle::ApertureRectangle() : x_dim(1.0), y_dim(1.0) {
    m_anchor = make_float3(-0.5, -0.5, 0.0); // anchor at the center of the rectangle
    m_v1 = make_float3(1.0f, 0.0f, 0.0f);   // width vector
    m_v2 = make_float3(0.0f, 1.0f, 0.0f);   // height vector
}

ApertureRectangle::ApertureRectangle(double xDim, double yDim) : x_dim(xDim), y_dim(yDim) {
    m_anchor = make_float3(-xDim / 2.0f, -yDim / 2.0f, 0.0f); // anchor at the center of the rectangle
    m_v1 = make_float3(x_dim, 0.0f, 0.0f);   // width vector
    m_v2 = make_float3(0.0f, y_dim, 0.0f);   // height vector
}

ApertureRectangle::~ApertureRectangle() {}

ApertureType ApertureRectangle::get_aperture_type() const {
    return ApertureType::RECTANGLE;
}

void ApertureRectangle::set_size(double x, double y) {
    x_dim = x;
    y_dim = y;
}

double ApertureRectangle::get_width() const { return x_dim; }
double ApertureRectangle::get_height() const { return y_dim; }
float3 ApertureRectangle::get_anchor() { return m_anchor; }
float3 ApertureRectangle::get_v1() { return m_v1; }
float3 ApertureRectangle::get_v2() { return m_v2; }

void ApertureRectangle::compute_device_aperture(Element* element) {
    // Get the element's position and aim point
    Vector3d pos = element->get_origin();
    Vector3d aim_point = element->get_aim_point();
    
    // Call the other compute_device_aperture implementation
    compute_device_aperture(pos, aim_point);
}

void ApertureRectangle::compute_device_aperture(Vector3d pos, Vector3d aim_point) {
    Vector3d normal = (aim_point - pos).normalized(); // surface normal is the vector from the origin to the aim point

    double nx = normal[0];
    double ny = normal[1];
    double nz = normal[2];

    // compute angles 
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
    assert(fabs(normal_check.x - nx) < 1e-6);
    assert(fabs(normal_check.y - ny) < 1e-6);
    assert(fabs(normal_check.z - nz) < 1e-6);
}

// ApertureCircle implementations
ApertureCircle::ApertureCircle() : radius(1.0) {}
ApertureCircle::ApertureCircle(double r) : radius(r) {}

ApertureType ApertureCircle::get_aperture_type() const {
    return ApertureType::CIRCLE;
}

void ApertureCircle::set_size(double r) { 
    radius = r; 
}

double ApertureCircle::get_radius() const { 
    return radius; 
}

double ApertureCircle::get_width() const { 
    return 2.0 * radius; 
}

double ApertureCircle::get_height() const { 
    return 2.0 * radius; 
}

// ApertureRectangleEasy implementations
ApertureRectangleEasy::ApertureRectangleEasy() : x_dim(1.0), y_dim(1.0) {}
ApertureRectangleEasy::ApertureRectangleEasy(double xDim, double yDim) : x_dim(xDim), y_dim(yDim) {}

ApertureType ApertureRectangleEasy::get_aperture_type() const {
    return ApertureType::EASY_RECTANGLE;
}

void ApertureRectangleEasy::compute_device_aperture(Vector3d pos, Vector3d normal) {
    // For the easy version, we'll just set the origin directly
    m_origin = make_float3(pos[0], pos[1], pos[2]);
    
    // Create a simple coordinate system based on the normal
    // This is a simplified version compared to the full ApertureRectangle implementation
    Vector3d up(0, 0, 1);
    Vector3d right = normal.cross(up).normalized();
    Vector3d new_up = normal.cross(right).normalized();
    
    m_x_axis = make_float3(right[0], right[1], right[2]);
    m_y_axis = make_float3(new_up[0], new_up[1], new_up[2]);
}

void ApertureRectangleEasy::compute_device_aperture(Element* element) {
    // get the origin and euler angles
    Vector3d origin = element->get_origin();
    m_origin = make_float3(origin[0], origin[1], origin[2]);

    // note that euler angles are initialized already when elements are added to the system 
    Matrix33d rotation_matrix = element->get_rotation_matrix();

    // populate m_x_axis and m_y_axis
    m_x_axis = mathUtil::toFloat3(rotation_matrix.get_x_basis());
    m_y_axis = mathUtil::toFloat3(rotation_matrix.get_y_basis());        
}

float3 ApertureRectangleEasy::get_anchor() { return m_origin; }
float3 ApertureRectangleEasy::get_v1() { return m_x_axis; }
float3 ApertureRectangleEasy::get_v2() { return m_y_axis; }
double ApertureRectangleEasy::get_width() const { return x_dim; }
double ApertureRectangleEasy::get_height() const { return y_dim; }
 