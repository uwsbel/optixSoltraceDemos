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
 