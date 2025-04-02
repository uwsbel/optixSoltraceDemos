#pragma once
#ifndef SOLTRACE_APERTURE_H
#define SOLTRACE_APERTURE_H

#include "SoltraceType.h"  // For ApertureType enum
#include <optix.h>
#include <vector_types.h>

// Forward declarations
class Element;
class Vector3d;
class Matrix33d;


// Abstract base class for apertures (e.g., rectangle, circle)
class Aperture {
public:
    Aperture();
    virtual ~Aperture() = default;

    virtual ApertureType get_aperture_type() const = 0;

    virtual double get_width() const;
    virtual double get_height() const;
    virtual double get_radius() const;

    // interface for defining the size of the aperture for device data
    virtual void compute_device_aperture(Vector3d pos, Vector3d normal) = 0; 
    virtual void compute_device_aperture(Element* element) = 0;

    // get anchor, v1 and v2
    virtual float3 get_anchor() = 0;
    virtual float3 get_v1() = 0;
    virtual float3 get_v2() = 0;
};

// Concrete class for a rectangular aperture.
class ApertureRectangle : public Aperture {
public:
    ApertureRectangle();
    ApertureRectangle(double xDim, double yDim);
    ~ApertureRectangle();

    virtual ApertureType get_aperture_type() const override;
    void set_size(double x, double y);
    double get_width() const override;
    double get_height() const override;
    virtual float3 get_anchor() override;
    virtual float3 get_v1() override;
    virtual float3 get_v2() override;
    void compute_device_aperture(Vector3d pos, Vector3d aim_point) override;
    void compute_device_aperture(Element* element) override;

private:
    double x_dim;
    double y_dim;
    float3 m_anchor;
    float3 m_v1;    // vector along the width
    float3 m_v2;    // vector along the height
};

// Concrete class for a circular aperture.
class ApertureCircle : public Aperture {
public:
    ApertureCircle();
    ApertureCircle(double r);
    virtual ~ApertureCircle() = default;

    virtual ApertureType get_aperture_type() const override;
    void set_size(double r);
    virtual double get_radius() const override;
    virtual double get_width() const override;
    virtual double get_height() const override;

private:
    double radius;
};

// Concrete class for an easy rectangular aperture.
class ApertureRectangleEasy : public Aperture {
public:
    ApertureRectangleEasy();
    ApertureRectangleEasy(double xDim, double yDim);
    virtual ~ApertureRectangleEasy() = default;

    virtual ApertureType get_aperture_type() const override;
    void compute_device_aperture(Vector3d pos, Vector3d normal) override;
    void compute_device_aperture(Element* element) override;
    virtual float3 get_anchor() override;
    virtual float3 get_v1() override;
    virtual float3 get_v2() override;
    virtual double get_width() const override;
    virtual double get_height() const override;

private:
    double x_dim;
    double y_dim;
    float3 m_origin;
    float3 m_x_axis;
    float3 m_y_axis;
};


#endif // SOLTRACE_APERTURE_H
