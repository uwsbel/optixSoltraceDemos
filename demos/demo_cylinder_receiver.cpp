#include "SolTrSystem.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <optix_function_table_definition.h>

int main(int argc, char* argv[]) {
    //////////////////////////////////////////////////////////////////
    // STEP 0: initialize the ray trace system with number of rays //
    /////////////////////////////////////////////////////////////////
    // number of rays launched for the simulation
    int num_rays = 10000;
    // Create the simulation system.
    SolTrSystem system(num_rays);

    /////////////////////////////////////////////////////////////
    // STEP 1.1 Create heliostat, flat rectangle mirror      //
    ///////////////////////////////////////////////////////////
    Vector3d origin(0, 0, 0); // origin of the element
    Vector3d aim_point(0, 0, 10); // aim point of the element
    auto e1 = std::make_shared<Element>();
    e1->set_origin(origin);
    e1->set_aim_point(aim_point); // Aim direction

    ///////////////////////////////////////////
    // STEP 1.2 create the flat surface     //
    /////////////////////////////////////////
    auto surface = std::make_shared<SurfaceFlat>();
    e1->set_surface(surface);

    ////////////////////////////////////////
    // STEP 1.3 create rectangle aperture //
    //////////////////////////////////////
    double dim_x = 1.0;
    double dim_y = 1.0;
    auto aperture = std::make_shared<ApertureRectangle>(dim_x, dim_y);
    e1->set_aperture(aperture);

    ////////////////////////////////////////////
    // STEP 1.4 Add the element to the system //
    //////////////////////////////////////////
    system.AddElement(e1);

    //////////////////////////////////////////////
    // STEP 2.1 Create receiver, cylinder      //
    /////////////////////////////////////////////
    Vector3d receiver_origin(0, -0.894427, 9.552786); // origin of the receiver
    Vector3d receiver_aim_point(0, 0, -1); // aim point of the receiver
    double receiver_radius = 1.0;  // radius of the cylinder
    double receiver_half_height = 1.2;  // half height of the cylinder

    auto e2 = std::make_shared<Element>();
    e2->set_origin(receiver_origin);
    e2->set_aim_point(receiver_aim_point); // Aim direction

    ///////////////////////////////////////////
    // STEP 2.2 create rectangle aperture    //
    ///////////////////////////////////////////
    auto receiver_aperture = std::make_shared<ApertureRectangle>(receiver_radius * 2., receiver_half_height * 2.);
    e2->set_aperture(receiver_aperture);

    /////////////////////////////////////
    // STEP 2.3 create cylinder surface//
    /////////////////////////////////////
    auto receiver_surface = std::make_shared<SurfaceCylinder>();
    receiver_surface->set_radius(receiver_radius);
    receiver_surface->set_half_height(receiver_half_height);
    e2->set_surface(receiver_surface);

    ////////////////////////////////////////////
    // STEP 2.4 Add the element to the system //
    //////////////////////////////////////////
    system.AddElement(e2); // Add the receiver to the system

    // set up sun vector and angle 
    Vector3d sun_vector(0.0, 0.0, 1.0); // sun vector
    double sun_angle = 0.0; // sun angle

    system.setSunVector(sun_vector);
    system.setSunAngle(sun_angle);

    ///////////////////////////////////
    // STEP 3  Initialize the system //
    ///////////////////////////////////
    system.initialize();

    ////////////////////////////
    // STEP 4  Run Ray Trace  //
    ///////////////////////////
    system.run();

    //////////////////////////
    // STEP 5  Post process //
    /////////////////////////
    system.writeOutput("output_cylinder.csv");

    /////////////////////////////////////////
    // STEP 6  Be a good citizen, clean up //
    /////////////////////////////////////////
    system.cleanup();

    return 0;
}