// main.cpp
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
    // STEP 1.1 Create heliostat, parabolic rectangle mirror  //
    ////////////////////////////////////////////////////////////
    Vector3d origin(0, 5, 0); // origin of the element
    Vector3d aim_point(0, 4.551982, 1.897836); // aim point of the element
    auto e1 = std::make_shared<Element>();
    e1->set_origin(origin);
    e1->set_aim_point(aim_point); // Aim direction

    ///////////////////////////////////////////
    // STEP 1.2 create the parabolic surface //
    ///////////////////////////////////////////
    double curv_x = 0.0170679f;
    double curv_y = 0.0370679f;
    auto surface = std::make_shared<SurfaceParabolic>();
    surface->set_curvature(curv_x, curv_y);
    e1->set_surface(surface);

    ////////////////////////////////////////
    // STEP 1.3 create rectangle aperture //
    ////////////////////////////////////////
    double dim_x = 1.0;
    double dim_y = 2.0;
    auto aperture = std::make_shared<ApertureRectangle>(dim_x, dim_y);
    e1->set_aperture(aperture);

    ////////////////////////////////////////////
    // STEP 1.4 Add the element to the system //
    ///////////////////////////////////////////
    system.AddElement(e1);

    //////////////////////////////////////////////
    // STEP 2.1 Create receiver, flat rectangle //
    //////////////////////////////////////////////
    Vector3d receiver_origin(0, 0, 10.0); // origin of the receiver
    Vector3d receiver_aim_point(0, 1.788856, 6.422292); // aim point of the receiver
    double receiver_dim_x = 2.0;
    double receiver_dim_y = 2.0;
    auto e2 = std::make_shared<Element>();
    e2->set_origin(receiver_origin);
    e2->set_aim_point(receiver_aim_point); // Aim direction


    ///////////////////////////////////////////
    // STEP 2.2 create rectangle aperture    //
    ///////////////////////////////////////////
    auto receiver_aperture = std::make_shared<ApertureRectangle>(receiver_dim_x, receiver_dim_y);
    e2->set_aperture(receiver_aperture);

    ///////////////////////////////////
    // STEP 2.3 create flat surface //
    //////////////////////////////////
    auto receiver_surface = std::make_shared<SurfaceFlat>();
    e2->set_surface(receiver_surface);

    ////////////////////////////////////////////
    // STEP 2.4 Add the element to the system //
    ///////////////////////////////////////////
    system.AddElement(e2); // Add the receiver to the system

    ///////////////////////////////////
    // STEP 3  Initialize the system //
    ///////////////////////////////////
    system.initialize();

    ////////////////////////////
    // STEP 4  Run Ray Trace //
    ///////////////////////////
    system.run();

    //////////////////////////
    // STEP 5  Post process //
    //////////////////////////
    system.writeOutput("output_parabolic.csv");

    /////////////////////////////////////////
    // STEP 6  Be a good citizen, clean up //
    /////////////////////////////////////////
    system.cleanup();



    return 0;
}
