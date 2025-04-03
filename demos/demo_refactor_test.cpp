// main.cpp
#include "SolTrSystem.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <optix_function_table_definition.h>


int main(int argc, char* argv[]) {
    bool stinput = true; // Set to true if using stinput file, false otherwise
    bool parabolic = true; // Set to true for parabolic mirrors, false for flat mirrors
    // number of rays launched for the simulation
    int num_rays = 3280833;
    // Create the simulation system.
    SolTrSystem system(num_rays);

    if (stinput) {
        const char* stinput_file = "../data/stinput/small_system_parabolic_heliostats_flat_receiver.stinput"; // Default stinput file name

        if (argc > 1) {
            stinput_file = argv[1]; // Get the stinput file name from command line argument
        }

        if (stinput) {
            // Read the system from the file
            std::cout << "Reading STINPUT file." << std::endl;
            system.readStinput(stinput_file);
        } else {
            std::cout << "Error: System setup failed." << std::endl;
        }
    } else {

        //////////////////////////////////////////////////////////////////
        // STEP 0: initialize the ray trace system with number of rays //
        /////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////
        // STEP 1.1 Create heliostats, parabolic rectangle mirror  //
        ////////////////////////////////////////////////////////////
        // Element 1
        Vector3d origin_e1(-5, 0, 0); // origin of the element
        Vector3d aim_point_e1(17.360680, 0, 94.721360); // aim point of the element
        auto e1 = std::make_shared<Element>();
        e1->set_origin(origin_e1);
        e1->set_aim_point(aim_point_e1); // Aim direction

        ///////////////////////////////////////////
        // STEP 1.2 create the parabolic surface //
        ///////////////////////////////////////////
        double curv_x = 0.0170679f;
        double curv_y = 0.0370679f;
        if (parabolic) {
            auto surface_e1 = std::make_shared<SurfaceParabolic>();
            surface_e1->set_curvature(curv_x, curv_y);
            e1->set_surface(surface_e1);
        } 
        else {
            // Create a flat surface if not parabolic
            auto surface_e1 = std::make_shared<SurfaceFlat>();
            e1->set_surface(surface_e1);
        }


        ////////////////////////////////////////
        // STEP 1.3 create rectangle aperture //
        ////////////////////////////////////////
        double dim_x = 1.0;
        double dim_y = 1.95;
        auto aperture_e1 = std::make_shared<ApertureRectangle>(dim_x, dim_y);
        e1->set_aperture(aperture_e1);

        ////////////////////////////////////////////
        // STEP 1.4 Add the element to the system //
        ///////////////////////////////////////////
        system.AddElement(e1);
        
        // Element 2
        Vector3d origin_e2(0, 5, 0); // origin of the element
        Vector3d aim_point_e2(0, -17.360680, 94.721360); // aim point of the element
        auto e2 = std::make_shared<Element>();
        e2->set_origin(origin_e2);
        e2->set_aim_point(aim_point_e2); // Aim direction

        if (parabolic) {
        auto surface_e2 = std::make_shared<SurfaceParabolic>();
        surface_e2->set_curvature(curv_x, curv_y);
        e2->set_surface(surface_e2);
        } 
        else {
            // Create a flat surface if not parabolic
            auto surface_e2 = std::make_shared<SurfaceFlat>();
            e2->set_surface(surface_e2);
        }

        auto aperture_e2 = std::make_shared<ApertureRectangle>(dim_x, dim_y);
        e2->set_aperture(aperture_e2);

        system.AddElement(e2);

        // Element 3
        Vector3d origin_e3(5, 0, 0); // origin of the element
        Vector3d aim_point_e3(-17.360680, 0, 94.721360); // aim point of the element
        auto e3 = std::make_shared<Element>();
        e3->set_origin(origin_e3);
        e3->set_aim_point(aim_point_e3); // Aim direction

        if (parabolic) {
        auto surface_e3 = std::make_shared<SurfaceParabolic>();
        surface_e3->set_curvature(curv_x, curv_y);
        e3->set_surface(surface_e3);
        } 
        else {
            // Create a flat surface if not parabolic
            auto surface_e3 = std::make_shared<SurfaceFlat>();
            e3->set_surface(surface_e3);
        }

        auto aperture_e3 = std::make_shared<ApertureRectangle>(dim_x, dim_y);
        e3->set_aperture(aperture_e3);

        system.AddElement(e3);

        //////////////////////////////////////////////
        // STEP 2.1 Create receiver, flat rectangle //
        //////////////////////////////////////////////
        Vector3d receiver_origin(0, 0, 10.0); // origin of the receiver
        Vector3d receiver_aim_point(0, 5, 0); // aim point of the receiver
        double receiver_dim_x = 2.0;
        double receiver_dim_y = 2.0;
        auto e4 = std::make_shared<Element>();
        e4->set_origin(receiver_origin);
        e4->set_aim_point(receiver_aim_point); // Aim direction


        ///////////////////////////////////////////
        // STEP 2.2 create rectangle aperture    //
        ///////////////////////////////////////////
        auto receiver_aperture = std::make_shared<ApertureRectangle>(receiver_dim_x, receiver_dim_y);
        e4->set_aperture(receiver_aperture);

        ///////////////////////////////////
        // STEP 2.3 create flat surface //
        //////////////////////////////////
        auto receiver_surface = std::make_shared<SurfaceFlat>();
        e4->set_surface(receiver_surface);

        ////////////////////////////////////////////
        // STEP 2.4 Add the element to the system //
        ///////////////////////////////////////////
        system.AddElement(e4); // Add the receiver to the system

        // set up sun vector and angle 
        Vector3d sun_vector(0.0, 0.0, 100.0); // sun vector
        double sun_angle = 0.0; // sun angle

        system.setSunVector(sun_vector);
        system.setSunAngle(sun_angle);
    }

    // set up sun angle 
    double sun_angle = 0.0; // sun angle
    system.setSunAngle(sun_angle);

    ///////////////////////////////////
    // STEP 3  Initialize the system //
    ///////////////////////////////////
    system.initialize();

    ////////////////////////////
    // STEP 4  Run Ray Trace //
    ///////////////////////////
    // TODO: set up different sun position trace // 
    system.run();

    //////////////////////////
    // STEP 5  Post process //
    //////////////////////////
    system.writeOutput("output_small_system_parabolic_heliostats_flat_receiver_stinput.csv");

    /////////////////////////////////////////
    // STEP 6  Be a good citizen, clean up //
    /////////////////////////////////////////
    system.cleanup();



    return 0;
}