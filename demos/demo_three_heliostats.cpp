// main.cpp
#include "SolTrSystem.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <optix_function_table_definition.h>
#include "Aperture.h"


int main(int argc, char* argv[]) {
    bool stinput = false; // Set to true if using stinput file, false otherwise
    bool parabolic = true; // Set to true for parabolic mirrors, false for flat mirrors
    // number of rays launched for the simulation
    int num_rays = 1000000;
    // Create the simulation system.
    SolTraceSystem system(num_rays);

    if (stinput) {
        // const char* stinput_file = "../data/stinput/toy_problem_parabolic.stinput"; // Default stinput file name

        // if (argc > 1) {
        //     stinput_file = argv[1]; // Get the stinput file name from command line argument
        // }

        // if (stinput) {
        //     // Read the system from the file
        //     std::cout << "Reading STINPUT file." << std::endl;
        //     system.readStinput(stinput_file);
        // } else {
        //     std::cout << "Error: System setup failed." << std::endl;
        // }
    } else {

        // Element 1
        Vector3d origin_e1(-5, 0, 0); // origin of the element
        Vector3d aim_point_e1(17.360680, 0, 94.721360); // aim point of the element
        //double z_rot_e1 = -88.75721138871927;
        double z_rot_e1 = -90.0;

        auto e1 = std::make_shared<Element>();
        e1->set_origin(origin_e1);
        e1->set_aim_point(aim_point_e1); // Aim direction
        e1->set_zrot(z_rot_e1);

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

        double dim_x = 1.0;
        double dim_y = 1.95;
        auto aperture_heliostat = std::make_shared<ApertureRectangle>(dim_x, dim_y);
        e1->set_aperture(aperture_heliostat);

        system.add_element(e1);
        
        // Element 2
        Vector3d origin_e2(0, 5, 0); // origin of the element
        Vector3d aim_point_e2(0, -17.360680, 94.721360); // aim point of the element
        auto e2 = std::make_shared<Element>();
        e2->set_origin(origin_e2);
        e2->set_aim_point(aim_point_e2); // Aim direction
		e2->set_zrot(0.0); // No rotation for the element

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
        e2->set_aperture(aperture_heliostat);

        system.add_element(e2);

        //// Element 3
        Vector3d origin_e3(5, 0, 0); // origin of the element
        Vector3d aim_point_e3(-17.360680, 0, 94.721360); // aim point of the element
        auto e3 = std::make_shared<Element>();
        e3->set_origin(origin_e3);
        e3->set_aim_point(aim_point_e3); // Aim direction
        e3->set_zrot(-90.0);

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
        e3->set_aperture(aperture_heliostat);

        system.add_element(e3);

        Vector3d receiver_origin(0, 0, 10.0); // origin of the receiver
        Vector3d receiver_aim_point(0, 5, 10.0); // aim point of the receiver
        double receiver_dim_x = 2.0;
        double receiver_dim_y = 2.0;
        auto e4 = std::make_shared<Element>();
        e4->set_origin(receiver_origin);
        e4->set_aim_point(receiver_aim_point); // Aim direction
		e4->set_zrot(0.0); // No rotation for the receiver

        ////////////////
        // receiver   //
        ////////////////
        double cyl_height = 2;
        double cyl_radius = 0.5;
        auto receiver_aperture = std::make_shared<ApertureRectangle>(2*cyl_radius, cyl_height);
        e4->set_aperture(receiver_aperture);

        ///////////////////////////////////
        // STEP 2.3 create flat surface //
        //////////////////////////////////
        auto receiver_surface = std::make_shared<SurfaceFlat>();
		auto receiver_surface_cylinder = std::make_shared<SurfaceCylinder>();
        e4->set_surface(receiver_surface_cylinder);

        ////////////////////////////////////////////
        // STEP 2.4 Add the element to the system //
        ///////////////////////////////////////////
        system.add_element(e4); // Add the receiver to the system

        // set up sun vector and angle 
        Vector3d sun_vector(0.0, 0.0, 100.0); // sun vector
        double sun_angle = 0.004; // sun angle

        system.set_sun_vector(sun_vector);
        system.set_sun_angle(sun_angle);
    }

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
    if (parabolic) {
        system.write_output("output_parabolic_heliostats.csv");
    }
    else {
		system.write_output("output_flat_heliostats.csv");
    }


    /////////////////////////////////////////
    // STEP 6  Be a good citizen, clean up //
    /////////////////////////////////////////
    system.clean_up();



    return 0;
}