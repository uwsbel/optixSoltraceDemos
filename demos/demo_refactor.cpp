// main.cpp
#include "SolTrSystem.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <optix_function_table_definition.h>


int main(int argc, char* argv[]) {
    try {
        // Default number of sun points; can be overridden via command-line.
        int numSunPoints = 10000;
        if (argc >= 2)
            numSunPoints = std::stoi(argv[1]);

        std::cout << "Starting ST_System simulation with " << numSunPoints << " sun points..." << std::endl;

        // Create the simulation system instance.
        SolTrSystem system(numSunPoints);

        double curv_x = 0.0170679f;
        double curv_y = 0.0370679f;
        double dim_x = 1.0;
        double dim_y = 1.95;

		Vector3d origin(0, 5, 0); // origin of the element
		Vector3d aim_point(0, 4.551982, 1.897836); // aim point of the element

        //GeometryData::Rectangle_Parabolic heliostat1(
        //    make_float3(-1.0f, 0.0f, 0.0f),
        //    make_float3(0.0f, 1.897836f, 0.448018f),
        //    make_float3(0.5f, 4.051082f, -0.224009f),
        //    0.0170679f, 0.0370679f);  // curvature parameters
        //heliostats.push_back(heliostat1);

        auto e1 = std::make_shared<Element>();
        e1->set_origin(origin);
        e1->set_aim_point(aim_point); // Aim direction

        auto surface = std::make_shared<SurfaceParabolic>();
        surface->set_curvature(curv_x, curv_y);
        e1->set_surface(surface);

        auto aperture = std::make_shared<ApertureRectangle>(dim_x, dim_y);
        e1->set_aperture(aperture);
		aperture->compute_device_aperture(origin, aim_point); // Compute the device aperture based on the origin and aim point

		aperture->get_anchor(); // Get the anchor point of the aperture (optional, for debugging)
		aperture->get_v1(); // Get the first vector of the aperture (optional, for debugging)
		aperture->get_v2(); // Get the second vector of the aperture (optional, for debugging)

		std::cout << "Element origin: " << e1->get_origin() << std::endl;
		std::cout << "Element aim point: " << e1->get_aim_point() << std::endl;
		float3 anchor = aperture->get_anchor(); // Get the anchor point of the aperture
		std::cout << "Aperture anchor: " << anchor.x << ", " << anchor.y << ", " << anchor.z << std::endl;
		std::cout << "Aperture v1: " << aperture->get_v1().x << ", " << aperture->get_v1().y << ", " << aperture->get_v1().z << std::endl;
		std::cout << "Aperture v2: " << aperture->get_v2().x << ", " << aperture->get_v2().y << ", " << aperture->get_v2().z << std::endl;

		system.AddElement(e1); // Add the element to the system

        //// Create a heliostat.
        //GeometryData::Rectangle_Parabolic heliostat1(
        //    make_float3(-1.0f, 0.0f, 0.0f),
        //    make_float3(0.0f, 1.897836f, 0.448018f),
        //    make_float3(0.5f, 4.051082f, -0.224009f),
        //    0.0170679f, 0.0370679f);  // curvature parameters
        //heliostats.push_back(heliostat1);





        // Initialize the simulation:
        // This will set up the OptiX context, load the modules, create program groups,
        // build the pipeline, and allocate & update device data.
        system.initialize();

        // Run the simulation (this calls optixLaunch internally).
        system.run();

        // write the result to a file 
		system.writeOutput("output_parabolic.csv");

        // Clean up all allocated resources.
        system.cleanup();

        std::cout << "Simulation completed successfully." << std::endl;
    }


    


    catch (const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
