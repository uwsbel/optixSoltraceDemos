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

        // Initialize the simulation:
        // This will set up the OptiX context, load the modules, create program groups,
        // build the pipeline, and allocate & update device data.
        system.initialize();

        // Run the simulation (this calls optixLaunch internally).
        system.run();

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
