// main.cpp
#include "SolTrSystem.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <filesystem>
#include <optix_function_table_definition.h>
#include "lib/timer.h"

int main(int argc, char* argv[]) {
    bool stinput = true; // Set to true if using stinput file, false otherwise
    bool parabolic = true; // Set to true for parabolic mirrors, false for flat mirrors
    bool use_cylindical = true;
    // number of rays launched for the simulation
    //int num_rays = 2181883;
	int num_rays = 1000000; // Number of rays to be launched
    // Create the simulation system.
    SolTraceSystem system(num_rays);

    if (stinput) {
        //const char* stinput_file = "../bin/data/stinput/toy_problem_parabolic.stinput"; // Default stinput file name
		// print out current directory
		//std::cout << "Current directory: " << std::filesystem::current_path()<< std::endl;
        //const char* stinput_file = "../bin/data/stinput/large-system-flat-heliostats-cylindrical.stinput";
        const char* stinput_file = "../data/stinput/large-system-parabolic-heliostats-cylindrical.stinput"; // Default stinput file name

        if (argc > 1) {
            stinput_file = argv[1]; // Get the stinput file name from command line argument
        }

        if (stinput) {
            // Read the system from the file
            std::cout << "Reading STINPUT file." << std::endl;
            system.read_st_input(stinput_file);
        } else {
            std::cout << "Error: System setup failed." << std::endl;
        }
    } 
    // set up sun angle 
    double sun_angle = 0; // 0.00465; // sun angle
    system.set_sun_angle(sun_angle);

    ///////////////////////////////////
    // STEP 3  Initialize the system //
    ///////////////////////////////////
    system.initialize();

    ////////////////////////////
    // STEP 4  Run Ray Trace //
    ///////////////////////////
    // TODO: set up different sun position trace // 

	Timer timer;
    timer.start();
    system.run();
    timer.stop();
	std::cout << "Time taken for ray tracing: " << timer.get_time_sec() << " seconds." << std::endl;
	std::cout << "time taken for ray tracing (miliseconds): " << timer.get_time_milli_sec() << " milliseconds." << std::endl;
    //////////////////////////
    // STEP 5  Post process //
    //////////////////////////
    //system.writeOutput("output_large_system_flat_heliostats_cylindrical_receiver_stinput-sun_shape_on.csv");

    /////////////////////////////////////////
    // STEP 6  Be a good citizen, clean up //
    /////////////////////////////////////////
    system.clean_up();



    return 0;
}