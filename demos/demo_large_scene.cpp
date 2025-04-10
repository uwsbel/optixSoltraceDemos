// main.cpp
#include "lib/soltrace_system.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <optix_function_table_definition.h>

int main(int argc, char* argv[]) {
    bool parabolic = true; // Set to true for parabolic mirrors, false for flat mirrors
	bool use_sun_shape = true; 
    int num_rays = 2181883;

    // setup command line input, parabolic? sun shape on? num of rays

    if (argc != 4) {
		std::cout << "Usage: " << argv[0] << " <parabolic?> <use_sun_shape?> <num_rays>" << std::endl;
		return 1;
    }
	// Parse command line arguments
	// parabolic flag (1 for true, 0 for false)
	parabolic = bool(std::stoi(argv[1]));
	use_sun_shape = bool(std::stoi(argv[2]));
	num_rays = std::stoi(argv[3]);

    // Create the simulation system.
    SolTraceSystem system(num_rays);

	const char* stinput_file; // Default stinput file name
    if (parabolic) {
		stinput_file = "../data/stinput/large-system-parabolic-heliostats-cylindrical.stinput"; // Default stinput file name
    }
    else {
		stinput_file = "../data/stinput/large-system-flat-heliostats-cylindrical.stinput"; // Default stinput file name
    }

    std::cout << "Reading STINPUT file." << std::endl;
    system.read_st_input(stinput_file);

    // set up sun angle 
    double sun_angle = 0;
    if (use_sun_shape) {
		sun_angle = 0.00465; // 0.00465; // sun angle
    }
    system.set_sun_angle(sun_angle);

    system.initialize();

    system.run();

	std::cout << "timing_setup_scene, " << system.get_time_setup() << " sec" << std::endl;
	std::cout << "timing_ray_trace,   " << system.get_time_trace() << " sec" << std::endl;
    //system.write_output("output_large_system_flat_heliostats_cylindrical_receiver_stinput-sun_shape_on.csv");

    system.clean_up();

    return 0;
}