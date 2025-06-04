#include <soltrace_system.h>
#include <soltrace_optix_interface.hpp>

soltrace_optix_interface::soltrace_optix_interface(int num_rays)
    : sys(num_rays) {
}

soltrace_optix_interface::~soltrace_optix_interface() {
    // Destructor logic if needed
}

void soltrace_optix_interface::set_sun_vector(double x, double y, double z) {
	Vector3d vect(x, y, z);
    sys.set_sun_vector(vect);
}

void soltrace_optix_interface::set_sun_angle(double deg) {
    sys.set_sun_angle(deg);
}

void soltrace_optix_interface::set_sun_points(int n) {
    sys.set_sun_points(n);
}

void soltrace_optix_interface::read_st_input(const char* stinputPath) {
    sys.read_st_input(stinputPath);
}

void soltrace_optix_interface::initialize() {
    sys.initialize();
}


void soltrace_optix_interface::run() {
    sys.run();
}

void soltrace_optix_interface::write_output(const char* outPath) {
    sys.write_output(outPath);
}
