#include <soltrace_system.h>
#include <soltrace_optix_interface.hpp>


struct soltrace_optix_interface::Impl {
	SolTraceSystem sys;

	Impl(int num_rays) : sys(num_rays) {}
};

soltrace_optix_interface::soltrace_optix_interface(int num_rays)
	: impl(std::make_unique<Impl>(num_rays)) {}


//soltrace_optix_interface::soltrace_optix_interface(int num_rays)
//    : sys(num_rays) {
//}

soltrace_optix_interface::~soltrace_optix_interface() {
    // Destructor logic if needed
}

void soltrace_optix_interface::set_sun_vector(double x, double y, double z) {
	Vector3d vect(x, y, z);
    impl->sys.set_sun_vector(vect);
}

void soltrace_optix_interface::set_sun_angle(double deg) {
    impl->sys.set_sun_angle(deg);
}

void soltrace_optix_interface::set_sun_points(int n) {
    impl->sys.set_sun_points(n);
}

void soltrace_optix_interface::read_st_input(const char* stinputPath) {
    impl->sys.read_st_input(stinputPath);
}

void soltrace_optix_interface::initialize() {
    impl->sys.initialize();
}


void soltrace_optix_interface::run() {
    impl->sys.run();
}

void soltrace_optix_interface::write_output(const char* outPath) {
    impl->sys.write_output(outPath);
}
