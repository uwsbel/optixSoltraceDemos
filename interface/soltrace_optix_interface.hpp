#pragma once
//#include <soltrace_system.h>
class soltrace_optix_interface {
public:
    explicit soltrace_optix_interface(int num_rays = 10000);
    ~soltrace_optix_interface();

    // Basic controls
    void set_sun_vector(double x, double y, double z);
    void set_sun_angle(double deg);
	void set_sun_points(int n);

    // I/O & lifecycle
    void read_st_input(const char* stinputPath);
    void initialize();         // builds OptiX pipeline
    void run();                // does the ray trace
    //void update();
    void write_output(const char* outPath);

private:
    //SolTraceSystem sys;
    struct Impl;
	std::unique_ptr<Impl> impl; // Pimpl idiom to hide implementation details
};
