#ifndef SOLTR_SYSTEM_H
#define SOLTR_SYSTEM_H

#include <string>
#include <vector>

#include <geometry_manager.h>
class pipelineManager;

using namespace soltrace;

class SolTraceSystem {
public:
    SolTraceSystem(int numSunPoints);
    ~SolTraceSystem();

    /// Call to this function mark the completion of the simulation setup
    void initialize();

    /// Execute the ray tracing simulation
    void run();

    /// Update launch params
    void update();

    // Read a stinput file for the simulation setup.
    bool read_st_input(const char* filename);

    // Write the output to a file.
    void write_output(const std::string& filename); 

    /// Explicit cleanup
    void clean_up();

	void set_verbose(bool verbose) { m_verbose = verbose; } // Set verbosity for debugging
	/// <summary>
	/// set the number of rays launched
	/// </summary>
	/// <param name="numSunPoints"></param>
	void set_sun_points(int num) { m_num_sunpoints = num; } 

	/// <summary>
	/// set normalized sun vector
	/// </summary>
	/// <param name="sunVector"></param>
    void set_sun_vector(Vector3d vect) {
        m_sun_vector = vect;
        Vector3d sun_v = m_sun_vector.normalized(); // Normalize the sun vector
        data_manager->launch_params_H.sun_vector = make_float3(sun_v[0], sun_v[1], sun_v[2]);
    } // Set the sun vector

	void set_sun_angle(double angle) { m_sun_angle = angle; } // Set the sun angle


	/// <summary>
	/// compute number of heliostat elements added to the system 
	/// </summary>
	/// <returns></returns>
	size_t get_num_heliostats() const
	{
		return m_element_list.size() - 1; // Return the number of heliostats (elements) added
	}

    /// <summary>
    /// compute number of receiver elements added to the system
    /// </summary>
    /// <returns></returns>
    int get_num_receivers() const {
		return 1; // Assuming one receiver for now, can be modified later
    }

	/// <summary>
	/// add element
	/// /// </summary>
    void add_element(std::shared_ptr<Element> element);

    double get_time_trace();
	double get_time_setup();

    void print_launch_params();


private:

    std::shared_ptr<GeometryManager> geometry_manager;
    std::shared_ptr<pipelineManager> pipeline_manager;
    std::shared_ptr<dataManager>     data_manager;

    int m_num_sunpoints;
    bool m_verbose; 

    Vector3d m_sun_vector;
    double m_sun_angle;
    SoltraceState m_state;

    std::vector<std::shared_ptr<Element>> m_element_list;
    void create_shader_binding_table();

    // Helper functions to read a stinput file
    bool read_system(FILE* fp);
    bool read_stage(FILE* fp);
    bool read_element(FILE* fp);
    bool read_optic(FILE* fp); 
    bool read_optic_surface(FILE* fp);
    bool read_sun(FILE* fp);
    void read_line(char* buf, int len, FILE* fp);
    std::vector<std::string> split(const std::string& str, const std::string& delim, bool ret_empty, bool ret_delim);

	Timer m_timer_setup;
    Timer m_timer_trace;

    // memory usage
    size_t m_mem_free_before;
    size_t m_mem_free_after;


};

#endif // SOLTR_SYSTEM_H
