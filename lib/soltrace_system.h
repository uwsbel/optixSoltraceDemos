#ifndef SOLTR_SYSTEM_H
#define SOLTR_SYSTEM_H

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/vec_math.h>
#include <soltrace_state.h>
#include <sampleConfig.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
// Sutil headers.
#include <sutil/Exception.h>
#include <sutil/Record.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <cuda/Soltrace.h>
#include <lib/data_manager.h>
#include <lib/pipeline_manager.h>
#include <lib/element.h>
#include <lib/timer.h>
#include <optix_types.h>


using namespace soltrace;

/**
 * @class geometryManager
 * @brief Given the geoemtry of the elements, populate the list of aabb, 
 * compute the sun plane, and build the GAS (Geometry Acceleration Structure) for ray tracing.
 */
class GeometryManager {
public:
	GeometryManager(SoltraceState& state) : m_state(state) {}
    ~GeometryManager() {} 

	/// go through the list of elements and collect the geometry info on the host: 
	/// - AABBs
	/// - GeometryDataST on the host
	/// - SBT index
    void collect_geometry_info(const std::vector<std::shared_ptr<Element>>& element_list,
                               LaunchParams& params);

	/// build the GAS (Geometry Acceleration Structure) using the AABB list, populate optix state
    void create_geometries(LaunchParams& params);


	/// update the GAS (Geometry Acceleration Structure) using the AABB list, populate optix state
	void update_geometry_info(const std::vector<std::shared_ptr<Element>>& element_list,
        LaunchParams& params);

    /// return the list of geometry data vector
	std::vector<GeometryDataST>& get_geometry_data_array() { return m_geometry_data_array_H; }

    // compute sun plane 
    void compute_sun_plane_H(LaunchParams& params);


private: 
	SoltraceState& m_state;
	float m_sun_plane_distance = -1.0f; // distance of the sun plane from the origin
    int m_obj_counts;

    // data related to the geometry and the scene on the host side
	std::vector<OptixAabb>      m_aabb_list_H;           // aabb list
    std::vector<GeometryDataST> m_geometry_data_array_H; // geometry data
    std::vector<uint32_t>       m_sbt_index_H;           // sbt offset index

    // members related to building GAS
	OptixBuildInput        m_aabb_input = {};                   // needed after the first build
	OptixAccelBuildOptions m_accel_build_options = {};  // needed after the first build
	CUdeviceptr            m_aabb_list_D{};          // device pointer to the aabb list
    
    
    CUdeviceptr m_output_buffer{};   // output buffer
	CUdeviceptr m_temp_buffer{};     // temporary buffer for building GAS
    size_t m_output_buffer_size = 0;   // size of that scratch
	size_t m_temp_buffer_size = 0;     // size of the output buffer
};

/**
 * @class SolTraceSystem
 * @brief Main simulation object to: 
 * - build the scene from the elements (heliostats and receiver)
 * - set sun vector and sun points
 * - manage the ray tracing pipeline and data transfer between the host and device
 * - run ray tracing simulation 
 * - and output results
 */
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
	void set_sun_vector(Vector3d vect) { m_sun_vector = vect; } // Set the sun vector

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
