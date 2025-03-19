#ifndef SOLTR_SYSTEM_H
#define SOLTR_SYSTEM_H

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/vec_math.h>
#include <SoltraceState.h>
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
#include <lib/dataManager.h>
#include <lib/pipelineManager.h>
#include <lib/Element.h>
#include <optix_types.h>

//TODO: shall I use float3 at the interface level? 


using namespace soltrace;

/**
 * @class geometryManager
 * @brief Given the geoemtry of the elements, populate the list of aabb, 
 * compute the sun plane, and build the GAS (Geometry Acceleration Structure) for ray tracing.
 */
class geometryManager {
public:
	geometryManager(SoltraceState& state) : m_state(state) {}
    ~geometryManager() {} 

	/// populate the AABB list from the elements
    void populate_aabb_list(const std::vector<std::shared_ptr<Element>>& element_list);

    /// compute the sun plane with sun vector and a list of AABBs 
    // TODO: need to think about who owns this, pipeline? 
    // TODO: create sun class
    void compute_sun_plane();

	/// build the GAS (Geometry Acceleration Structure) using the AABB list, populate optix state
    void create_geometries();


private: 
	SoltraceState& m_state;

    /// list of AABB vertices 
    // TODO: using OptixAABB for now, but should be changed to host side data structure later
	std::vector<OptixAabb> m_aabb_list;
};

/**
 * @class SolTrSystem
 * @brief Main simulation object to: 
 * - build the scene from the elements (heliostats and receiver)
 * - set sun vector and sun points
 * - manage the ray tracing pipeline and data transfer between the host and device
 * - run ray tracing simulation 
 * - and output results
 */
class SolTrSystem {
public:
    SolTrSystem(int numSunPoints);
    ~SolTrSystem();

    /// Call to this function mark the completion of the simulation setup
    void initialize();

    /// Execute the ray tracing simulation
    void run();

    // Write the output to a file.
    void writeOutput(const std::string& filename); 

    /// Explicit cleanup
    void cleanup();

    /// set methods 
	void setVerbose(bool verbose);
	/// <summary>
	/// set the number of rays launched
	/// </summary>
	/// <param name="numSunPoints"></param>
	void setSunPoints(int numSunPoints);

	/// <summary>
	/// set the sun vector
	/// </summary>
	/// <param name="sunVector"></param>
	void setSunVector(float3 sunVector);

	/// <summary>
	/// compute number of heliostat elements added to the system 
	/// </summary>
	/// <returns></returns>
	int get_num_heliostats() const
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
    void AddElement(std::shared_ptr<Element> element);


private:

    std::shared_ptr<geometryManager> geometry_manager;
    std::shared_ptr<pipelineManager> pipeline_manager;
    std::shared_ptr<dataManager>     data_manager;

    int m_num_sunpoints;
    bool m_verbose; 

    SoltraceState m_state;

    std::vector<std::shared_ptr<Element>> m_element_list;

    void createSBT();




};

#endif // SOLTR_SYSTEM_H
