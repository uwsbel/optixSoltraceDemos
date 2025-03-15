#ifndef SOLTR_SYSTEM_H
#define SOLTR_SYSTEM_H

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/vec_math.h>
#include <lib/geometry.h>
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


class geometryManager {
public:
	geometryManager(SoltraceState& state) : m_state(state) {}
    ~geometryManager() {} 

	// populate the AABB list from the elements
    // AABB is for computing the sun plane and for buildGAS 
	void populate_aabb_list(const std::vector<std::shared_ptr<Element>>& element_list) {}

    // compute sun plane, use list of AABB and sun vector from m_state 
	// to populate sun plane vectors in m_state 
    // TODO: need to think about who owns this, pipeline? 
    // top priority: create sun class
    void compute_sun_plane() {
		// update m_state usiing m_aabb_list
    }

	// build the GAS (Geometry Acceleration Structure) using the AABB list
    void buildGas() {
		// update m_state gas handles, etc, using m_aabb_list
    }


private: 
	SoltraceState m_state;

    // list of AABB vertices 
    // TODO: using OptixAABB for now, but should be changed to host side data structure later
	std::vector<OptixAabb> m_aabb_list;
};

//Interface to soltrace simulation system
//wrapper around scene setup, pipeline, ray trace sim, post process, etc
//hardward agnostic, can be extended for vulkan 
class SolTrSystem {
public:
    SolTrSystem(int numSunPoints);
    ~SolTrSystem();

    // add element
    // void addElement(Element e);

    // Call to this function mark the completion of the simulation setup
    void initialize();

    // Execute the ray tracing simulation
    void run();

    // Write the output to a file.
    void writeOutput(const std::string& filename); 

    // Explicit cleanup, also invoked in the destructor.
    void cleanup();

    // set methods 
	void setVerbose(bool verbose);

	void setSunPoints(int numSunPoints);

	void setSunVector(float3 sunVector);

    // get methods 
	int get_num_heliostats() const
	{
		return m_element_list.size() - 1; // Return the number of heliostats (elements) added
	}

    int get_num_receivers() const {
		return 1; // Assuming one receiver for now, can be modified later
    }

    void AddElement(std::shared_ptr<Element> element);


private:

    std::shared_ptr<geometryManager> geometry_manager;
    std::shared_ptr<pipelineManager> pipeline_manager;
    std::shared_ptr<dataManager>     data_manager;

    int m_num_sunpoints;
    bool m_verbose; 

    SoltraceState m_state;

    std::vector<std::shared_ptr<Element>> m_element_list;

    void createSBT(std::vector<GeometryData::Rectangle_Parabolic>& helistat_list, std::vector<GeometryData::Parallelogram> receiver_list);




};

#endif // SOLTR_SYSTEM_H
