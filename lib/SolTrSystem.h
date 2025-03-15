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

//TODO: shall I use float3 at the interface level? 


using namespace soltrace;


class geometryManager {
//public:
//    geometryManager();
//    ~geometryManager();
//
//    // Setup geometry for the scene.
//    // This function takes the simulation state and vectors of geometry data (heliostats and receivers)
//    // and builds the corresponding acceleration structures (GAS) for OptiX.
//    void setupGeometry(SoltraceState& state,
//        const std::vector<GeometryData::Rectangle_Parabolic>& heliostats,
//        const std::vector<GeometryData::Parallelogram>& receivers);
//
//    // Optionally, you can add getters to retrieve geometry-related info if needed.
//private:
//    // Private helper functions can be added here, for example:
//    // void buildGAS(SoltraceState &state, ...);
};

// class pipelineManager {
// public:
//     pipelineManager(SoltraceState& state);
//     ~pipelineManager();

//     void createPipeline();

//     OptixPipeline getPipeline() const;

// private:
//     std::string loadPtxFromFile(const std::string& kernelName);
//     void loadModules();
// 	void createSunProgram();
// 	void createMirrorProgram();
// 	void createReceiverProgram();
// 	void createMissProgram();
// 	SoltraceState& m_state;
//     std::vector<OptixProgramGroup> m_program_groups;
// }; 

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
