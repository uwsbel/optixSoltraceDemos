#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <optix.h>
#include <sampleConfig.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
// Sutil headers.
#include <sutil/Exception.h>
#include <sutil/Record.h>
#include <sutil/sutil.h>

#include <cuda/Soltrace.h>
#include <lib/dataManager.h>

#include <SoltraceState.h>
#include <lib/pipelineManager.h>
#include <fstream>


char LOG[2048] = {};   // A mutable log buffer.
size_t LOG_SIZE = sizeof(LOG);


const char* intersectionFuncs[] = {
    "__intersection__parallelogram",
    "__intersection__cylinder_y",
    "__intersection__rectangle_parabolic"
};

const char* closestHitFuncs[] = {
    "__closesthit__mirror",
    "__closesthit__mirror",
    "__closesthit__mirror__parabolic"
};

pipelineManager::pipelineManager(SoltraceState& state) : m_state(state) {}


pipelineManager::~pipelineManager() {
    //OPTIX_CHECK(optixProgramGroupDestroy(m_state.m_raygen_prog_group));
    //OPTIX_CHECK(optixProgramGroupDestroy(m_radiance_mirror_prog_group));
    //OPTIX_CHECK(optixProgramGroupDestroy(m_radiance_receiver_prog_group));
    //OPTIX_CHECK(optixProgramGroupDestroy(m_radiance_miss_prog_group));

    //OPTIX_CHECK(optixModuleDestroy(m_geometry_module));
    //OPTIX_CHECK(optixModuleDestroy(m_shading_module));
    //OPTIX_CHECK(optixModuleDestroy(m_sun_module));

    //OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
}

std::string pipelineManager::loadPtxFromFile(const std::string& kernelName) {
    std::string ptxFile = std::string(SAMPLES_PTX_DIR) + "demo_refactor_generated_" + kernelName + ".cu.ptx";
    std::ifstream file(ptxFile);
    if (!file.good())
        throw std::runtime_error("PTX file not found: " + ptxFile);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void pipelineManager::loadModules() {
    OptixModuleCompileOptions moduleCompileOptions = {};
#if !defined(NDEBUG)
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    // Geometry module.
    {
        std::string ptx = loadPtxFromFile("intersection");
        OPTIX_CHECK(optixModuleCreate(
            m_state.context,
            &moduleCompileOptions,
            &m_state.pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            LOG, &LOG_SIZE,
            &m_state.geometry_module));
    }
    // Shading/materials module.
    {
        std::string ptx = loadPtxFromFile("materials");
        OPTIX_CHECK(optixModuleCreate(
            m_state.context,
            &moduleCompileOptions,
            &m_state.pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            LOG, &LOG_SIZE,
            &m_state.shading_module));
    }
    // Sun module.
    {
        std::string ptx = loadPtxFromFile("sun");
        OPTIX_CHECK(optixModuleCreate(
            m_state.context,
            &moduleCompileOptions,
            &m_state.pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            LOG, &LOG_SIZE,
            &m_state.sun_module));
    }
}

void pipelineManager::createPipeline()
{
    m_state.pipeline_compile_options = {
        false,                                                  // usesMotionBlur: Disable motion blur.
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags: Allow only a single GAS.
        2,    /* RadiancePRD uses 5 payloads */                 // numPayloadValues
        5,    /* Parallelogram intersection uses 5 attrs */     // numAttributeValues 
        OPTIX_EXCEPTION_FLAG_NONE,                              // exceptionFlags
        "params"                                                // pipelineLaunchParamsVariableName
    };

    // Prepare modules and program groups
    loadModules();
    createSunProgram();
    createMirrorPrograms();
    createReceiverProgram();
    createMissProgram();

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {};
    // TODO max trace belong to who? 
    pipeline_link_options.maxTraceDepth = MAX_TRACE_DEPTH; // Maximum recursion depth for ray tracing.

    // Create the OptiX pipeline by linking the program groups.
    OPTIX_CHECK_LOG(optixPipelineCreate(
        m_state.context,                        // OptiX context.
        &m_state.pipeline_compile_options,      // Compile options for the pipeline.
        &pipeline_link_options,               // Link options for the pipeline.
        m_program_groups.data(),                // Array of program groups.
        static_cast<unsigned int>(m_program_groups.size()), // Number of program groups.
        LOG, &LOG_SIZE,                       // Logs for diagnostics.
        &m_state.pipeline                       // Output: Handle for the created pipeline.
    ));

    // Compute and configure the stack sizes for the pipeline.
    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : m_program_groups)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, m_state.pipeline));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;

    // Compute stack sizes based on the maximum trace depth and other settings.
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,                      // Input stack sizes.
        MAX_TRACE_DEPTH,                         // Maximum trace depth.
        0,                                 // maxCCDepth: Maximum depth of continuation callables (none in this case).
        0,                                 // maxDCDepth: Maximum depth of direct callables (none in this case).
        &direct_callable_stack_size_from_traversal, // Output: Stack size for callable traversal.
        &direct_callable_stack_size_from_state,     // Output: Stack size for callable state.
        &continuation_stack_size                    // Output: Stack size for continuation stack.
    ));
    // Set the computed stack sizes for the pipeline.
    OPTIX_CHECK(optixPipelineSetStackSize(
        m_state.pipeline,                               // Pipeline to configure.
        direct_callable_stack_size_from_traversal,    // Stack size for direct callable traversal.
        direct_callable_stack_size_from_state,        // Stack size for direct callable state.
        continuation_stack_size,                      // Stack size for continuation stack.
        1                                            // maxTraversableDepth: Maximum depth of traversable hierarchy.
    ));
}

OptixPipeline pipelineManager::getPipeline() const {
	return m_state.pipeline;
}

void pipelineManager::createHitGroupProgram(OptixProgramGroup& group,
    OptixModule intersectionModule, const char* intersectionFunc,
    OptixModule closestHitModule, const char* closestHitFunc) {
    OptixProgramGroupOptions options = {};
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc.hitgroup.moduleIS = intersectionModule;
    desc.hitgroup.entryFunctionNameIS = intersectionFunc;
    desc.hitgroup.moduleCH = closestHitModule;
    desc.hitgroup.entryFunctionNameCH = closestHitFunc;
    desc.hitgroup.moduleAH = nullptr;
    desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_state.context,
        &desc,
        1,
        &options,
        LOG, &LOG_SIZE,
        &group
    ));
}

// TODO: simplify 
void pipelineManager::createSunProgram()
{
    OptixProgramGroup           group;                 // Handle for the sun program group.
    OptixProgramGroupOptions    options = {};    // Options for the program group (none needed in this case).
    OptixProgramGroupDesc       desc = {};       // Descriptor to define the program group.

    // Specify the kind of program group (Ray Generation).
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    // Link the ray generation program to the sun module and specify the function name.
    desc.raygen.module = m_state.sun_module;
    desc.raygen.entryFunctionName = "__raygen__sun_source";

    // Create the program group
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_state.context,                 // OptiX context.
        &desc,          // Descriptor defining the program group.
        1,                             // Number of program groups to create (1 in this case).
        &options,       // Options for the program group.
        LOG, &LOG_SIZE,                // Logs to capture diagnostic information.
        &group                // Output: Handle for the created program group.
    ));

    m_program_groups.push_back(group);
    m_state.raygen_prog_group = group;
}

// Create program group for handling rays interacting with mirrors.
void pipelineManager::createMirrorPrograms()
{   

    // number of mirror programs
	size_t numMirrorPrograms = sizeof(intersectionFuncs) / sizeof(intersectionFuncs[0]);

	for (size_t i = 0; i < numMirrorPrograms; i++) {
		OptixProgramGroup group; 

		createHitGroupProgram(group,
            			      m_state.geometry_module, 
                              intersectionFuncs[i],
            			      m_state.shading_module, 
                              closestHitFuncs[i]);

		m_program_groups.push_back(group);
        m_state.radiance_mirror_prog_group = group;
	}   


	//OptixProgramGroup group; // Handle for the program group.

 //   createHitGroupProgram(group,
 //       m_state.geometry_module, "__intersection__rectangle_parabolic",
 //       m_state.shading_module,  "__closesthit__mirror__parabolic");

 //   m_program_groups.push_back(group);
}


// TODO: think about if we actually need the receiver program ... 
// since the only difference is the closest hit program, which can be modified 
// in reflectivity and absorption? 
void pipelineManager::createReceiverProgram()
{
    OptixProgramGroup           group;
	createHitGroupProgram(group,
		m_state.geometry_module, "__intersection__parallelogram",
		m_state.shading_module, "__closesthit__receiver");

    m_program_groups.push_back(group);
    m_state.radiance_receiver_prog_group = group;
}

// Create program group for handling rays that miss all geometry.
void pipelineManager::createMissProgram()
{
    OptixProgramGroup           group;       // Handle for the miss program group.
    OptixProgramGroupOptions    options = {};   // Options for the program group (none needed).
    OptixProgramGroupDesc       desc = {};      // Descriptor for the program group.

    // Specify the kind of program group (Miss Program for handling missed rays).
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    // Link the miss shader (background or environment shading) to the shading module.
    desc.miss.module = m_state.shading_module;
    desc.miss.entryFunctionName = "__miss__ms";

    // Create the program grou
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_state.context,
        &desc,
        1,
        &options,
        LOG, &LOG_SIZE,
        &group));

    m_program_groups.push_back(group);
    m_state.radiance_miss_prog_group = group;
}
