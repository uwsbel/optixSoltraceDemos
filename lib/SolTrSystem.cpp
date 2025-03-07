#include "SolTrSystem.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <iomanip>

char LOG[2048] = {};   // A mutable log buffer.
size_t LOG_SIZE = sizeof(LOG);
int max_trace = 5;  // Maximum recursion depth for ray tracing.

typedef sutil::Record<soltrace::HitGroupData> HitGroupRecord;


// -------------------- dataManager Implementation --------------------
dataManager::dataManager() : device_launch_params(nullptr) { 
	host_launch_params.width = 0;
	host_launch_params.height = 0;
	host_launch_params.max_depth = 0;

}

dataManager::~dataManager() {
    if (device_launch_params) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(device_launch_params)));
        device_launch_params = nullptr;
    }
}
//
soltrace::LaunchParams* dataManager::getDeviceLaunchParams() const { return device_launch_params; }


void dataManager::allocateLaunchParams() {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_launch_params), sizeof(LaunchParams)));
}

void dataManager::updateLaunchParams() {
    CUDA_CHECK(cudaMemcpy(device_launch_params, &host_launch_params, sizeof(LaunchParams), cudaMemcpyHostToDevice));
}


// -------------------- pipeline_manager Implementation --------------------
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
    createMirrorProgram();
    createReceiverProgram();
    createMissProgram();

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace; // Maximum recursion depth for ray tracing.

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
        max_trace,                         // Maximum trace depth.
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

void pipelineManager::createSunProgram()
{
    OptixProgramGroup           sun_prog_group;                 // Handle for the sun program group.
    OptixProgramGroupOptions    sun_prog_group_options = {};    // Options for the program group (none needed in this case).
    OptixProgramGroupDesc       sun_prog_group_desc = {};       // Descriptor to define the program group.

    // Specify the kind of program group (Ray Generation).
    sun_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    // Link the ray generation program to the sun module and specify the function name.
    sun_prog_group_desc.raygen.module = m_state.sun_module;
    sun_prog_group_desc.raygen.entryFunctionName = "__raygen__sun_source";

    // Create the program group
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_state.context,                 // OptiX context.
        &sun_prog_group_desc,          // Descriptor defining the program group.
        1,                             // Number of program groups to create (1 in this case).
        &sun_prog_group_options,       // Options for the program group.
        LOG, &LOG_SIZE,                // Logs to capture diagnostic information.
        &sun_prog_group                // Output: Handle for the created program group.
    ));

    m_program_groups.push_back(sun_prog_group);
    m_state.raygen_prog_group = sun_prog_group;
}

// Create program group for handling rays interacting with mirrors.
void pipelineManager::createMirrorProgram()
{
    OptixProgramGroup           radiance_mirror_prog_group;                 // Handle for the mirror program group.
    OptixProgramGroupOptions    radiance_mirror_prog_group_options = {};    // Options for the program group (none needed).
    OptixProgramGroupDesc       radiance_mirror_prog_group_desc = {};       // Descriptor for the program group.

    // Specify the kind of program group (Hit Group for handling intersections and shading).
    radiance_mirror_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    // Link the intersection shader (geometry handling) to the geometry module.
    radiance_mirror_prog_group_desc.hitgroup.moduleIS = m_state.geometry_module;
    radiance_mirror_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__rectangle_parabolic";
    // Link the closest-hit shader (shading logic) to the shading module.
    radiance_mirror_prog_group_desc.hitgroup.moduleCH = m_state.shading_module;
    radiance_mirror_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__mirror__parabolic";
    // No any-hit shader is used in this configuration (set to nullptr).
    radiance_mirror_prog_group_desc.hitgroup.moduleAH = nullptr;
    radiance_mirror_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    // Create the program group
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_state.context,
        &radiance_mirror_prog_group_desc,
        1,
        &radiance_mirror_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_mirror_prog_group));

    m_program_groups.push_back(radiance_mirror_prog_group);
    m_state.radiance_mirror_prog_group = radiance_mirror_prog_group;
}

// Create program group for handling rays interacting with the receiver.
void pipelineManager::createReceiverProgram()
{
    OptixProgramGroup           radiance_receiver_prog_group;               // Handle for the receiver program group.
    OptixProgramGroupOptions    radiance_receiver_prog_group_options = {};  // Options for the program group (none needed).
    OptixProgramGroupDesc       radiance_receiver_prog_group_desc = {};     // Descriptor for the program group.

    // Specify the kind of program group (Hit Group for handling intersections and shading).
    radiance_receiver_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    // Link the intersection shader (geometry handling) to the geometry module.
    radiance_receiver_prog_group_desc.hitgroup.moduleIS = m_state.geometry_module;
    radiance_receiver_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__parallelogram";
    // Link the closest-hit shader (shading logic) to the shading module.
    radiance_receiver_prog_group_desc.hitgroup.moduleCH = m_state.shading_module;
    radiance_receiver_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__receiver";
    // No any-hit shader is used in this configuration (set to nullptr).
    radiance_receiver_prog_group_desc.hitgroup.moduleAH = nullptr;
    radiance_receiver_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    // Create the program group
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_state.context,
        &radiance_receiver_prog_group_desc,
        1,
        &radiance_receiver_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_receiver_prog_group));

    m_program_groups.push_back(radiance_receiver_prog_group);
    m_state.radiance_receiver_prog_group = radiance_receiver_prog_group;
}

// Create program group for handling rays that miss all geometry.
void pipelineManager::createMissProgram()
{
    OptixProgramGroup           radiance_miss_prog_group;       // Handle for the miss program group.
    OptixProgramGroupOptions    miss_prog_group_options = {};   // Options for the program group (none needed).
    OptixProgramGroupDesc       miss_prog_group_desc = {};      // Descriptor for the program group.

    // Specify the kind of program group (Miss Program for handling missed rays).
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    // Link the miss shader (background or environment shading) to the shading module.
    miss_prog_group_desc.miss.module = m_state.shading_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";

    // Create the program grou
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_state.context,
        &miss_prog_group_desc,
        1,
        &miss_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_miss_prog_group));

    m_program_groups.push_back(radiance_miss_prog_group);
    m_state.radiance_miss_prog_group = radiance_miss_prog_group;
}


// -------------------- SolTrSystem Implementation --------------------
SolTrSystem::SolTrSystem(int numSunPoints)
    : m_num_sunpoints(numSunPoints)
{

    // need to attach state to it
    geometry_manager = std::make_shared<geometryManager>();
    data_manager = std::make_shared<dataManager>();


    CUDA_CHECK(cudaFree(0));
    CUcontext cuCtx = 0;
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = [](unsigned int level, const char* tag, const char* message, void*) {
        std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
        };
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &m_state.context));

    pipeline_manager = std::make_shared<pipelineManager>(m_state);
}

SolTrSystem::~SolTrSystem() {
    cleanup();
}

void SolTrSystem::initialize() {

    std::vector<GeometryData::Rectangle_Parabolic> heliostats;
    std::vector<GeometryData::Parallelogram> receivers;

    // Create a heliostat.
    GeometryData::Rectangle_Parabolic heliostat(
        make_float3(-1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 1.897836f, 0.448018f),
        make_float3(0.5f, 4.051082f, -0.224009f),
        0.0170679f, 0.0370679f);  // curvature parameters
    heliostats.push_back(heliostat);

    // Create a receiver.
    GeometryData::Parallelogram receiver(
        make_float3(2.0f, 0.0f, 0.0f),
        make_float3(0.0f, 1.788854f, 0.894428f),
        make_float3(-1.0f, -0.894427f, 9.552786f));
    receivers.push_back(receiver);

    // do this for state as well 
    m_state.params.sun_vector = make_float3(0.0f, 0.0f, 100.0f);
    m_state.params.max_sun_angle = 0.00465;     // 4.65 mrad


    // Call your function to build the geometry acceleration structure.
    // need to separate this from the sun part 
    createGeometry_parabolic(m_state, heliostats, receivers);
    // Pipeline setup.
    pipeline_manager->createPipeline();

    createSBT(heliostats, receivers);

    // set up input related to sun
	data_manager->host_launch_params.sun_vector = make_float3(0.0f, 0.0f, 100.0f);
    data_manager->host_launch_params.max_sun_angle = 0.00465;     // 4.65 mrad

    // Initialize launch params
    data_manager->host_launch_params.width = m_num_sunpoints;
    data_manager->host_launch_params.height = 1;
    data_manager->host_launch_params.max_depth = MAX_TRACE_DEPTH;

    // Allocate memory for the hit point buffer, size is number of rays launched * depth
    // TODO: why is this float4? 
    const size_t hit_point_buffer_size = data_manager->host_launch_params.width * data_manager->host_launch_params.height * sizeof(float4) * data_manager->host_launch_params.max_depth;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&data_manager->host_launch_params.hit_point_buffer),
        hit_point_buffer_size
    ));
    CUDA_CHECK(cudaMemset(data_manager->host_launch_params.hit_point_buffer, 0, hit_point_buffer_size));

    // Create a CUDA stream for asynchronous operations.
    CUDA_CHECK(cudaStreamCreate(&m_state.stream));


    // Link the GAS handle.
    data_manager->host_launch_params.handle = m_state.gas_handle;
	// now copy sun_v0, sun_v1, sun_v2, sun_v3 to launch params
	data_manager->host_launch_params.sun_v0 = m_state.params.sun_v0;
	data_manager->host_launch_params.sun_v1 = m_state.params.sun_v1;
	data_manager->host_launch_params.sun_v2 = m_state.params.sun_v2;
	data_manager->host_launch_params.sun_v3 = m_state.params.sun_v3;

    // print all the field in the launch params
    std::cout << "print launch params: " << std::endl;
    std::cout << "width: " << data_manager->host_launch_params.width << std::endl;
    std::cout << "height: " << data_manager->host_launch_params.height << std::endl;
    std::cout << "max_depth: " << data_manager->host_launch_params.max_depth << std::endl;
    std::cout << "hit_point_buffer: " << data_manager->host_launch_params.hit_point_buffer << std::endl;
    std::cout << "sun_vector: " << data_manager->host_launch_params.sun_vector.x << " " << data_manager->host_launch_params.sun_vector.y << " " << data_manager->host_launch_params.sun_vector.z << std::endl;
    std::cout << "max_sun_angle: " << data_manager->host_launch_params.max_sun_angle << std::endl;
    std::cout << "sun_v0: " << data_manager->host_launch_params.sun_v0.x << " " << data_manager->host_launch_params.sun_v0.y << " " << data_manager->host_launch_params.sun_v0.z << std::endl;
    std::cout << "sun_v1: " << data_manager->host_launch_params.sun_v1.x << " " << data_manager->host_launch_params.sun_v1.y << " " << data_manager->host_launch_params.sun_v1.z << std::endl;
    std::cout << "sun_v2: " << data_manager->host_launch_params.sun_v2.x << " " << data_manager->host_launch_params.sun_v2.y << " " << data_manager->host_launch_params.sun_v2.z << std::endl;
    std::cout << "sun_v3: " << data_manager->host_launch_params.sun_v3.x << " " << data_manager->host_launch_params.sun_v3.y << " " << data_manager->host_launch_params.sun_v3.z << std::endl;

    // copy launch params to device
    data_manager->allocateLaunchParams();
    data_manager->updateLaunchParams();

    // TODO: this is redundant but put here for now. need to separate optix and non-optix related members
    m_state.params = data_manager->host_launch_params;

}

void SolTrSystem::run() {

    auto params = data_manager->getDeviceLaunchParams();
    if (!params) {
        std::cerr << "LaunchParams pointer is null!" << std::endl;
        return;
    }

    int width = data_manager->host_launch_params.width;
    int height = data_manager->host_launch_params.height;

    // Launch the simulation.
    OPTIX_CHECK(optixLaunch(
        m_state.pipeline,
        m_state.stream,  // Assume this stream is properly created.
        reinterpret_cast<CUdeviceptr>(data_manager->getDeviceLaunchParams()),
        sizeof(soltrace::LaunchParams),
		&m_state.sbt,    // Shader Binding Table.
        width,  // Launch dimensions
        height,
        1));
    CUDA_SYNC_CHECK();

    int output_size = width * height * data_manager->host_launch_params.max_depth;
    std::vector<float4> hp_output_buffer(output_size);
    CUDA_CHECK(cudaMemcpy(hp_output_buffer.data(), data_manager->host_launch_params.hit_point_buffer, output_size * sizeof(float4), cudaMemcpyDeviceToHost));

	std::string filename = "output.csv";

    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open the file " << filename << " for writing." << std::endl;
        return;
    }

    // Write header
    outFile << "number,stage,loc_x,loc_y,loc_z\n";

    int currentRay = 1;
    int stage = 0;

    for (const auto& element : hp_output_buffer) {

        // Inline check: if y, z, and w are all zero, treat as marker for new ray.
        if ((element.y == 0) && (element.z == 0) && (element.w == 0)) {
            if (stage > 0) {
                currentRay++;
                stage = 0;
            }
            continue;  // Skip printing this marker element.
        }

        // If we haven't reached max_trace stages for the current ray, print the element.
        if (stage < data_manager->host_launch_params.max_depth) {
            outFile << currentRay << ","
                << element.x << "," << element.y << ","
                << element.z << "," << element.w << "\n";
            stage++;
        }
        else {
            // If max_trace stages reached, move to next ray and reset stage counter.
            currentRay++;
            stage = 0;
            outFile << currentRay << ","
                << element.x << "," << element.y << ","
                << element.z << "," << element.w << "\n";
            stage++;
        }


    }

    outFile.close();
    std::cout << "Data successfully written to " << filename << std::endl;




}

void SolTrSystem::cleanup() {
    // Cleanup pipeline_manager resources via its destructor.
    OPTIX_CHECK(optixDeviceContextDestroy(m_state.context));
    // data_manager cleanup is handled in its destructor.
    // geometry_manager cleanup can be added when implemented.
}

// Ccreate and configure the Shader Binding Table (SBT).
// The SBT is a crucial data structure in OptiX that links geometry and ray types
// with their corresponding programs (ray generation, miss, and hit group).
void SolTrSystem::createSBT(std::vector<GeometryData::Rectangle_Parabolic>& helistat_list, std::vector<GeometryData::Parallelogram> receiver_list)
{
    int num_heliostats = helistat_list.size();
    int num_receivers = receiver_list.size();
    int obj_count = helistat_list.size() + receiver_list.size();

    // Ray generation program record
    {
        CUdeviceptr d_raygen_record;                   // Device pointer to hold the raygen SBT record.
        size_t      sizeof_raygen_record = sizeof(sutil::EmptyRecord);

        // Allocate memory for the raygen SBT record on the device.
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_raygen_record),
            sizeof_raygen_record));

        sutil::EmptyRecord rg_sbt;  // Host representation of the raygen SBT record.

        // Pack the program header into the raygen SBT record.
        optixSbtRecordPackHeader(m_state.raygen_prog_group, &rg_sbt);

        // Copy the raygen SBT record from host to device.
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_raygen_record),
            &rg_sbt,
            sizeof_raygen_record,
            cudaMemcpyHostToDevice
        ));

        // Assign the device pointer to the raygenRecord field in the SBT.
        m_state.sbt.raygenRecord = d_raygen_record;
    }

    // Miss program record
    {
        CUdeviceptr d_miss_record;
        size_t sizeof_miss_record = sizeof(sutil::EmptyRecord);

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_miss_record),
            sizeof_miss_record * soltrace::RAY_TYPE_COUNT));

        sutil::EmptyRecord ms_sbt[soltrace::RAY_TYPE_COUNT];
        // Pack the program header into the first miss SBT record.
        optixSbtRecordPackHeader(m_state.radiance_miss_prog_group, &ms_sbt[0]);

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_miss_record),
            ms_sbt,
            sizeof_miss_record * soltrace::RAY_TYPE_COUNT,
            cudaMemcpyHostToDevice
        ));

        // Configure the SBT miss program fields.
        m_state.sbt.missRecordBase = d_miss_record;                   // Base address of the miss records.
        m_state.sbt.missRecordCount = soltrace::RAY_TYPE_COUNT;        // Number of miss records.
        m_state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof_miss_record);    // Stride between miss records.
    }

    // Hitgroup program record
    {
        // Total number of hitgroup records : one per ray type per object.
        const size_t count_records = soltrace::RAY_TYPE_COUNT * obj_count;
        std::vector<HitGroupRecord> hitgroup_records_list(count_records);

        // Note: Fill SBT record array the same order that acceleration structure is built.
        int sbt_idx = 0; // Index to track current record.

        // TODO: Material params - arbitrary right now

        for (int i = 0; i < helistat_list.size(); i++) {
            // Configure Heliostat SBT record.
            OPTIX_CHECK(optixSbtRecordPackHeader(m_state.radiance_mirror_prog_group, &hitgroup_records_list[sbt_idx]));
            hitgroup_records_list[sbt_idx].data.geometry_data.setRectangleParabolic(helistat_list[i]);
            hitgroup_records_list[sbt_idx].data.material_data.mirror = {
                0.875425f, // Reflectivity.
                0.0f,  // Transmissivity.
                0.0f,  // Slope error.
                0.0f   // Specularity error.
            };
            sbt_idx++;
        }

        for (int i = 0; i < num_receivers; i++) {
            // Configure Receiver SBT record.
            OPTIX_CHECK(optixSbtRecordPackHeader(
                m_state.radiance_receiver_prog_group,
                &hitgroup_records_list[sbt_idx]));
            hitgroup_records_list[sbt_idx].data.geometry_data.setParallelogram(receiver_list[i]);
            hitgroup_records_list[sbt_idx].data.material_data.receiver = {
                0.95f, // Reflectivity.
                0.0f,  // Transmissivity.
                0.0f,  // Slope error.
                0.0f   // Specularity error.
            };
            sbt_idx++;
        }

        // Allocate memory for hitgroup records on the device.
        CUdeviceptr d_hitgroup_records;
        size_t      sizeof_hitgroup_record = sizeof(HitGroupRecord);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_hitgroup_records),
            sizeof_hitgroup_record * count_records
        ));

        // Copy hitgroup records from host to device.
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_hitgroup_records),
            hitgroup_records_list.data(),
            sizeof_hitgroup_record * count_records,
            cudaMemcpyHostToDevice
        ));

        // Configure the SBT hitgroup fields.
        m_state.sbt.hitgroupRecordBase = d_hitgroup_records;             // Base address of hitgroup records.
        m_state.sbt.hitgroupRecordCount = count_records;                  // Total number of hitgroup records.
        m_state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof_hitgroup_record);  // Stride size.
    }
}
