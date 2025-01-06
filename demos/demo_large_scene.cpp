// TODO: 
// refactor, so things like loading shaders and creating modules are in separate cpp files
// obj count is hardcoded, make it dynamic
// aabb_input_flags should be the size of the object count


#include <cuda_runtime.h>
#include <sampleConfig.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <sutil/Exception.h>
#include <sutil/Record.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <chrono>

#include <iomanip>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>


#include <cuda/Soltrace.h>
#include <lib/geometry.h>

typedef sutil::Record<soltrace::HitGroupData> HitGroupRecord;

const int      max_trace = 5;

// Print a float3 structure
void printFloat3(const char* label, const float3& vec) {
    std::cout << label << ": (" << vec.x << ", " << vec.y << ", " << vec.z << ")\n";
}

// Load ptx given shader strings 
std::string loadPtxFromFile(const std::string& kernel_name) {
    // Construct the full PTX file path based on the kernel name
    const std::string ptx_file = std::string(SAMPLES_PTX_DIR) + "demo_large_scene_generated_" + kernel_name + ".cu.ptx";

    std::cout << "PTX file name: " << ptx_file << "\n";
    // Read the PTX file into a string
    std::ifstream f(ptx_file);
    if (!f.good()) {
        throw std::runtime_error("PTX file not found: " + ptx_file);
    }

    std::stringstream source_buffer;
    source_buffer << f.rdbuf();
    return source_buffer.str(); // Return the PTX content
}


// Create OptiX modules for different components of the application.
// Modules correspond to different functionality, such as geometry handling, materials, and the sun.
void createModules( SoltraceState &state )
{
    // Options to control optimization and debugging settings.
    OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    // Create geometry module.
    {

        size_t inputSize = 0;   // Variable to store the size of the CUDA input source.
        //const char* input = sutil::getInputData(
        //    "optixSoltraceDemo",                      // No additional sample name.
        //    SAMPLES_PTX_DIR,              // Use your project's build directory.
        //    "parallelogram.cu",           // Name of the CUDA file.
        //    inputSize                     // Output: Size of the input CUDA source code.
        //);

		std::string ptx = loadPtxFromFile("parallelogram");

        OPTIX_CHECK_LOG(optixModuleCreate(
            state.context,                       // OptiX context for the application.
            &module_compile_options,             // Module compilation options.
            &state.pipeline_compile_options,     // Pipeline-level compile options.
            ptx.c_str(),                               // CUDA source code as input.
            ptx.size(),                           // Size of the CUDA source code.
            LOG, &LOG_SIZE,                      // Logs for diagnostic output.
            &state.geometry_module               // Output: Handle for the compiled module.
        ));
    }

    // Create shading/materials module.
    {
		std::string ptx = loadPtxFromFile("materials");
        OPTIX_CHECK_LOG(optixModuleCreate(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            LOG, &LOG_SIZE,
            &state.shading_module
        ));
    }

    // Create the sun module.
    {
        std::string ptx = loadPtxFromFile("sun");

        OPTIX_CHECK_LOG(optixModuleCreate(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            LOG, &LOG_SIZE,
            &state.sun_module
        ));
    }
}

// Create program group for the sun's ray generation program.
static void createSunProgram( SoltraceState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           sun_prog_group;                 // Handle for the sun program group.
    OptixProgramGroupOptions    sun_prog_group_options = {};    // Options for the program group (none needed in this case).
    OptixProgramGroupDesc       sun_prog_group_desc = {};       // Descriptor to define the program group.
    
    // Specify the kind of program group (Ray Generation).
    sun_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    // Link the ray generation program to the sun module and specify the function name.
    sun_prog_group_desc.raygen.module            = state.sun_module;
    sun_prog_group_desc.raygen.entryFunctionName = "__raygen__sun_source";

    // Create the program group
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,                 // OptiX context.
        &sun_prog_group_desc,          // Descriptor defining the program group.
        1,                             // Number of program groups to create (1 in this case).
        &sun_prog_group_options,       // Options for the program group.
        LOG, &LOG_SIZE,                // Logs to capture diagnostic information.
        &sun_prog_group                // Output: Handle for the created program group.
    ));

    program_groups.push_back(sun_prog_group);
    state.raygen_prog_group = sun_prog_group;
}

// Create program group for handling rays interacting with mirrors.
static void createMirrorProgram( SoltraceState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_mirror_prog_group;                 // Handle for the mirror program group.
    OptixProgramGroupOptions    radiance_mirror_prog_group_options = {};    // Options for the program group (none needed).
    OptixProgramGroupDesc       radiance_mirror_prog_group_desc = {};       // Descriptor for the program group.

    // Specify the kind of program group (Hit Group for handling intersections and shading).
    radiance_mirror_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    // Link the intersection shader (geometry handling) to the geometry module.
    radiance_mirror_prog_group_desc.hitgroup.moduleIS               = state.geometry_module;
    radiance_mirror_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__parallelogram";
    // Link the closest-hit shader (shading logic) to the shading module.
    radiance_mirror_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    radiance_mirror_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__mirror";
    // No any-hit shader is used in this configuration (set to nullptr).
    radiance_mirror_prog_group_desc.hitgroup.moduleAH               = nullptr;
    radiance_mirror_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    // Create the program group
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &radiance_mirror_prog_group_desc,
        1,
        &radiance_mirror_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_mirror_prog_group ) );

    program_groups.push_back(radiance_mirror_prog_group);
    state.radiance_mirror_prog_group = radiance_mirror_prog_group;
}

// Create program group for handling rays interacting with the receiver.
static void createReceiverProgram( SoltraceState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_receiver_prog_group;               // Handle for the receiver program group.
    OptixProgramGroupOptions    radiance_receiver_prog_group_options = {};  // Options for the program group (none needed).
    OptixProgramGroupDesc       radiance_receiver_prog_group_desc = {};     // Descriptor for the program group.

    // Specify the kind of program group (Hit Group for handling intersections and shading).
    radiance_receiver_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    // Link the intersection shader (geometry handling) to the geometry module.
    radiance_receiver_prog_group_desc.hitgroup.moduleIS               = state.geometry_module;
    radiance_receiver_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__parallelogram";
    // Link the closest-hit shader (shading logic) to the shading module.
    radiance_receiver_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    radiance_receiver_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__receiver";
    // No any-hit shader is used in this configuration (set to nullptr).
    radiance_receiver_prog_group_desc.hitgroup.moduleAH               = nullptr;
    radiance_receiver_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    // Create the program group
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &radiance_receiver_prog_group_desc,
        1,
        &radiance_receiver_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_receiver_prog_group ) );

    program_groups.push_back(radiance_receiver_prog_group);
    state.radiance_receiver_prog_group = radiance_receiver_prog_group;
}

// Create program group for handling rays that miss all geometry.
static void createMissProgram( SoltraceState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_miss_prog_group;       // Handle for the miss program group.
    OptixProgramGroupOptions    miss_prog_group_options = {};   // Options for the program group (none needed).
    OptixProgramGroupDesc       miss_prog_group_desc = {};      // Descriptor for the program group.
    
    // Specify the kind of program group (Miss Program for handling missed rays).
    miss_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    // Link the miss shader (background or environment shading) to the shading module.
    miss_prog_group_desc.miss.module             = state.shading_module;
    miss_prog_group_desc.miss.entryFunctionName  = "__miss__ms";

    // Create the program grou
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1,
        &miss_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_miss_prog_group ) );

    program_groups.push_back(radiance_miss_prog_group);
    state.radiance_miss_prog_group = radiance_miss_prog_group;
}

// Create and configure the OptiX pipeline.
// The pipeline is a core component in OptiX that ties together all program groups, modules, 
// and other configurations needed for ray tracing execution.
void createPipeline( SoltraceState &state )
{
    std::vector<OptixProgramGroup> program_groups;

    // Configure the pipeline compile options.
    state.pipeline_compile_options = {
        false,                                                  // usesMotionBlur: Disable motion blur.
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags: Allow only a single GAS.
        2,    /* RadiancePRD uses 5 payloads */                 // numPayloadValues
        5,    /* Parallelogram intersection uses 5 attrs */     // numAttributeValues 
        OPTIX_EXCEPTION_FLAG_NONE,                              // exceptionFlags
        "params"                                                // pipelineLaunchParamsVariableName
    };

    // Prepare modules and program groups
    createModules( state );
    createSunProgram( state, program_groups );
    createMirrorProgram( state, program_groups );
    createReceiverProgram( state, program_groups );
    createMissProgram( state, program_groups );

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = max_trace; // Maximum recursion depth for ray tracing.
    
    // Create the OptiX pipeline by linking the program groups.
    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,                        // OptiX context.
        &state.pipeline_compile_options,      // Compile options for the pipeline.
        &pipeline_link_options,               // Link options for the pipeline.
        program_groups.data(),                // Array of program groups.
        static_cast<unsigned int>(program_groups.size()), // Number of program groups.
        LOG, &LOG_SIZE,                       // Logs for diagnostics.
        &state.pipeline                       // Output: Handle for the created pipeline.
    ));

    // Compute and configure the stack sizes for the pipeline.
    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, state.pipeline ) );
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
        state.pipeline,                               // Pipeline to configure.
        direct_callable_stack_size_from_traversal,    // Stack size for direct callable traversal.
        direct_callable_stack_size_from_state,        // Stack size for direct callable state.
        continuation_stack_size,                      // Stack size for continuation stack.
        1                                            // maxTraversableDepth: Maximum depth of traversable hierarchy.
    ));
}

// Ccreate and configure the Shader Binding Table (SBT).
// The SBT is a crucial data structure in OptiX that links geometry and ray types
// with their corresponding programs (ray generation, miss, and hit group).
void createSBT( SoltraceState &state, std::vector<GeometryData::Parallelogram>& helistat_list, std::vector<GeometryData::Parallelogram> receiver_list)
{
	int num_heliostats = helistat_list.size();
	int num_receivers = receiver_list.size();
    int obj_count = helistat_list.size() + receiver_list.size();
   
    // Ray generation program record
    {
        CUdeviceptr d_raygen_record;                   // Device pointer to hold the raygen SBT record.
        size_t      sizeof_raygen_record = sizeof( sutil::EmptyRecord );
        
        // Allocate memory for the raygen SBT record on the device.
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_raygen_record ),
            sizeof_raygen_record ) );

        sutil::EmptyRecord rg_sbt;  // Host representation of the raygen SBT record.

        // Pack the program header into the raygen SBT record.
        optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt );

        // Copy the raygen SBT record from host to device.
        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_raygen_record ),
            &rg_sbt,
            sizeof_raygen_record,
            cudaMemcpyHostToDevice
        ) );

        // Assign the device pointer to the raygenRecord field in the SBT.
        state.sbt.raygenRecord = d_raygen_record;
    }

    // Miss program record
    {
        CUdeviceptr d_miss_record;
        size_t sizeof_miss_record = sizeof( sutil::EmptyRecord );
        
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_miss_record ),
            sizeof_miss_record*soltrace::RAY_TYPE_COUNT ) );

        sutil::EmptyRecord ms_sbt[soltrace::RAY_TYPE_COUNT];
        // Pack the program header into the first miss SBT record.
        optixSbtRecordPackHeader( state.radiance_miss_prog_group, &ms_sbt[0] );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_miss_record ),
            ms_sbt,
            sizeof_miss_record*soltrace::RAY_TYPE_COUNT,
            cudaMemcpyHostToDevice
        ) );

        // Configure the SBT miss program fields.
        state.sbt.missRecordBase          = d_miss_record;                   // Base address of the miss records.
        state.sbt.missRecordCount         = soltrace::RAY_TYPE_COUNT;        // Number of miss records.
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof_miss_record );    // Stride between miss records.
    }

    // Hitgroup program record
    {
        // Total number of hitgroup records : one per ray type per object.
        const size_t count_records = soltrace::RAY_TYPE_COUNT * obj_count;
		std::vector<HitGroupRecord> hitgroup_records_list(count_records);

        // Note: Fill SBT record array the same order that acceleration structure is built.
        int sbt_idx = 0; // Index to track current record.

        // TODO: Material params - arbitrary right now

		for (int i = 0; i < num_heliostats; i++) {
			// Configure Heliostat SBT record.
            OPTIX_CHECK(optixSbtRecordPackHeader(
				state.radiance_mirror_prog_group,
				&hitgroup_records_list[sbt_idx]));
			hitgroup_records_list[sbt_idx].data.geometry_data.setParallelogram(helistat_list[i]);
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
				state.radiance_receiver_prog_group,
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
        size_t      sizeof_hitgroup_record = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_hitgroup_records ),
            sizeof_hitgroup_record*count_records
        ) );

        // Copy hitgroup records from host to device.
        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_hitgroup_records ),
            hitgroup_records_list.data(),
            sizeof_hitgroup_record*count_records,
            cudaMemcpyHostToDevice
        ) );

        // Configure the SBT hitgroup fields.
        state.sbt.hitgroupRecordBase            = d_hitgroup_records;             // Base address of hitgroup records.
        state.sbt.hitgroupRecordCount           = count_records;                  // Total number of hitgroup records.
        state.sbt.hitgroupRecordStrideInBytes   = static_cast<uint32_t>( sizeof_hitgroup_record );  // Stride size.
    }
}

// Callback function for logging messages from the OptiX context.
static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    // Format and print the log message to the standard error stream.
    // The message includes:
    // - The log level (e.g., error, warning, info).
    // - A tag for categorization (e.g., "API", "Internal").
    // - The actual message content.
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
        << message << "\n";
}

// Create and initialize an OptiX context.
void createContext( SoltraceState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // Use the current CUDA context (zero indicates current).

    // Set OptiX device context options.
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;    // Optional: Set a logging callback function for debugging.
    options.logCallbackLevel          = 4;                  // Verbosity level for logging (e.g., errors, warnings, etc.).
    
    // Create and store OptiX device context
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
    state.context = context;
}

// Initialize the launch parameters used for ray tracing.
void initLaunchParams( SoltraceState& state )
{
    // Set maximum ray depth.
    state.params.max_depth = max_trace;

    // Allocate memory for the hit point buffer.
    // The size depends on the sun (ray generation resolution) parameters and the maximum ray depth.
    // Start the timer
    auto start_buff = std::chrono::high_resolution_clock::now();
    const size_t hit_point_buffer_size = state.params.width * state.params.height * sizeof(float4) * state.params.max_depth;
    
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.hit_point_buffer),
        hit_point_buffer_size
    ));
    CUDA_CHECK(cudaMemset(state.params.hit_point_buffer, 0, hit_point_buffer_size));

    // Reflected direction buffer (commented out for memory saving).
    /*
    const size_t reflected_dir_buffer_size = state.params.width * state.params.height * sizeof(float4) * state.params.max_depth;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.reflected_dir_buffer),
        reflected_dir_buffer_size
    ));
    CUDA_CHECK(cudaMemset(state.params.reflected_dir_buffer, 0, reflected_dir_buffer_size));
    */ 

    // Create a CUDA stream for asynchronous operations.
    CUDA_CHECK( cudaStreamCreate( &state.stream ) );

    // Allocate memory for device-side launch parameters.
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( soltrace::LaunchParams ) ) );

    // Link the GAS handle.
    state.params.handle = state.gas_handle;
}

// Clean up resources and deallocate memory
void cleanupState( SoltraceState& state )
{
    OPTIX_CHECK( optixPipelineDestroy     ( state.pipeline                ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.raygen_prog_group       ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_mirror_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_receiver_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_miss_prog_group         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.shading_module          ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.geometry_module         ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context                 ) );


    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer    ) ) );
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.hit_point_buffer)));
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params               ) ) );
}

// Check if components of a float4 are all zero. 
// If they are, it returns TRUE; otherwise, it returns FALSE.
// TODO: Move to util file
bool allZeros(float4 element) {
    return (element.y == 0 && element.z == 0 && element.w == 0);
}

// Write a vector of float4 data to a CSV file, filtering out rows based on allZeros.
void writeVectorToCSV(const std::string& filename, const std::vector<float4>& data) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open the file " << filename << " for writing." << std::endl;
        return;
    }

    // Write header
    outFile << "number,stage,loc_x,loc_y,loc_z\n";

    // Write data
    int ray_idx = 1;
    int idx = 0;
    for (const auto& element : data) {
        ++idx;
        if (idx <= max_trace && !allZeros(element)) {
            outFile << ray_idx << "," << element.x << "," << element.y << "," << element.z << "," << element.w << "\n";
        }
        else {
            idx = 0;
            ++ray_idx;
        }
    }

    outFile.close();
    std::cout << "Data successfully written to " << filename << std::endl;
}

int main(int argc, char* argv[])

{
    int num_elements = 1700;
	int num_sun_points = 1000000;

	if (argc > 1) {
        num_elements = std::stoi(argv[1]);
        num_sun_points = std::stoi(argv[2]);
	}

    std::string heliostat_data_file = "../data/field_" + std::to_string(num_elements) + "_elements.csv";
	std::vector<GeometryData::Parallelogram> heliostats_list  = GenerateHeliostatsFromFile(heliostat_data_file);

    std::vector<GeometryData::Parallelogram> receiver_list;

 //   GeometryData::Parallelogram receiver1(
	//	make_float3(4.0f, 0.0f, 0.0f),    // v1
	//	make_float3(0.0f, 0.0f, 7.0f),    // v2
	//	make_float3(-4.5f, 0.0f, 76.5f)     // anchor
	//);
	//receiver_list.push_back(receiver1);

 //   GeometryData::Parallelogram receiver2(
 //       make_float3(4.0f, 0.0f, 0.0f),    // v1
 //       make_float3(0.0f, 0.0f, 7.0f),    // v2
 //       make_float3(0.5f, 0.0f, 76.5f)     // anchor
 //   );
	//receiver_list.push_back(receiver2);

    GeometryData::Parallelogram receiver(
    	make_float3(9.0f, 0.0f, 0.0f),    // v1
    	make_float3(0.0f, 0.0f, 7.0f),    // v2
    	make_float3(-4.5f, 0.0f, 76.5f)     // anchor
    );
    receiver_list.push_back(receiver);


    SoltraceState state;
	std::cout << "Starting Soltrace OptiX simulation..." << std::endl;
	std::cout << "samples ptx dir: " << SAMPLES_PTX_DIR << std::endl;

    /*
    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();
    */

    try
    {
        // Initialize simulation parameters
        //state.params.sun_center = make_float3(0.0f, 0.0f, state.params.sun_radius);state.params.sun_center = make_float3(0.0f, 0.0f, state.params.sun_radius);    // z-component computed in raygen based on dims of sun
        state.params.sun_vector = make_float3(-235.034f, -5723.13f, 8196.98f);
        state.params.max_sun_angle = 0.00465;     // 4.65 mrad
        state.params.num_sun_points = num_sun_points;

        state.params.width  = state.params.num_sun_points;
        state.params.height = 1;

        // Initialize OptiX components
        createContext(state);
        createGeometry(state, heliostats_list, receiver_list);
        createPipeline(state);
        createSBT(state, heliostats_list, receiver_list);
        initLaunchParams(state);

        // Copy launch parameters to device memory
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params), &state.params,
            sizeof(soltrace::LaunchParams), cudaMemcpyHostToDevice, state.stream));

        // Start the timer
        auto start = std::chrono::high_resolution_clock::now();

        // Launch the OptiX pipeline
        OPTIX_CHECK( optixLaunch(
        state.pipeline,     // OptiX pipeline
        state.stream,       // CUDA stream used for this launch
        reinterpret_cast<CUdeviceptr>( state.d_params ),    // Device pointer to the launch parameters structure.
        sizeof( soltrace::LaunchParams ),                   // Size of launch parameters structure
        &state.sbt,          // Shader Binding Table(SBT): contains the pointers to program groups and associated data.
        state.params.width,  // Number of threads to launch along the X dimension.
        state.params.height, // Number of threads to launch along the Y dimension.
        1                    // Number of threads to launch along the Z dimension. Often set to 1 for 2D images or other non-volumetric workloads.
        ) );

        CUDA_SYNC_CHECK();

        // Stop the timer
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Execution time ray launch: " << duration_ms.count() << " milliseconds" << std::endl;

        /*
        // Stop the timer
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Execution time full sim: " << duration_ms.count() << " milliseconds" << std::endl;
        */

        // Copy hit point results from device memory
        std::vector<float4> hp_output_buffer(state.params.width * state.params.height * state.params.max_depth);
        CUDA_CHECK(cudaMemcpy(hp_output_buffer.data(), state.params.hit_point_buffer, state.params.width * state.params.height * state.params.max_depth * sizeof(float4), cudaMemcpyDeviceToHost));

        // Copy reflected direction results from device memory
        /*
        std::vector<float4> rd_output_buffer(state.params.width * state.params.height * state.params.max_depth);
        CUDA_CHECK(cudaMemcpy(rd_output_buffer.data(), state.params.reflected_dir_buffer, state.params.width * state.params.height * state.params.max_depth * sizeof(float4), cudaMemcpyDeviceToHost));
        */
		std::string output_filename = std::to_string(num_elements) + "_elements_" + std::to_string(num_sun_points) + "_rays.csv";
        writeVectorToCSV(output_filename, hp_output_buffer);

        cleanupState(state);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}