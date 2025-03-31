#include "SolTrSystem.h"
#include "dataManager.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <SoltraceType.h>
#include <Element.h>


// TODO: optix related type should go into one header file
typedef sutil::Record<soltrace::HitGroupData> HitGroupRecord;

SolTrSystem::SolTrSystem(int numSunPoints)
    : m_num_sunpoints(numSunPoints)
{
    m_verbose = false;
    // TODO: think about m_state again, attach or not attach
    geometry_manager = std::make_shared<geometryManager>(m_state);
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
    //cleanup();
}

void SolTrSystem::initialize() {

	Vector3d sun_vec = m_sun_vector.normalized(); // normalize the sun vector

    // initialize soltrace state variable 
    m_state.params.sun_vector = make_float3(sun_vec[0], sun_vec[1], sun_vec[2]);
    m_state.params.max_sun_angle = 0.00465;     // 4.65 mrad

    // set up input related to sun
    data_manager->host_launch_params.sun_vector = m_state.params.sun_vector;
    data_manager->host_launch_params.max_sun_angle = 0.00465;     // 4.65 mrad


    // compute aabb box 
    geometry_manager->populate_aabb_list(m_element_list);

    // compute sun plane vertices
    geometry_manager->compute_sun_plane();

    // handles geoemtries building
    geometry_manager->create_geometries();

    // Pipeline setup.
    pipeline_manager->createPipeline();

    createSBT();


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

    // TODO: need to get rid of the m_state.params 
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
}

void SolTrSystem::writeOutput(const std::string& filename) {
    int output_size = data_manager->host_launch_params.width * data_manager->host_launch_params.height * data_manager->host_launch_params.max_depth;
    std::vector<float4> hp_output_buffer(output_size);
    CUDA_CHECK(cudaMemcpy(hp_output_buffer.data(), data_manager->host_launch_params.hit_point_buffer, output_size * sizeof(float4), cudaMemcpyDeviceToHost));

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


    CUDA_CHECK(cudaDeviceSynchronize());
    // destroy pipeline related resources
	pipeline_manager->cleanup();

    // destory CUDA stream
	if (m_state.stream) {
		CUDA_CHECK(cudaStreamDestroy(m_state.stream));
	}

    OPTIX_CHECK(optixDeviceContextDestroy(m_state.context));


    // Free OptiX shader binding table (SBT) memory
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.sbt.hitgroupRecordBase)));

    // Free OptiX GAS output buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.d_gas_output_buffer)));

    // Free device-side launch parameter memory
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.params.hit_point_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.d_params)));

    data_manager->cleanup();


}

// Create and configure the Shader Binding Table (SBT).
// The SBT is a crucial data structure in OptiX that links geometry and ray types
// with their corresponding programs (ray generation, miss, and hit group).
void SolTrSystem::createSBT(){

	int obj_count = m_element_list.size();  // Number of objects in the scene (heliostats + receivers)

    // Ray generation program record
	// TODO: Move this out of here, has nothing to do with the scene geometry
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
        // TODO: separate number of heliostats and receivers 

		int num_heliostats = m_element_list.size() - 1; // Assuming the last element is the receiver
        int num_receivers = 1;

        for (int i = 0; i < num_heliostats; i++) {

			auto element = m_element_list.at(i);

            // TODO: setRectangleParabolic should be matched automaitcally.
			SurfaceType surface_type = element->get_surface_type();
			ApertureType aperture_type = element->get_aperture_type();
			SurfaceApertureMap map = { surface_type, aperture_type };
            


            OPTIX_CHECK(optixSbtRecordPackHeader(pipeline_manager->getMirrorProgram(map), 
                                                 &hitgroup_records_list[sbt_idx]));
            // assign geometry data to the corresponding hitgroup record 
            hitgroup_records_list[sbt_idx].data.geometry_data = element->toDeviceGeometryData();
            hitgroup_records_list[sbt_idx].data.material_data.mirror = {
                                                 0.875425f, // Reflectivity.
                                                 0.0f,  // Transmissivity.
                                                 0.0f,  // Slope error.
                                                 0.0f   // Specularity error.
            };
            sbt_idx++;
        }

        // TODO: perform the same way 
        for (int i = num_heliostats; i < m_element_list.size(); i++) {
            auto element = m_element_list.at(i);
            // Configure Receiver SBT record.
            SurfaceType surface_type = element->get_surface_type();
            OPTIX_CHECK(optixSbtRecordPackHeader(
                pipeline_manager->getReceiverProgram(surface_type),
                &hitgroup_records_list[sbt_idx]));
            hitgroup_records_list[sbt_idx].data.geometry_data = element->toDeviceGeometryData();
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

void SolTrSystem::AddElement(std::shared_ptr<Element> e)
{
    m_element_list.push_back(e);
}