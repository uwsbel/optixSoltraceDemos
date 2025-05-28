#include "soltrace_system.h"
#include <cfloat> 
#include <cuda/GeometryDataST.h>
#include <cuda/Soltrace.h>
#include <sun_utils.h>
#include <soltrace_state.h>
#include <optix.h>
#include <vector>

// TODO 
// need to extractr sun plane 



// this one need to be refactored
// note that the sun plane is part of the pre-processing, need to be able to 
// compute sun plane fast with updated scene geometry and sun vector
// ask allie about the abs() part, the distance for buffer calculation and the bin size for flux map computation

void GeometryManager::collect_geometry_info(const std::vector<std::shared_ptr<Element>>& element_list,
                                            LaunchParams& params) {    
    m_aabb_list_H.clear(); // Clear the existing AABB list
    m_sbt_index_H.clear(); // Clear the existing SBT index list
	m_geometry_data_array_H.clear(); // Clear the existing geometry data array

	m_obj_counts = element_list.size(); // Number of objects in the scene

	// Resize
	m_aabb_list_H.resize(m_obj_counts);
	m_geometry_data_array_H.resize(m_obj_counts);
    m_sbt_index_H.resize(m_obj_counts);


    for (int i = 0; i < m_obj_counts; i++) {

		std::shared_ptr<Element> element = element_list[i];

        // Create an OptixAabb from the geometry data
        OptixAabb aabb;
        float3 m_min;
        float3 m_max;
        uint32_t sbt_offset = 0;

        if (element->get_aperture_type() == ApertureType::CIRCLE) {
            m_min = make_float3(0.0f, 0.0f, 0.0f); // Initialize min to a large value
            m_max = make_float3(0.0f, 0.0f, 0.0f); // Initialize max to a small value
        }


        if (element->get_aperture_type() == ApertureType::RECTANGLE) {
            element->compute_bounding_box();
            m_min.x = element->get_lower_bounding_box()[0];
            m_min.y = element->get_lower_bounding_box()[1];
            m_min.z = element->get_lower_bounding_box()[2];

            m_max.x = element->get_upper_bounding_box()[0];
            m_max.y = element->get_upper_bounding_box()[1];
            m_max.z = element->get_upper_bounding_box()[2];

            if (element->get_surface_type() == SurfaceType::PARABOLIC) {
                sbt_offset = static_cast<uint32_t>(OpticalEntityType::RECTANGLE_PARABOLIC_MIRROR);
            }
            else if (element->get_surface_type() == SurfaceType::FLAT) {
                sbt_offset = static_cast<uint32_t>(OpticalEntityType::RECTANGLE_FLAT_MIRROR);
            }
            else {
                // print error message 
                std::cerr << "Unknown surface type for element " << i << std::endl;
            }
        }


        aabb.minX = m_min.x;
        aabb.minY = m_min.y;
        aabb.minZ = m_min.z;

        aabb.maxX = m_max.x;
        aabb.maxY = m_max.y;
        aabb.maxZ = m_max.z;


		m_aabb_list_H[i] = aabb; // Store the AABB in the list
        m_sbt_index_H[i] = sbt_offset; // Store the SBT index
        m_geometry_data_array_H[i] = element_list[i]->toDeviceGeometryData();
    }


    // add receiver sbt_index
	std::shared_ptr<Element> receiver = element_list[m_obj_counts - 1];

    if (receiver->get_surface_type() == SurfaceType::FLAT) {
        m_sbt_index_H[m_obj_counts -1] = static_cast<uint32_t>(OpticalEntityType::RECTANGLE_FLAT_RECEIVER);
    }
    else if (receiver->get_surface_type() == SurfaceType::CYLINDER) {
        m_sbt_index_H[m_obj_counts -1] = static_cast<uint32_t>(OpticalEntityType::CYLINDRICAL_RECEIVER);
    }
    else {
		// print error message 
		std::cerr << "Error: Unknown surface type for receiver " << std::endl; 
    }

    // print out computed minimum distance 
	std::cout << "Minimum distance to sun plane: " << m_sun_plane_distance << std::endl;
}



// Build a GAS (Geometry Acceleration Structure) for the scene.
static void buildGas(
    const SoltraceState& state,
    const OptixAccelBuildOptions& accel_options,
    const OptixBuildInput& build_input,
    OptixTraversableHandle& gas_handle,
    CUdeviceptr& d_gas_output_buffer,
    size_t& scratch_bytes) {

    OptixAccelBufferSizes gas_buffer_sizes;     // Holds required sizes for temp and output buffers.
    CUdeviceptr d_temp_buffer_gas;              // Temporary buffer for building the GAS.

    // Query the memory usage required for building the GAS.
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &build_input,
        1,
        &gas_buffer_sizes));

	scratch_bytes = gas_buffer_sizes.tempSizeInBytes; // Store the scratch size for later use

    // Allocate memory for the temporary buffer on the device.
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer_gas),
        gas_buffer_sizes.tempSizeInBytes));

    // Non-compacted output and size of compacted GAS
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
        compactedSizeOffset + 8
    ));

    // Emit property to store the compacted GAS size.
    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    // Build the GAS.
    OPTIX_CHECK(optixAccelBuild(
        state.context,                                  // OptiX context
        0,                                              // CUDA stream (default is 0)
        &accel_options,                                 // Acceleration build options
        &build_input,                                   // Build inputs
        1,                                              // Number of build inputs
        d_temp_buffer_gas,                              // Temporary buffer
        gas_buffer_sizes.tempSizeInBytes,               // Size of temporary buffer
        d_buffer_temp_output_gas_and_compacted_size,    // Output buffer
        gas_buffer_sizes.outputSizeInBytes,             // Size of output buffer
        &gas_handle,                                    // Output handle
        &emitProperty,                                  // Emitted properties
        1));                                           // Number of emitted properties

    CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    // If the compacted GAS size is smaller, allocate a smaller buffer and compact the GAS
    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(state.context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));

        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}


void GeometryManager::compute_sun_plane_H(LaunchParams& params) {

    float3 sun_vector = params.sun_vector;

    float* sun_plane_dist_D;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sun_plane_dist_D), sizeof(float)));

    compute_d_on_gpu(reinterpret_cast<const OptixAabb*>(m_aabb_list_D), m_obj_counts, sun_vector, sun_plane_dist_D);

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(&m_sun_plane_distance),
                         reinterpret_cast<void*>(sun_plane_dist_D),
                         sizeof(float),
                         cudaMemcpyDeviceToHost));

    float3 sun_u, sun_v;
    float3 axis = (abs(sun_vector.x) < 0.9f) ? make_float3(1.0f, 0.0f, 0.0f) : make_float3(0.0f, 1.0f, 0.0f);
    sun_u = normalize(cross(axis, sun_vector));
    sun_v = normalize(cross(sun_vector, sun_u));
    // allocate uv bounds on device, float array of size four
    float sun_uv_bounds[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    // allocate memory for the sun_u_bounds on device
    float* sun_uv_bounds_D;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sun_uv_bounds_D), 4 * sizeof(float)));

    compute_uv_bounds_on_gpu(reinterpret_cast<const OptixAabb*>(m_aabb_list_D),
        m_obj_counts,
        m_sun_plane_distance,
        sun_vector,
        sun_u,
        sun_v,
        tan(params.max_sun_angle),
        sun_uv_bounds_D);

    // Copy the computed bounds back to the host
    cudaMemcpy(sun_uv_bounds, sun_uv_bounds_D, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    float u_min = sun_uv_bounds[0];
    float u_max = sun_uv_bounds[1];
    float v_min = sun_uv_bounds[2];
    float v_max = sun_uv_bounds[3];

    params.sun_v0 = u_min * sun_u + v_min * sun_v + m_sun_plane_distance * sun_vector;
    params.sun_v1 = u_max * sun_u + v_min * sun_v + m_sun_plane_distance * sun_vector;
    params.sun_v2 = u_max * sun_u + v_max * sun_v + m_sun_plane_distance * sun_vector;
    params.sun_v3 = u_min * sun_u + v_max * sun_v + m_sun_plane_distance * sun_vector;

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sun_plane_dist_D)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sun_uv_bounds_D)));

}

void GeometryManager::create_geometries(LaunchParams& params) {

    // Allocate memory on the device for the AABB array.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_aabb_list_D), m_obj_counts * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_aabb_list_D),
               m_aabb_list_H.data(), 
		       m_obj_counts * sizeof(OptixAabb),
               cudaMemcpyHostToDevice));

	compute_sun_plane_H(params);

    // populate aabb_input_flags vector, size of types, no rebuild
    std::vector<uint32_t> aabb_input_flags(NUM_OPTICAL_ENTITY_TYPES);
    for (int i = 0; i < NUM_OPTICAL_ENTITY_TYPES; i++) {
        aabb_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    }

	// device vector for SBT index, no need to rebuild
    CUdeviceptr d_sbt_index;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_index), m_obj_counts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_sbt_index),
                          m_sbt_index_H.data(),
        m_obj_counts * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Configure the input for the GAS build process.
    m_aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    m_aabb_input.customPrimitiveArray.aabbBuffers = &m_aabb_list_D;
    m_aabb_input.customPrimitiveArray.flags = aabb_input_flags.data();
    m_aabb_input.customPrimitiveArray.numSbtRecords = NUM_OPTICAL_ENTITY_TYPES;
    m_aabb_input.customPrimitiveArray.numPrimitives = m_obj_counts;
    m_aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_index;
    m_aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    m_aabb_input.customPrimitiveArray.primitiveIndexOffset = 0;

    // Set up acceleration structure (AS) build options.
    m_accel_build_options = {
		OPTIX_BUILD_FLAG_ALLOW_COMPACTION |   // enable compaction to reduce memory usage.
        OPTIX_BUILD_FLAG_ALLOW_UPDATE,        // allow update
        OPTIX_BUILD_OPERATION_BUILD           // operation type, build a new aceleration structure
    };

    // Build the GAS using the defined AABBs and options.
    buildGas(m_state,                        // input:  state with OptiX context.
             m_accel_build_options,          // input:  build options.
             m_aabb_input,                   // input:  AABB input description.
             m_state.gas_handle,             // output: traversable handle for the GAS.
             m_state.d_gas_output_buffer,    // output: device buffer for the GAS.
		     m_scratch_bytes);               // output: size of the scratch buffer 

	// store the scratch buffer for refit
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_scratch_refit), m_scratch_bytes));
}


void GeometryManager::update_geometry_info(const std::vector<std::shared_ptr<Element>>& element_list,
	LaunchParams& params) {
	// Recollect geometry info
	collect_geometry_info(element_list, params);

	CUdeviceptr d_aabb = m_aabb_input.customPrimitiveArray.aabbBuffers[0]; // get the device pointer to the AABB buffer

    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(d_aabb),
        m_aabb_list_H.data(),
        m_aabb_list_H.size() * sizeof(OptixAabb),
        cudaMemcpyHostToDevice, m_state.stream));

    m_accel_build_options.operation = OPTIX_BUILD_OPERATION_UPDATE; // set to update 

    //// GAS build
    //OPTIX_CHECK(optixAccelBuild(
    //    m_state.context,
    //    m_state.stream,
    //    &m_cached_build_options,
    //    &m_aabb_input, 
    //    1,
    //    m_d_scratch_refit, 
    //    m_scratch_bytes,
    //    m_state.d_gas_output_buffer,     // same output buffer
    //    nullptr, 
    //    0));

    // update sun plane as well.
}