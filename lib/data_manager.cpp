#include "soltrace_system.h"
#include "data_manager.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "data_manager.h"

dataManager::dataManager() : launch_params_D(nullptr) {
	
    // Initialize launch parameters with default values
	launch_params_H.width = 10;
	launch_params_H.height = 1;
	launch_params_H.max_depth = 5;

	launch_params_H.hit_point_buffer = nullptr;
	launch_params_H.sun_vector = make_float3(0.0f, 0.0f, 10.0f);
	launch_params_H.max_sun_angle = 0.0f;

	launch_params_H.sun_v0 = make_float3(0.0f, 0.0f, 0.0f);
	launch_params_H.sun_v1 = make_float3(0.0f, 0.0f, 0.0f);
	launch_params_H.sun_v2 = make_float3(0.0f, 0.0f, 0.0f);
	launch_params_H.sun_v3 = make_float3(0.0f, 0.0f, 0.0f);
}

dataManager::~dataManager() {
	cleanup(); 
}


soltrace::LaunchParams* dataManager::getDeviceLaunchParams() const { return launch_params_D; }


void dataManager::allocateLaunchParams() {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&launch_params_D), sizeof(LaunchParams)));
}

void dataManager::updateLaunchParams() {
    CUDA_CHECK(cudaMemcpy(launch_params_D, &launch_params_H, sizeof(LaunchParams), cudaMemcpyHostToDevice));
}

void dataManager::allocateGeometryDataArray(const std::vector<std::shared_ptr<Element>> element_list) {

	// resize geometry_data_array_H to the number of elements
	geometry_data_array_H.resize(element_list.size());

	for (int i = 0; i < element_list.size(); i++) {

		geometry_data_array_H[i] = element_list[i]->toDeviceGeometryData();

	}

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&geometry_data_array_D),
        geometry_data_array_H.size() * sizeof(GeometryDataST)));

	CUDA_CHECK(cudaMemcpy(geometry_data_array_D, geometry_data_array_H.data(),
		geometry_data_array_H.size() * sizeof(GeometryDataST), cudaMemcpyHostToDevice));
	// make sure launch_params_H is updated with the new geometry data array
	launch_params_H.geometry_data_array = geometry_data_array_D;

	// print out the geometry data array type for debugging 
	//for (int i = 0; i < geometry_data_array_H.size(); i++) {
	//	std::cout << "geometry_data_array_H[" << i << "] = " << geometry_data_array_H[i].type << std::endl;
	//}

}

void dataManager::cleanup() {
	CUDA_CHECK(cudaFree(launch_params_D));
	launch_params_D = nullptr;

	CUDA_CHECK(cudaFree(geometry_data_array_D));
	geometry_data_array_D = nullptr;
}
