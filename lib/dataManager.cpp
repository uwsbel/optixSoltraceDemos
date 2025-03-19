#include "SolTrSystem.h"
#include "dataManager.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "dataManager.h"

dataManager::dataManager() : device_launch_params(nullptr) { 
	
    // Initialize launch parameters with default values
    host_launch_params.width = 10;
	host_launch_params.height = 1;
	host_launch_params.max_depth = 5;

	host_launch_params.hit_point_buffer = nullptr;
	host_launch_params.sun_vector = make_float3(0.0f, 0.0f, 10.0f);
	host_launch_params.max_sun_angle = 0.0f;

	host_launch_params.sun_v0 = make_float3(0.0f, 0.0f, 0.0f);
	host_launch_params.sun_v1 = make_float3(0.0f, 0.0f, 0.0f);
	host_launch_params.sun_v2 = make_float3(0.0f, 0.0f, 0.0f);
	host_launch_params.sun_v3 = make_float3(0.0f, 0.0f, 0.0f);
}

dataManager::~dataManager() {
	cleanup(); 
}


soltrace::LaunchParams* dataManager::getDeviceLaunchParams() const { return device_launch_params; }


void dataManager::allocateLaunchParams() {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_launch_params), sizeof(LaunchParams)));
}

void dataManager::updateLaunchParams() {
    CUDA_CHECK(cudaMemcpy(device_launch_params, &host_launch_params, sizeof(LaunchParams), cudaMemcpyHostToDevice));
}

void dataManager::cleanup() {
    if (device_launch_params) {
        CUDA_CHECK(cudaFree(device_launch_params));
        device_launch_params = nullptr;
    }
}