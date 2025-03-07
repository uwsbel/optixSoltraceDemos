#pragma once
#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <cuda/Soltrace.h>

using namespace soltrace;


// Class to manage data on the host and device.

class dataManager {
public:
    // Host copy of launch parameters.
    soltrace::LaunchParams host_launch_params;
    // Device pointer to launch parameters.
    soltrace::LaunchParams* device_launch_params;

    dataManager();
    ~dataManager();

    soltrace::LaunchParams* getDeviceLaunchParams() const;

    void allocateLaunchParams();

	// todo : might need a different name to indicate that this function is copying the host launch params to the device 
    void updateLaunchParams();
};

#endif  // DATAMANAGER_H
