#pragma once
#ifndef PIPELINEMANAGER_H
#define PIPELINEMANAGER_H

#include <string>
#include <vector>
#include "optix.h"
#include <cuda/Soltrace.h>

class pipelineManager {
public:
    pipelineManager(SoltraceState& state);
    ~pipelineManager();

    std::string loadPtxFromFile(const std::string& kernelName);
    void loadModules();
    void createPipeline();
    OptixPipeline getPipeline() const;

	// create all the sun, mirror, receiver, and miss programs
    void createSunProgram();  // note: now we only have one source as ray gen, we can extend to more sources in here.

    void createMirrorPrograms();
    void createReceiverProgram();
    void createMissProgram();

	// helper function to create a hit group program given the intersection and closest hit programs
    void createHitGroupProgram(OptixProgramGroup& group,
                               OptixModule intersectionModule, 
                               const char* intersectionFunc,
                               OptixModule closestHitModule, 
                               const char* closestHitFunc);

private:
    SoltraceState& m_state;
    std::vector<OptixProgramGroup> m_program_groups;   // store all the program groups
};

#endif  // PIPELINEMANAGER_H
