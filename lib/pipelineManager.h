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

    // Functions to create individual program groups.
    void createSunProgram();
    void createMirrorProgram();
    void createReceiverProgram();
    void createMissProgram();

private:
    SoltraceState& m_state;
    std::vector<OptixProgramGroup> m_program_groups;
};

#endif  // PIPELINEMANAGER_H
