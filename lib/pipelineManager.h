#pragma once
#ifndef PIPELINEMANAGER_H
#define PIPELINEMANAGER_H

#include <string>
#include <vector>
#include "optix.h"
#include <cuda/Soltrace.h>
#include <SoltraceType.h>

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

	// a bunch of get methods to get the program groups
	// TODO need to add implementation for sun, miss and receiver programs 
	OptixProgramGroup getRaygenProgram() const;
	OptixProgramGroup getMissProgram() const;
	OptixProgramGroup getReceiverProgram() const;

	// given heliostat type find the corresponding mirror programming group
    OptixProgramGroup getMirrorProgram(SurfaceApertureMap map) const;


private:
    SoltraceState& m_state;
    std::vector<OptixProgramGroup> m_program_groups;   // store all the program groups

    // possible members ... 
    // index for the program groups 
    // TODO: this is hardcoded for now .... 
    // need to think about what to do with this 
	int num_raygen_programs = 1;
    int num_heliostat_programs = 2;
    int num_receiver_programs = 2; 
	int num_miss_programs = 1;
};

#endif  // PIPELINEMANAGER_H
