#pragma once
#include <optix.h>
#include <sutil/vec_math.h>
#include <vector>
#include <cuda/GeometryDataST.h>
#include <cuda/Soltrace.h>
#include <string>
#include <SoltraceState.h>

int foo(int a, GeometryData::Parallelogram test);

std::vector<GeometryData::Parallelogram> GenerateHeliostatsFromFile(std::string filename);

void createGeometry(SoltraceState& state, std::vector<GeometryData::Parallelogram>& helistat_list);