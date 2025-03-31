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
    m_state.params.max_sun_angle = m_sun_angle;

    // set up input related to sun
    data_manager->host_launch_params.sun_vector = m_state.params.sun_vector;
    data_manager->host_launch_params.max_sun_angle = m_state.params.max_sun_angle;


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

bool SolTrSystem::readStinput(const char* filename) {
    FILE* fp = fopen(filename, "r");
	if (!fp)
	{
		printf("failed to open system input file\n");
		return -1;
	}

	printf("input file: %s\n", filename);
	if ( !read_system( fp ) )
	{
		printf("error in input file.\n");
		fclose(fp);
		return -1;
	}
	
	fclose(fp);

    return true;
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
            OPTIX_CHECK(optixSbtRecordPackHeader(
                m_state.radiance_receiver_prog_group,
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

std::vector<std::string> SolTrSystem::split(const std::string& str, const std::string& delim, bool ret_empty, bool ret_delim) {
	
    std::vector<std::string> list;

	char cur_delim[2] = {0,0};
	std::string::size_type m_pos = 0;
	std::string token;
	
	while (m_pos < str.length())
	{
		std::string::size_type pos = str.find_first_of(delim, m_pos);
		if (pos == std::string::npos)
		{
			cur_delim[0] = 0;
			token.assign(str, m_pos, std::string::npos);
			m_pos = str.length();
		}
		else
		{
			cur_delim[0] = str[pos];
			std::string::size_type len = pos - m_pos;			
			token.assign(str, m_pos, len);
			m_pos = pos + 1;
		}
		
		if (token.empty() && !ret_empty)
			continue;

		list.push_back( token );
		
		if ( ret_delim && cur_delim[0] != 0 && m_pos < str.length() )
			list.push_back( std::string( cur_delim ) );
	}
	
	return list;
}

void SolTrSystem::read_line(char* buf, int len, FILE* fp) {
	fgets(buf, len, fp);
	int nch = strlen(buf);
	if (nch > 0 && buf[nch-1] == '\n')
		buf[nch-1] = 0;
	if (nch-1 > 0 && buf[nch-2] == '\r')
		buf[nch-2] = 0;
}

bool SolTrSystem::read_sun(FILE* fp) {
	
    if (!fp) return false;

	char buf[1024];
	int bi = 0, count = 0;
	char cshape = 'g';
	double Sigma, HalfWidth;
	bool PointSource;

	read_line( buf, 1023, fp );

	sscanf(buf, "SUN\tPTSRC\t%d\tSHAPE\t%c\tSIGMA\t%lg\tHALFWIDTH\t%lg",
		&bi, &cshape, &Sigma, &HalfWidth);
	PointSource = (bi!=0);
	cshape = tolower(cshape);

    // TODO: Update if supporting other sun shapes
    setSunAngle(Sigma);

	read_line( buf, 1023, fp );
	double X, Y, Z, Latitude, Day, Hour;
	bool UseLDHSpec;
	sscanf(buf, "XYZ\t%lg\t%lg\t%lg\tUSELDH\t%d\tLDH\t%lg\t%lg\t%lg",
		&X, &Y, &Z, &bi, &Latitude, &Day, &Hour);
	UseLDHSpec = (bi!=0);
	
    // TODO: Update if supporting LDHSpec
	// if ( UseLDHSpec )
	// {
	// 	st_sun_position(cxt, Latitude, Day, Hour, &X, &Y, &Z);
	// }

    Vector3d sun_vector(X, Y, Z);
	setSunVector(sun_vector);

	printf("sun ps? %d cs: %c  %lg %lg %lg\n", PointSource?1:0, cshape, X, Y, Z);

	read_line( buf, 1023, fp );
	sscanf(buf, "USER SHAPE DATA\t%d", &count);
    // TODO: Update if supporting user shape data
	// if (count > 0)
	// {
	// 	double *angle = new double[count];
	// 	double *intensity = new double[count];

	// 	for (int i=0;i<count;i++)
	// 	{
	// 		double x, y;
	// 		read_line( buf, 1023, fp );
	// 		sscanf(buf, "%lg\t%lg", &x, &y);
	// 		angle[i] = x;
	// 		intensity[i] = y;
	// 	}

	// 	st_sun_userdata(cxt, count, angle, intensity );

	// 	delete [] angle;
	// 	delete [] intensity;	
	// }

	return true;
}

bool SolTrSystem::read_optic_surface(FILE* fp) {
	
    if (!fp) return false;
	char buf[1024];
	read_line(buf, 1023, fp);
	std::vector<std::string> parts  = split( std::string(buf), "\t", true, false );
	if (parts.size() < 15)
	{
		printf("too few tokens for optical surface: %d\n", parts.size());
		printf("\t>> %s\n", buf);
		return false;
	}

	char ErrorDistribution = 'g';
	if (parts[1].length() > 0)
		ErrorDistribution = parts[1][0];

	int ApertureStopOrGratingType = atoi( parts[2].c_str() );
	int OpticalSurfaceNumber = atoi( parts[3].c_str() );
	int DiffractionOrder = atoi( parts[4].c_str() );
	double Reflectivity = atof( parts[5].c_str() );
	double Transmissivity = atof( parts[6].c_str() );
	double RMSSlope = atof( parts[7].c_str() );
	double RMSSpecularity = atof( parts[8].c_str() );
	double RefractionIndexReal = atof( parts[9].c_str() );
	double RefractionIndexImag = atof( parts[10].c_str() );
	double GratingCoeffs[4];
	GratingCoeffs[0] = atof( parts[11].c_str() );
	GratingCoeffs[1] = atof( parts[12].c_str() );
	GratingCoeffs[2] = atof( parts[13].c_str() );
	GratingCoeffs[3] = atof( parts[14].c_str() );

	bool UseReflectivityTable = false;
	int refl_npoints = 0;
	double *refl_angles = 0;
	double *refls = 0;

	bool UseTransmissivityTable = false;
	int trans_npoints = 0;
	double* trans_angles = 0;
	double* transs = 0;

	if (parts.size() >= 17)
	{
		UseReflectivityTable = (atoi( parts[15].c_str() ) > 0);
		refl_npoints = atoi( parts[16].c_str() );
		if (parts.size() >= 19)
		{
			UseTransmissivityTable = (atoi(parts[17].c_str()) > 0);
			trans_npoints = atoi(parts[18].c_str());
		}
	}

	if (UseReflectivityTable)
	{
		refl_angles = new double[refl_npoints];
		refls = new double[refl_npoints];

		for (int i=0;i<refl_npoints;i++)
		{
			read_line(buf,1023,fp);
			sscanf(buf, "%lg %lg", &refl_angles[i], &refls[i]);
		}
	}
	if (UseTransmissivityTable)
	{
		trans_angles = new double[trans_npoints];
		transs = new double[trans_npoints];

		for (int i = 0; i < trans_npoints; i++)
		{
			read_line(buf, 1023, fp);
			sscanf(buf, "%lg %lg", &trans_angles[i], &transs[i]);
		}
	}

	// TODO: Update once optical surface params are implemented
	// st_optic( cxt, iopt, fb, ErrorDistribution,
	// 	OpticalSurfaceNumber, ApertureStopOrGratingType, DiffractionOrder,
	// 	RefractionIndexReal, RefractionIndexImag,
	// 	Reflectivity, Transmissivity,
	// 	GratingCoeffs, RMSSlope, RMSSpecularity,
	// 	UseReflectivityTable ? 1 : 0, refl_npoints,
	// 	refl_angles, refls,
	// 	UseTransmissivityTable? 1 : 0, trans_npoints,
	// 	trans_angles, transs
	// 	);

	if (refl_angles != 0) delete [] refl_angles;
	if (refls != 0) delete [] refls;
	if (trans_angles != 0) delete[] trans_angles;
	if (transs != 0) delete[] transs;
	return true;
}

bool SolTrSystem::read_optic(FILE* fp) {
	if (!fp) return false;
	char buf[1024];
	read_line( buf, 1023, fp );

	if (strncmp( buf, "OPTICAL PAIR", 12) == 0)
	{
		read_optic_surface( fp );
		read_optic_surface( fp );
		return true;
	}
	else return false;
}

bool SolTrSystem::read_element(FILE* fp) {
	
    //int ielm = ::st_add_element( cxt, istage );

	char buf[1024];
	read_line(buf, 1023, fp);

	std::vector<std::string> tok = split( buf, "\t", true, false );
	if (tok.size() < 29)
	{
		printf("too few tokens for element: %d\n", tok.size());
		printf("\t>> %s\n", buf);
		return false;
	}

	//st_element_enabled( cxt, istage, ielm,  atoi( tok[0].c_str() ) ? 1 : 0 );
    auto elem = std::make_shared<Element>();
    Vector3d origin(atof(tok[1].c_str()),
                    atof(tok[2].c_str()),
                    atof(tok[3].c_str())); // origin of the element
    Vector3d aim_point(atof(tok[4].c_str()),
                       atof(tok[5].c_str()),
                       atof(tok[6].c_str())); // aim point of the element

    elem->set_origin(origin);
    elem->set_aim_point(aim_point);

    // st_element_zrot( cxt, istage, ielm,  atof( tok[7].c_str() ) );
	
    // TODO: Add more aperature and surface types
    if (!tok[8].empty() && tok[8][0] == 'r') {
        double dim_x = atof(tok[9].c_str());
        double dim_y = atof(tok[10].c_str());
        auto aperture = std::make_shared<ApertureRectangle>(dim_x, dim_y);
        elem->set_aperture(aperture);
    }

    if (!tok[17].empty() && tok[17][0] == 'p') {
        double curv_x = atof(tok[18].c_str());
        double curv_y = atof(tok[19].c_str());
        auto surface = std::make_shared<SurfaceParabolic>();
        surface->set_curvature(curv_x, curv_y);
        elem->set_surface(surface);
    } else {
        // Create a flat surface if not parabolic
        auto surface = std::make_shared<SurfaceFlat>();
        elem->set_surface(surface);
    }
	
	// st_element_optic( cxt, istage, ielm,  tok[27].c_str() );
	// st_element_interaction( cxt, istage, ielm,  atoi( tok[28].c_str()) );

    AddElement(elem); // Add the element to the system

	return true;
}

bool SolTrSystem::read_stage(FILE* fp) {
	
    if (!fp) return false;

	char buf[1024];
	read_line( buf, 1023, fp );

	int virt=0,multi=1,count=0,tr=0;
	double X, Y, Z, AX, AY, AZ, ZRot;


	sscanf(buf, "STAGE\tXYZ\t%lg\t%lg\t%lg\tAIM\t%lg\t%lg\t%lg\tZROT\t%lg\tVIRTUAL\t%d\tMULTIHIT\t%d\tELEMENTS\t%d\tTRACETHROUGH\t%d",
		&X, &Y, &Z,
		&AX, &AY, &AZ,
		&ZRot,
		&virt,
		&multi,
		&count,
		&tr );

	read_line( buf, 1023, fp ); // read name

	printf("stage '%s': [%d] %lg %lg %lg   %lg %lg %lg   %lg   %d %d %d\n",
		buf, count, X, Y, Z, AX, AY, AZ, ZRot, virt, multi, tr );
	
	for (int i=0;i<count;i++)
		if (!read_element( fp )) 
		{ printf("error in element %d\n", i ); return false; }

	return true;
}

bool SolTrSystem::read_system(FILE* fp) {
	
    if (!fp) return false;

	char buf[1024];

	char c = fgetc(fp);
	if ( c == '#' )
	{
		int vmaj = 0, vmin = 0, vmic = 0;
		read_line( buf, 1023, fp ); sscanf( buf, " SOLTRACE VERSION %d.%d.%d INPUT FILE", &vmaj, &vmin, &vmic);

		unsigned int file_version = vmaj*10000 + vmin*100 + vmic;
		
		printf( "loading input file version %d.%d.%d\n", vmaj, vmin, vmic );
	}
	else
	{
		ungetc( c, fp );
		printf("input file must start with '#'\n");
		return false;
	}

	if ( !read_sun( fp ) ) return false;
	
	int count = 0;

	count = 0;
	read_line( buf, 1023, fp ); sscanf(buf, "OPTICS LIST COUNT\t%d", &count);
	
	for (int i=0;i<count;i++)
		if (!read_optic( fp )) return false;

	count = 0;
	read_line( buf, 1023, fp ); sscanf(buf, "STAGE LIST COUNT\t%d", &count);
	for (int i=0;i<count;i++)
		if (!read_stage( fp )) return false;

	return true;
}