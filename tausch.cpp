#include "tausch.h"

Tausch::Tausch(int localDimX, int localDimY, int mpiNumX, int mpiNumY, bool blockingSyncCpuGpu, bool withOpenCL, bool setupOpenCL, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {

    // get MPI info
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    this->mpiNumX = mpiNumX;
    this->mpiNumY = mpiNumY;

    // store configuration
    this->localDimX = localDimX;
    this->localDimY = localDimY;
    this->blockingSyncCpuGpu = blockingSyncCpuGpu;

    // check if this rank has a boundary with another rank
    haveBoundary[Left] = (mpiRank%mpiNumX != 0);
    haveBoundary[Right] = ((mpiRank+1)%mpiNumX != 0);
    haveBoundary[Top] = (mpiRank < mpiSize-mpiNumX);
    haveBoundary[Bottom] = (mpiRank > mpiNumX-1);

    // a send and recv buffer for the CPU-CPU communication
    cpuToCpuSendBuffer = new double*[4];
    cpuToCpuSendBuffer[Left] = new double[localDimY+2]{};
    cpuToCpuSendBuffer[Right] = new double[localDimY+2]{};
    cpuToCpuSendBuffer[Top] = new double[localDimX+2]{};
    cpuToCpuSendBuffer[Bottom] = new double[localDimX+2]{};
    cpuToCpuRecvBuffer = new double*[4];
    cpuToCpuRecvBuffer[Left] = new double[localDimY+2]{};
    cpuToCpuRecvBuffer[Right] = new double[localDimY+2]{};
    cpuToCpuRecvBuffer[Top] = new double[localDimX+2]{};
    cpuToCpuRecvBuffer[Bottom] = new double[localDimX+2]{};

    // gpu disabled by default, only enabled if flag is set
    gpuEnabled = false;
    if(withOpenCL) {
        // currently fixed here, local workgroup size
        cl_kernelLocalSize = clLocalWorkgroupSize;
        // Tausch can either set up OpenCL itself, or if not it needs to be passed some OpenCL variables by the user
        if(setupOpenCL)
            this->setupOpenCL(giveOpenCLDeviceName);
    } else {
        cpuToGpuBuffer = new double[1];
        gpuToCpuBuffer = new double[1];
    }

    // whether the cpu/gpu pointers have been passed
    gpuInfoGiven = false;
    cpuInfoGiven = false;

    // cpu at beginning
    cpuRecvsPosted = false;
    gpuToCpuStarted = false;
    cpuToGpuStarted = false;

    // communication to neither edge has been started
    cpuStarted[Left] = false;
    cpuStarted[Right] = false;
    cpuStarted[Top] = false;
    cpuStarted[Bottom] = false;

    // used for syncing the CPU and GPU thread
    syncpointCpu = 0;
    syncpointGpu = 0;

}

Tausch::~Tausch() {
    // clean up memory
    for(int i = 0; i < 4; ++i) {
        delete[] cpuToCpuSendBuffer[i];
        delete[] cpuToCpuRecvBuffer[i];
    }
    delete[] cpuToCpuSendBuffer;
    delete[] cpuToCpuRecvBuffer;
    delete[] cpuToGpuBuffer;
    delete[] gpuToCpuBuffer;
}

// get a pointer to the CPU data
void Tausch::setCPUData(double *dat) {
    cpuInfoGiven = true;
    cpuData = dat;
}

// get a pointer to the GPU buffer and its dimensions
void Tausch::setGPUData(cl::Buffer &dat, int gpuWidth, int gpuHeight) {

    // check whether OpenCL has been set up
    if(!gpuEnabled) {
        std::cerr << "ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    gpuInfoGiven = true;

    // store parameters
    gpuData = dat;
    this->gpuWidth = gpuWidth;
    this->gpuHeight = gpuHeight;

    // store buffer to store the GPU and the CPU part of the halo.
    // We do not need two buffers each, as each thread has direct access to both arrays, no communication necessary
    cpuToGpuBuffer = new double[2*(gpuWidth+2) + 2*gpuHeight]{};
    gpuToCpuBuffer = new double[2*gpuWidth + 2*gpuHeight]{};

    // set up buffers on device
    try {
        cl_gpuToCpuBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (2*(gpuWidth+2)+2*gpuHeight)*sizeof(double));
        cl_queue.enqueueFillBuffer(cl_gpuToCpuBuffer, 0, 0, (2*(gpuWidth+2) + 2*gpuHeight)*sizeof(double));
        cl_gpuWidth = cl::Buffer(cl_context, &gpuWidth, (&gpuWidth)+1, true);
        cl_gpuHeight = cl::Buffer(cl_context, &gpuHeight, (&gpuHeight)+1, true);
    } catch(cl::Error error) {
        std::cout << "[setup send/recv buffer] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

// post the MPI_Irecv's for inter-rank communication
void Tausch::postCpuReceives() {

    if(!cpuInfoGiven) {
        std::cerr << "ERROR: You didn't tell me yet where to find the data! Abort..." << std::endl;
        exit(1);
    }

    cpuRecvsPosted = true;

    if(haveBoundary[Left])
        MPI_Irecv(&cpuToCpuRecvBuffer[Left][0], localDimY+2, MPI_DOUBLE, mpiRank-1, 0, MPI_COMM_WORLD, &cpuToCpuRecvRequest[Left]);
    if(haveBoundary[Right])
        MPI_Irecv(&cpuToCpuRecvBuffer[Right][0], localDimY+2, MPI_DOUBLE, mpiRank+1, 2, MPI_COMM_WORLD, &cpuToCpuRecvRequest[Right]);
    if(haveBoundary[Top])
        MPI_Irecv(&cpuToCpuRecvBuffer[Top][0], localDimX+2, MPI_DOUBLE, mpiRank+mpiNumX, 1, MPI_COMM_WORLD, &cpuToCpuRecvRequest[Top]);
    if(haveBoundary[Bottom])
        MPI_Irecv(&cpuToCpuRecvBuffer[Bottom][0], localDimX+2, MPI_DOUBLE, mpiRank-mpiNumX, 3, MPI_COMM_WORLD, &cpuToCpuRecvRequest[Bottom]);

}

void Tausch::startCpuTauschEdge(Edge edge) {

    if(!cpuRecvsPosted) {
        std::cerr << "ERROR: No CPU Recvs have been posted yet... Abort!" << std::endl;
        exit(1);
    }

    if(edge != Left && edge != Right && edge != Top && edge != Bottom) {
        std::cerr << "startCpuTauschEdge(): ERROR: Invalid edge specified: " << edge << std::endl;
        exit(1);
    }

    cpuStarted[edge] = true;

    if(edge == Left && haveBoundary[Left]) {
        for(int i = 0; i < localDimY+2; ++i)
            cpuToCpuSendBuffer[Left][i] = cpuData[1+ i*(localDimX+2)];
        MPI_Isend(&cpuToCpuSendBuffer[Left][0], localDimY+2, MPI_DOUBLE, mpiRank-1, 2, MPI_COMM_WORLD, &cpuToCpuSendRequest[Left]);
    } else if(edge == Right && haveBoundary[Right]) {
        for(int i = 0; i < localDimY+2; ++i)
            cpuToCpuSendBuffer[Right][i] = cpuData[(i+1)*(localDimX+2) -2];
        MPI_Isend(&cpuToCpuSendBuffer[Right][0], localDimY+2, MPI_DOUBLE, mpiRank+1, 0, MPI_COMM_WORLD, &cpuToCpuSendRequest[Right]);
    } else if(edge == Top && haveBoundary[Top]) {
        for(int i = 0; i < localDimX+2; ++i)
            cpuToCpuSendBuffer[Top][i] = cpuData[(localDimX+2)*localDimY + i];
        MPI_Isend(&cpuToCpuSendBuffer[Top][0], localDimX+2, MPI_DOUBLE, mpiRank+mpiNumX, 3, MPI_COMM_WORLD, &cpuToCpuSendRequest[Top]);
    } else if(edge == Bottom && haveBoundary[Bottom]) {
        for(int i = 0; i < localDimX+2; ++i)
            cpuToCpuSendBuffer[Bottom][i] = cpuData[localDimX+2 + i];
        MPI_Isend(&cpuToCpuSendBuffer[Bottom][0], localDimX+2, MPI_DOUBLE, mpiRank-mpiNumX, 1, MPI_COMM_WORLD, &cpuToCpuSendRequest[Bottom]);
    }

}

// collect cpu side of cpu/gpu halo and store in buffer
void Tausch::startCpuToGpuTausch() {

    // check whether GPU is enabled
    if(!gpuEnabled) {
        std::cerr << "ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    cpuToGpuStarted = true;

    // left
    for(int i = 0; i < gpuHeight; ++ i)
        cpuToGpuBuffer[i] = cpuData[((localDimY-gpuHeight)/2 +i+1)*(localDimX+2) + (localDimX-gpuWidth)/2];
    // right
    for(int i = 0; i < gpuHeight; ++ i)
        cpuToGpuBuffer[gpuHeight + i] = cpuData[((localDimY-gpuHeight)/2 +i+1)*(localDimX+2) + (localDimX-gpuWidth)/2 + gpuWidth+1];
    // top
    for(int i = 0; i < gpuWidth+2; ++ i)
        cpuToGpuBuffer[2*gpuHeight + i] = cpuData[((localDimY-gpuHeight)/2 +gpuHeight+1)*(localDimX+2) + (localDimX-gpuWidth)/2 + i];
    // bottom
    for(int i = 0; i < gpuWidth+2; ++ i)
        cpuToGpuBuffer[2*gpuHeight+gpuWidth+2 + i] = cpuData[((localDimY-gpuHeight)/2)*(localDimX+2) + (localDimX-gpuWidth)/2 + i];

}

// collect gpu side of cpu/gpu halo and download into buffer
void Tausch::startGpuToCpuTausch() {

    // check whether GPU is enabled
    if(!gpuEnabled) {
        std::cerr << "ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }
    // check whether GPU info was given
    if(!gpuInfoGiven) {
        std::cerr << "ERROR: GPU info not available! Did you call setOpenCLInfo()? Abort..." << std::endl;
        exit(1);
    }

    gpuToCpuStarted = true;

    try {

        auto kernel_collectHalo = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(cl_programs, "collectHaloData");

        int globalSize = ((2*gpuWidth+2*gpuHeight)/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_collectHalo(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                           cl_gpuWidth, cl_gpuHeight, gpuData, cl_gpuToCpuBuffer);

        cl::copy(cl_queue, cl_gpuToCpuBuffer, &gpuToCpuBuffer[0], (&gpuToCpuBuffer[2*gpuWidth+2*gpuHeight-1])+1);

    } catch(cl::Error error) {
        std::cout << "[kernel collectHalo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

// Complete CPU-CPU exchange to the left
void Tausch::completeCpuTauschEdge(Edge edge) {

    if(edge != Left && edge != Right && edge != Top && edge != Bottom) {
        std::cerr << "completeCpuTauschEdge(): ERROR: Invalid edge specified: " << edge << std::endl;
        exit(1);
    }

    if(!cpuStarted[edge]) {
        std::cerr << "ERROR: No edge #" << edge << " CPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    if(edge == Left && haveBoundary[Left]) {
        MPI_Wait(&cpuToCpuRecvRequest[Left], MPI_STATUS_IGNORE);
        for(int i = 0; i < localDimY+2; ++i)
            cpuData[i*(localDimX+2)] = cpuToCpuRecvBuffer[Left][i];
        MPI_Wait(&cpuToCpuSendRequest[Left], MPI_STATUS_IGNORE);
    } else if(edge == Right && haveBoundary[Right]) {
        MPI_Wait(&cpuToCpuRecvRequest[Right], MPI_STATUS_IGNORE);
        for(int i = 0; i < localDimY+2; ++i)
            cpuData[(i+1)*(localDimX+2)-1] = cpuToCpuRecvBuffer[Right][i];
        MPI_Wait(&cpuToCpuSendRequest[Right], MPI_STATUS_IGNORE);
    } else if(edge == Top && haveBoundary[Top]) {
        MPI_Wait(&cpuToCpuRecvRequest[Top], MPI_STATUS_IGNORE);
        for(int i = 0; i < localDimX+2; ++i)
            cpuData[(localDimX+2)*(localDimY+1) + i] = cpuToCpuRecvBuffer[Top][i];
        MPI_Wait(&cpuToCpuSendRequest[Top], MPI_STATUS_IGNORE);
    } else if(edge == Bottom && haveBoundary[Bottom]) {
        MPI_Wait(&cpuToCpuRecvRequest[Bottom], MPI_STATUS_IGNORE);
        for(int i = 0; i < localDimX+2; ++i)
            cpuData[i] = cpuToCpuRecvBuffer[Bottom][i];
        MPI_Wait(&cpuToCpuSendRequest[Bottom], MPI_STATUS_IGNORE);
    }

}

// Complete CPU side of CPU/GPU halo exchange
void Tausch::completeCpuToGpuTausch() {

    if(!cpuToGpuStarted) {
        std::cerr << "ERROR: No CPU->GPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    // we need to wait for the GPU thread to arrive here
    syncCpuAndGpu(true);

    // left
    for(int i = 0; i < gpuHeight; ++ i)
        cpuData[((localDimY-gpuHeight)/2 +i+1)*(localDimX+2) + (localDimX-gpuWidth)/2 +1] = gpuToCpuBuffer[i];
    // right
    for(int i = 0; i < gpuHeight; ++ i)
        cpuData[((localDimY-gpuHeight)/2 +i+1)*(localDimX+2) + (localDimX-gpuWidth)/2 + gpuWidth] = gpuToCpuBuffer[gpuHeight + i];
    // top
    for(int i = 0; i < gpuWidth; ++ i)
        cpuData[((localDimY-gpuHeight)/2 +gpuHeight)*(localDimX+2) + (localDimX-gpuWidth)/2 + i+1] = gpuToCpuBuffer[2*gpuHeight + i];
    // bottom
    for(int i = 0; i < gpuWidth; ++ i)
        cpuData[((localDimY-gpuHeight)/2 +1)*(localDimX+2) + (localDimX-gpuWidth)/2 + i+1] = gpuToCpuBuffer[2*gpuHeight+gpuWidth + i];

}

// Complete GPU side of CPU/GPU halo exchange
void Tausch::completeGpuToCpuTausch() {

    if(!gpuToCpuStarted) {
        std::cerr << "ERROR: No GPU->CPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    // we need to wait for the CPU thread to arrive here
    syncCpuAndGpu(false);

    try {

        cl::copy(cl_queue, &cpuToGpuBuffer[0], (&cpuToGpuBuffer[2*(gpuWidth+2)+2*gpuHeight-1])+1, cl_gpuToCpuBuffer);

        auto kernel_distributeHaloData = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(cl_programs, "distributeHaloData");

        int globalSize = ((2*(gpuWidth+2) + 2*gpuHeight)/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_distributeHaloData(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                                  cl_gpuWidth, cl_gpuHeight, gpuData, cl_gpuToCpuBuffer);

    } catch(cl::Error error) {
        std::cout << "[dist halo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }


}

// both the CPU and GPU have to arrive at this point before either can continue
void Tausch::syncCpuAndGpu(bool iAmTheCPU) {

    if(!blockingSyncCpuGpu) return;

    if(iAmTheCPU) {
        syncpointCpu = 31;
        bool wait = true;
        while(wait) {
            if(syncpointGpu == 31) {
                wait = false;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // the cpu resets the gpu sync variable, to avoid deadlock
        syncpointGpu = 0;
    } else {
        syncpointGpu = 31;
        bool wait = true;
        while(wait) {
            if(syncpointCpu == 31) {
                wait = false;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // the gpu resets the cpu sync variable, to avoid deadlock
        syncpointCpu = 0;
    }

    // this wait ensures that the other thread has indeed caught the variable, avoids potential deadlock for when the other thread could reach the next sync too fast
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

// If Tausch didn't set up OpenCL, the user needs to pass some OpenCL variables
void Tausch::setOpenCLInfo(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue) {

    this->cl_defaultDevice = cl_defaultDevice;
    this->cl_context = cl_context;
    this->cl_queue = cl_queue;

    gpuEnabled = true;

    compileKernels();

}

void Tausch::compileKernels() {

    // Tausch requires two kernels: One for collecting the halo data and one for distributing that data
    std::string oclstr = R"d(
    kernel void collectHaloData(global const int * restrict const dimX, global const int * restrict const dimY,
                            global const double * restrict const vec, global double * sync) {
    unsigned int current = get_global_id(0);
    unsigned int maxNum = 2*(*dimX) + 2*(*dimY);
    if(current >= maxNum)
        return;
    // left
    if(current < *dimY) {
        sync[current] = vec[(1+current)*(*dimX+2) +1];
        return;
    }
    // right
    if(current < 2*(*dimY)) {
        sync[current] = vec[(2+(current-(*dimY)))*(*dimX+2) -2];
        return;
    }
    // top
    if(current < 2*(*dimY) + *dimX) {
        sync[current] = vec[(*dimX+2)*(*dimY)+1 + current-(2*(*dimY))];
        return;
    }
    // bottom
    sync[current] = vec[1+(*dimX+2)+(current-2*(*dimY)-(*dimX))];
    }
    kernel void distributeHaloData(global const int * restrict const dimX, global const int * restrict const dimY,
                               global double * vec, global const double * restrict const sync) {
    unsigned int current = get_global_id(0);
    unsigned int maxNum = 2*(*dimX+2) + 2*(*dimY);
    if(current >= maxNum)
        return;
    // left
    if(current < *dimY) {
        vec[(1+current)*(*dimX+2)] = sync[current];
        return;
    }
    // right
    if(current < 2*(*dimY)) {
        vec[(2+(current-(*dimY)))*(*dimX+2) -1] = sync[current];
        return;
    }
    // top
    if(current < 2*(*dimY)+(*dimX+2)) {
        vec[(*dimX+2)*(*dimY+1) + current-(2*(*dimY))] = sync[current];
        return;
    }
    // bottom
    vec[current-2*(*dimY)-(*dimX+2)] = sync[current];
    }
                         )d";

    try {
        cl_programs = cl::Program(cl_context, oclstr, false);
        cl_programs.build("");
    } catch(cl::Error error) {
        std::cout << "[kernel compile] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        if(error.err() == -11) {
            std::string log = cl_programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_defaultDevice);
            std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** " << std::endl << std::endl;
        }
    }

}

// Create OpenCL context and choose a device (if multiple devices are available, the MPI ranks will split up evenly)
void Tausch::setupOpenCL(bool giveOpenCLDeviceName) {

    gpuEnabled = true;

    try {

        // Get platform count
        std::vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        int platform_length = all_platforms.size();

        // We need at most mpiSize many devices
        int *platform_num = new int[mpiSize]{};
        int *device_num = new int[mpiSize]{};

        // Counter so that we know when to stop
        int num = 0;

        // Loop over platforms
        for(int i = 0; i < platform_length; ++i) {
            // Get devices on platform
            std::vector<cl::Device> all_devices;
            all_platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
            int device_length = all_devices.size();
            // Loop over platforms
            for(int j = 0; j < device_length; ++j) {
                // Store current pair
                platform_num[num] = i;
                device_num[num] = j;
                ++num;
                // and stop
                if(num == mpiSize) {
                    i = platform_length;
                    break;
                }
            }
        }

        // Get the platform and device to be used by this MPI thread
        cl_platform = all_platforms[platform_num[mpiRank%num]];
        std::vector<cl::Device> all_devices;
        cl_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        cl_defaultDevice = all_devices[device_num[mpiRank%num]];

        // Give some feedback of the choice.
        if(giveOpenCLDeviceName) {
            std::stringstream ss;
            ss << "Rank " << mpiRank << " using OpenCL platform #" << platform_num[mpiRank%num] << " with device #" << device_num[mpiRank%num] << ": " << cl_defaultDevice.getInfo<CL_DEVICE_NAME>() << std::endl;
            EveryoneOutput(ss.str());
            if(mpiRank == 0)
                std::cout << std::endl;
        }

        delete[] platform_num;
        delete[] device_num;

        // Create context and queue
        cl_context = cl::Context({cl_defaultDevice});
        cl_queue = cl::CommandQueue(cl_context,cl_defaultDevice);

    } catch(cl::Error error) {
        std::cout << "[setup] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    // And compile kernels
    compileKernels();

}

// Let every MPI rank one by one output the stuff
void Tausch::EveryoneOutput(const std::string &inMessage) {

    for(int iRank = 0; iRank < mpiSize; ++iRank){
        if(mpiRank == iRank)
            std::cout << inMessage;
        MPI_Barrier(MPI_COMM_WORLD);
    }

}

void Tausch::checkOpenCLError(cl_int clErr, std::string loc) {

    if(clErr == CL_SUCCESS) return;

    int err = clErr;

    std::string errstr = "";

    // run-time and JIT compiler errors
    if(err == 0) errstr = "CL_SUCCESS";
    else if(err == -1) errstr = "CL_DEVICE_NOT_FOUND";
    else if(err == -2) errstr = "CL_DEVICE_NOT_AVAILABLE";
    else if(err == -3) errstr = "CL_COMPILER_NOT_AVAILABLE";
    else if(err == -4) errstr = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    else if(err == -5) errstr = "CL_OUT_OF_RESOURCES";
    else if(err == -6) errstr = "CL_OUT_OF_HOST_MEMORY";
    else if(err == -7) errstr = "CL_PROFILING_INFO_NOT_AVAILABLE";
    else if(err == -8) errstr = "CL_MEM_COPY_OVERLAP";
    else if(err == -9) errstr = "CL_IMAGE_FORMAT_MISMATCH";
    else if(err == -10) errstr = "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    else if(err == -11) errstr = "CL_BUILD_PROGRAM_FAILURE";
    else if(err == -12) errstr = "CL_MAP_FAILURE";
    else if(err == -13) errstr = "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    else if(err == -14) errstr = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    else if(err == -15) errstr = "CL_COMPILE_PROGRAM_FAILURE";
    else if(err == -16) errstr = "CL_LINKER_NOT_AVAILABLE";
    else if(err == -17) errstr = "CL_LINK_PROGRAM_FAILURE";
    else if(err == -18) errstr = "CL_DEVICE_PARTITION_FAILED";
    else if(err == -19) errstr = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    else if(err == -30) errstr = "CL_INVALID_VALUE";
    else if(err == -31) errstr = "CL_INVALID_DEVICE_TYPE";
    else if(err == -32) errstr = "CL_INVALID_PLATFORM";
    else if(err == -33) errstr = "CL_INVALID_DEVICE";
    else if(err == -34) errstr = "CL_INVALID_CONTEXT";
    else if(err == -35) errstr = "CL_INVALID_QUEUE_PROPERTIES";
    else if(err == -36) errstr = "CL_INVALID_COMMAND_QUEUE";
    else if(err == -37) errstr = "CL_INVALID_HOST_PTR";
    else if(err == -38) errstr = "CL_INVALID_MEM_OBJECT";
    else if(err == -39) errstr = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    else if(err == -40) errstr = "CL_INVALID_IMAGE_SIZE";
    else if(err == -41) errstr = "CL_INVALID_SAMPLER";
    else if(err == -42) errstr = "CL_INVALID_BINARY";
    else if(err == -43) errstr = "CL_INVALID_BUILD_OPTIONS";
    else if(err == -44) errstr = "CL_INVALID_PROGRAM";
    else if(err == -45) errstr = "CL_INVALID_PROGRAM_EXECUTABLE";
    else if(err == -46) errstr = "CL_INVALID_KERNEL_NAME";
    else if(err == -47) errstr = "CL_INVALID_KERNEL_DEFINITION";
    else if(err == -48) errstr = "CL_INVALID_KERNEL";
    else if(err == -49) errstr = "CL_INVALID_ARG_INDEX";
    else if(err == -50) errstr = "CL_INVALID_ARG_VALUE";
    else if(err == -51) errstr = "CL_INVALID_ARG_SIZE";
    else if(err == -52) errstr = "CL_INVALID_KERNEL_ARGS";
    else if(err == -53) errstr = "CL_INVALID_WORK_DIMENSION";
    else if(err == -54) errstr = "CL_INVALID_WORK_GROUP_SIZE";
    else if(err == -55) errstr = "CL_INVALID_WORK_ITEM_SIZE";
    else if(err == -56) errstr = "CL_INVALID_GLOBAL_OFFSET";
    else if(err == -57) errstr = "CL_INVALID_EVENT_WAIT_LIST";
    else if(err == -58) errstr = "CL_INVALID_EVENT";
    else if(err == -59) errstr = "CL_INVALID_OPERATION";
    else if(err == -60) errstr = "CL_INVALID_GL_OBJECT";
    else if(err == -61) errstr = "CL_INVALID_BUFFER_SIZE";
    else if(err == -62) errstr = "CL_INVALID_MIP_LEVEL";
    else if(err == -63) errstr = "CL_INVALID_GLOBAL_WORK_SIZE";
    else if(err == -64) errstr = "CL_INVALID_PROPERTY";
    else if(err == -65) errstr = "CL_INVALID_IMAGE_DESCRIPTOR";
    else if(err == -66) errstr = "CL_INVALID_COMPILER_OPTIONS";
    else if(err == -67) errstr = "CL_INVALID_LINKER_OPTIONS";
    else if(err == -68) errstr = "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    else if(err == -1000) errstr = "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    else if(err == -1001) errstr = "CL_PLATFORM_NOT_FOUND_KHR";
    else if(err == -1002) errstr = "CL_INVALID_D3D10_DEVICE_KHR";
    else if(err == -1003) errstr = "CL_INVALID_D3D10_RESOURCE_KHR";
    else if(err == -1004) errstr = "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    else if(err == -1005) errstr = "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    else errstr = "Unknown OpenCL error";

    std::cout << "[" << loc << "] OpenCL exception caught: " << errstr << " (" << err << ")" << std::endl;

    if(err == CL_BUILD_PROGRAM_FAILURE) {
        std::string log = cl_programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_defaultDevice);
        std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** " << std::endl << std::endl;
    }

    exit(1);
}
