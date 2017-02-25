#include "tausch.h"

Tausch::Tausch(int localDimX, int localDimY, int mpiNumX, int mpiNumY, bool withOpenCL, bool setupOpenCL) {

    // get MPI info
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    this->mpiNumX = mpiNumX;
    this->mpiNumY = mpiNumY;

    // store configuration
    this->localDimX = localDimX;
    this->localDimY = localDimY;

    haveLeftBorder = (mpiRank%mpiNumX == 0);
    haveRightBorder = ((mpiRank+1)%mpiNumX == 0);
    haveTopBorder = (mpiRank > mpiSize-mpiNumX-1);
    haveBottomBorder = (mpiRank < mpiNumX);

    haveLeftBoundary = !haveLeftBorder;
    haveRightBoundary = !haveRightBorder;
    haveTopBoundary = !haveTopBorder;
    haveBottomBoundary = !haveBottomBorder;

    cpuToCpuSendBuffer = new double[2*(localDimX+2)+2*(localDimY+2)]{};
    cpuToCpuRecvBuffer = new double[2*(localDimX+2)+2*(localDimY+2)]{};

    gpuEnabled = false;
    if(withOpenCL) {
        cl_kernelLocalSize = 64;
        if(setupOpenCL)
            this->setupOpenCL();
    }

    gpuInfoGiven = false;
    cpuInfoGiven = false;

    cpuRecvsPosted = false;
    gpuRecvsPosted = false;
    gpuStarted = false;

    cpuLeftStarted = false;
    cpuRightStarted = false;
    cpuTopStarted = false;
    cpuBottomStarted = false;
    cpuToGpuLeftStarted = false;
    cpuToGpuRightStarted = false;
    cpuToGpuTopStarted = false;
    cpuToGpuBottomStarted = false;

    syncpointCpu = 0;
    syncpointGpu = 0;

}

Tausch::~Tausch() {
        delete[] cpuToCpuSendBuffer;
        delete[] cpuToCpuRecvBuffer;
        delete[] cpuToGpuBuffer;
        delete[] gpuToCpuBuffer;
}

void Tausch::setCPUData(double *dat) {
    cpuInfoGiven = true;
    cpuData = dat;
}
void Tausch::setGPUData(cl::Buffer &dat, int gpuWidth, int gpuHeight) {

    if(!gpuEnabled) {
        std::cerr << "ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    gpuInfoGiven = true;
    gpuData = dat;
    this->gpuWidth = gpuWidth;
    this->gpuHeight = gpuHeight;

    cpuToGpuBuffer = new double[2*(gpuWidth+2) + 2*gpuHeight]{};
    gpuToCpuBuffer = new double[2*gpuWidth + 2*gpuHeight]{};

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

void Tausch::postCpuReceives() {

    if(!cpuInfoGiven) {
        std::cerr << "ERROR: You didn't tell me yet where to find the data! Abort..." << std::endl;
        exit(1);
    }

    cpuRecvsPosted = true;

    if(haveLeftBoundary)
        MPI_Irecv(&cpuToCpuRecvBuffer[0], localDimY+2, MPI_DOUBLE, mpiRank-1, 0, MPI_COMM_WORLD, &cpuToCpuLeftRecvRequest);

    if(haveRightBoundary)
        MPI_Irecv(&cpuToCpuRecvBuffer[localDimY+2], localDimY+2, MPI_DOUBLE, mpiRank+1, 2, MPI_COMM_WORLD, &cpuToCpuRightRecvRequest);

    if(haveTopBoundary)
        MPI_Irecv(&cpuToCpuRecvBuffer[2*(localDimY+2)], localDimX+2, MPI_DOUBLE, mpiRank+mpiNumX, 1, MPI_COMM_WORLD, &cpuToCpuTopRecvRequest);

    if(haveBottomBoundary)
        MPI_Irecv(&cpuToCpuRecvBuffer[2*(localDimY+2)+(localDimX+2)], localDimX+2, MPI_DOUBLE, mpiRank-mpiNumX, 3, MPI_COMM_WORLD, &cpuToCpuBottomRecvRequest);


}

void Tausch::startCpuTauschLeft() {

    if(!cpuRecvsPosted) {
        std::cerr << "ERROR: No CPU Recvs have been posted yet... Abort!" << std::endl;
        exit(1);
    }

    cpuLeftStarted = true;

    // Left
    if(haveLeftBoundary) {
        for(int i = 0; i < localDimY+2; ++i)
            cpuToCpuSendBuffer[i] = cpuData[1+ i*(localDimX+2)];
        MPI_Isend(&cpuToCpuSendBuffer[0], localDimY+2, MPI_DOUBLE, mpiRank-1, 2, MPI_COMM_WORLD, &cpuToCpuLeftSendRequest);
    }

}

void Tausch::startCpuTauschRight() {

    if(!cpuRecvsPosted) {
        std::cerr << "ERROR: No CPU Recvs have been posted yet... Abort!" << std::endl;
        exit(1);
    }

    cpuRightStarted = true;

    // Right
    if(haveRightBoundary) {
        for(int i = 0; i < localDimY+2; ++i)
            cpuToCpuSendBuffer[localDimY+2 + i] = cpuData[(i+1)*(localDimX+2) -2];
        MPI_Isend(&cpuToCpuSendBuffer[localDimY+2], localDimY+2, MPI_DOUBLE, mpiRank+1, 0, MPI_COMM_WORLD, &cpuToCpuRightSendRequest);
    }

}

void Tausch::startCpuTauschTop() {

    if(!cpuRecvsPosted) {
        std::cerr << "ERROR: No CPU Recvs have been posted yet... Abort!" << std::endl;
        exit(1);
    }

    cpuTopStarted = true;

    // Top
    if(haveTopBoundary) {
        for(int i = 0; i < localDimX+2; ++i)
            cpuToCpuSendBuffer[2*(localDimY+2) + i] = cpuData[(localDimX+2)*localDimY + i];
        MPI_Isend(&cpuToCpuSendBuffer[2*(localDimY+2)], localDimX+2, MPI_DOUBLE, mpiRank+mpiNumX, 3, MPI_COMM_WORLD, &cpuToCpuTopSendRequest);
    }

}

void Tausch::startCpuTauschBottom() {

    if(!cpuRecvsPosted) {
        std::cerr << "ERROR: No CPU Recvs have been posted yet... Abort!" << std::endl;
        exit(1);
    }

    cpuBottomStarted = true;

    // Bottom
    if(haveBottomBoundary) {
        for(int i = 0; i < localDimX+2; ++i)
            cpuToCpuSendBuffer[2*(localDimY+2)+(localDimX+2) + i] = cpuData[localDimX+2 + i];
        MPI_Isend(&cpuToCpuSendBuffer[2*(localDimY+2)+(localDimX+2)], localDimX+2, MPI_DOUBLE, mpiRank-mpiNumX, 1, MPI_COMM_WORLD, &cpuToCpuBottomSendRequest);
    }

}

void Tausch::startCpuToGpuTausch() {

    gpuStarted = true;

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

void Tausch::startGpuToCpuTausch() {

    gpuStarted = true;

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

void Tausch::completeCpuTauschLeft() {

    if(!cpuLeftStarted) {
        std::cerr << "ERROR: No left CPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    if(haveLeftBoundary) {
        MPI_Wait(&cpuToCpuLeftRecvRequest, MPI_STATUS_IGNORE);
        for(int i = 0; i < localDimY+2; ++i)
            cpuData[i*(localDimX+2)] = cpuToCpuRecvBuffer[i];
        MPI_Wait(&cpuToCpuLeftSendRequest, MPI_STATUS_IGNORE);
    }


}

void Tausch::completeCpuTauschRight() {

    if(!cpuRightStarted) {
        std::cerr << "ERROR: No right CPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    if(haveRightBoundary) {
        MPI_Wait(&cpuToCpuRightRecvRequest, MPI_STATUS_IGNORE);
        for(int i = 0; i < localDimY+2; ++i)
            cpuData[(i+1)*(localDimX+2)-1] = cpuToCpuRecvBuffer[localDimY+2+i];
        MPI_Wait(&cpuToCpuRightSendRequest, MPI_STATUS_IGNORE);
    }


}

void Tausch::completeCpuTauschTop() {

    if(!cpuTopStarted) {
        std::cerr << "ERROR: No top CPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    if(haveTopBoundary) {
        MPI_Wait(&cpuToCpuTopRecvRequest, MPI_STATUS_IGNORE);
        for(int i = 0; i < localDimX+2; ++i)
            cpuData[(localDimX+2)*(localDimY+1) + i] = cpuToCpuRecvBuffer[2*(localDimY+2)+i];
        MPI_Wait(&cpuToCpuTopSendRequest, MPI_STATUS_IGNORE);
    }


}

void Tausch::completeCpuTauschBottom() {

    if(!cpuBottomStarted) {
        std::cerr << "ERROR: No bottom CPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    if(haveBottomBoundary) {
        MPI_Wait(&cpuToCpuBottomRecvRequest, MPI_STATUS_IGNORE);
        for(int i = 0; i < localDimX+2; ++i)
            cpuData[i] = cpuToCpuRecvBuffer[2*(localDimY+2)+(localDimX+2)+i];
        MPI_Wait(&cpuToCpuBottomSendRequest, MPI_STATUS_IGNORE);
    }


}

void Tausch::completeCpuToGpuTausch() {

    if(!gpuStarted) {
        std::cerr << "ERROR: No GPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

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

void Tausch::completeGpuToCpuTausch() {

    if(!gpuStarted) {
        std::cerr << "ERROR: No GPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

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

void Tausch::syncCpuWaitsForGpu(bool iAmTheCPU) {

    if(iAmTheCPU) {
        bool wait = true;
        while(wait) {
            if(syncpointGpu == 11) {
                wait = false;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } else {
        syncpointGpu = 11;
    }

    syncpointCpu = 0;
    syncpointGpu = 0;

}

void Tausch::syncGpuWaitsForCpu(bool iAmTheCPU) {

    if(iAmTheCPU) {
        syncpointCpu = 21;
    } else {
        bool wait = true;
        while(wait) {
            if(syncpointCpu == 21) {
                wait = false;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        syncpointGpu = 21;
    }

    syncpointCpu = 0;
    syncpointGpu = 0;

}

void Tausch::syncCpuAndGpu(bool iAmTheCPU) {

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
}

void Tausch::setOpenCLInfo(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue) {

    this->cl_defaultDevice = cl_defaultDevice;
    this->cl_context = cl_context;
    this->cl_queue = cl_queue;

    gpuEnabled = true;

    try {
        compileKernels();
    } catch(cl::Error error) {
        std::cout << "[kernel compile] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        if(error.err() == -11) {
            std::string log = cl_programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_defaultDevice);
            std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** " << std::endl << std::endl;
        }
    }

}

void Tausch::compileKernels() {

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

    cl_programs = cl::Program(cl_context, oclstr, false);
    cl_programs.build("");

}

// Create OpenCL context and choose a device (if multiple devices are available, the MPI ranks will split up evenly)
void Tausch::setupOpenCL() {

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
        std::cout << "Rank " << mpiRank << " using OpenCL platform #" << platform_num[mpiRank%num] << " with device #" << device_num[mpiRank%num] << ": " << cl_defaultDevice.getInfo<CL_DEVICE_NAME>() << std::endl;

        delete[] platform_num;
        delete[] device_num;

        // Create context and queue
        cl_context = cl::Context({cl_defaultDevice});
        cl_queue = cl::CommandQueue(cl_context,cl_defaultDevice);

        // The OpenCL kernel
        compileKernels();

    } catch(cl::Error error) {
        std::cout << "[setup] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        if(error.err() == -11) {
            std::string log = cl_programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_defaultDevice);
            std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** " << std::endl << std::endl;
        }
        exit(1);
    }

}
