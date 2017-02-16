#include "tausch.h"

Tausch::Tausch(int localDimX, int localDimY, int mpiNumX, int mpiNumY, bool withOpenCL) {

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

    cpuToGpuSendBuffer = nullptr;
    cpuToGpuRecvBuffer = nullptr;
    gpuToCpuSendBuffer = nullptr;
    gpuToCpuRecvBuffer = nullptr;

    cpuToCpuSendBuffer = new double[2*localDimX+2*localDimY + 4]{};
    cpuToCpuRecvBuffer = new double[2*localDimX+2*localDimY + 4]{};

    gpuEnabled = false;
    if(withOpenCL)
        setupOpenCL();

    gpuInfoGiven = false;
    cpuInfoGiven = false;

}

Tausch::~Tausch() {
//    if(cpuToCpuSendBuffer != nullptr)
//        delete[] cpuToCpuSendBuffer;
//    if(cpuToCpuRecvBuffer != nullptr)
//        delete[] cpuToCpuRecvBuffer;
//    if(cpuToGpuSendBuffer != nullptr)
//        delete[] cpuToGpuSendBuffer;
//    if(cpuToGpuRecvBuffer != nullptr)
//        delete[] cpuToGpuRecvBuffer;
//    if(gpuToCpuSendBuffer != nullptr)
//        delete[] gpuToCpuSendBuffer;
//    if(gpuToCpuRecvBuffer != nullptr)
//        delete[] gpuToCpuRecvBuffer;
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
    this->haloWidth = haloWidth;
    gpuData = dat;
    this->gpuWidth = gpuWidth;
    this->gpuHeight = gpuHeight;

    cpuToGpuSendBuffer = new double[2*(gpuWidth+2) + 2*gpuHeight]{};
    cpuToGpuRecvBuffer = new double[2*gpuWidth + 2*gpuHeight]{};
    gpuToCpuSendBuffer = new double[2*gpuWidth + 2*gpuHeight]{};
    gpuToCpuRecvBuffer = new double[2*(gpuWidth+2) + 2*gpuHeight]{};

    try {
        cl_gpuToCpuSendBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (2*gpuWidth+2*gpuHeight)*sizeof(double));
        cl_gpuToCpuRecvBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (2*(gpuWidth+2)+2*gpuHeight)*sizeof(double));
        cl_queue.enqueueFillBuffer(cl_gpuToCpuSendBuffer, 0, 0, 2*gpuWidth + 2*gpuHeight*sizeof(double));
        cl_queue.enqueueFillBuffer(cl_gpuToCpuRecvBuffer, 0, 0, 2*(gpuWidth+2) + 2*gpuHeight*sizeof(double));
        cl_gpuWidth = cl::Buffer(cl_context, &gpuWidth, (&gpuWidth)+1, true);
        cl_gpuHeight = cl::Buffer(cl_context, &gpuHeight, (&gpuHeight)+1, true);
    } catch(cl::Error error) {
        std::cout << "[setup send/recv buffer] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

void Tausch::postCpuReceives() {

    if(haveLeftBoundary) {
        MPI_Irecv(&cpuToCpuRecvBuffer[0], localDimY, MPI_DOUBLE, mpiRank-1, 0, MPI_COMM_WORLD, &cpuToCpuLeftRecvRequest);
        allCpuRequests.push_back(cpuToCpuLeftRecvRequest);
    }
    if(haveRightBoundary) {
        MPI_Irecv(&cpuToCpuRecvBuffer[localDimY], localDimY, MPI_DOUBLE, mpiRank+1, 2, MPI_COMM_WORLD, &cpuToCpuRightRecvRequest);
        allCpuRequests.push_back(cpuToCpuRightRecvRequest);
    }
    if(haveTopBoundary) {
        MPI_Irecv(&cpuToCpuRecvBuffer[2*localDimY], localDimX, MPI_DOUBLE, mpiRank+mpiNumX, 1, MPI_COMM_WORLD, &cpuToCpuTopRecvRequest);
        allCpuRequests.push_back(cpuToCpuTopRecvRequest);
    }
    if(haveBottomBoundary) {
        MPI_Irecv(&cpuToCpuRecvBuffer[2*localDimY+localDimX], localDimX, MPI_DOUBLE, mpiRank-mpiNumX, 3, MPI_COMM_WORLD, &cpuToCpuBottomRecvRequest);
        allCpuRequests.push_back(cpuToCpuBottomRecvRequest);
    }

    if(haveLeftBoundary && haveBottomBoundary) {
        MPI_Irecv(&cpuToCpuRecvBuffer[2*localDimY+2*localDimX], 1, MPI_DOUBLE, mpiRank-mpiNumX-1, 4, MPI_COMM_WORLD, &cpuToCpuBottomLeftRecvRequest);
        allCpuRequests.push_back(cpuToCpuBottomLeftRecvRequest);
    }
    if(haveRightBoundary && haveBottomBoundary) {
        MPI_Irecv(&cpuToCpuRecvBuffer[2*localDimY+2*localDimX +1], 1, MPI_DOUBLE, mpiRank-mpiNumX+1, 5, MPI_COMM_WORLD, &cpuToCpuBottomRightRecvRequest);
        allCpuRequests.push_back(cpuToCpuBottomRightRecvRequest);
    }
    if(haveLeftBoundary && haveTopBoundary) {
        MPI_Irecv(&cpuToCpuRecvBuffer[2*localDimY+2*localDimX +2], 1, MPI_DOUBLE, mpiRank+mpiNumX-1, 6, MPI_COMM_WORLD, &cpuToCpuTopLeftRecvRequest);
        allCpuRequests.push_back(cpuToCpuTopLeftRecvRequest);
    }
    if(haveRightBoundary && haveTopBoundary) {
        MPI_Irecv(&cpuToCpuRecvBuffer[2*localDimY+2*localDimX+3], 1, MPI_DOUBLE, mpiRank+mpiNumX+1, 7, MPI_COMM_WORLD, &cpuToCpuTopRightRecvRequest);
        allCpuRequests.push_back(cpuToCpuTopRightRecvRequest);
    }

    if(gpuEnabled) {

        MPI_Irecv(&cpuToGpuRecvBuffer[0], gpuWidth, MPI_DOUBLE, mpiRank, 10, MPI_COMM_WORLD, &cpuToGpuLeftRecvRequest);
        allCpuRequests.push_back(cpuToGpuLeftRecvRequest);

        MPI_Irecv(&cpuToGpuRecvBuffer[gpuWidth], gpuWidth, MPI_DOUBLE, mpiRank, 12, MPI_COMM_WORLD, &cpuToGpuRightRecvRequest);
        allCpuRequests.push_back(cpuToGpuRightRecvRequest);

        MPI_Irecv(&cpuToGpuRecvBuffer[2*gpuWidth], gpuHeight, MPI_DOUBLE, mpiRank, 11, MPI_COMM_WORLD, &cpuToGpuTopRecvRequest);
        allCpuRequests.push_back(cpuToGpuTopRecvRequest);

        MPI_Irecv(&cpuToGpuRecvBuffer[2*gpuWidth + gpuHeight], gpuHeight, MPI_DOUBLE, mpiRank, 13, MPI_COMM_WORLD, &cpuToGpuBottomRecvRequest);
        allCpuRequests.push_back(cpuToGpuBottomRecvRequest);

    }

}

void Tausch::postGpuReceives() {

    MPI_Request req;

    MPI_Irecv(&gpuToCpuRecvBuffer[0], gpuHeight, MPI_DOUBLE, mpiRank, 14, MPI_COMM_WORLD, &gpuToCpuLeftRecvRequest);
    allGpuRequests.push_back(gpuToCpuLeftRecvRequest);

    MPI_Irecv(&gpuToCpuRecvBuffer[gpuHeight], gpuHeight, MPI_DOUBLE, mpiRank, 16, MPI_COMM_WORLD, &gpuToCpuRightRecvRequest);
    allGpuRequests.push_back(gpuToCpuRightRecvRequest);

    MPI_Irecv(&gpuToCpuRecvBuffer[2*gpuHeight], gpuWidth+2, MPI_DOUBLE, mpiRank, 15, MPI_COMM_WORLD, &gpuToCpuTopRecvRequest);
    allGpuRequests.push_back(gpuToCpuTopRecvRequest);

    MPI_Irecv(&gpuToCpuRecvBuffer[2*gpuHeight + gpuWidth+2], gpuWidth+2, MPI_DOUBLE, mpiRank, 17, MPI_COMM_WORLD, &gpuToCpuBottomRecvRequest);
    allGpuRequests.push_back(gpuToCpuBottomRecvRequest);

}

// Start creating sends and recvs for the halo data
void Tausch::startCpuTausch() {

    if(!cpuInfoGiven) {
        std::cerr << "ERROR: You didn't tell me yet where to find the data! Abort..." << std::endl;
        exit(1);
    }

    // Left
    if(haveLeftBoundary) {
        for(int i = 0; i < localDimY; ++i)
            cpuToCpuSendBuffer[i] = cpuData[1+ (i+1)*(localDimX+2)];
        MPI_Isend(&cpuToCpuSendBuffer[0], localDimY, MPI_DOUBLE, mpiRank-1, 2, MPI_COMM_WORLD, &cpuToCpuLeftSendRequest);
        allCpuRequests.push_back(cpuToCpuLeftSendRequest);
    }

    // Right
    if(haveRightBoundary) {
        for(int i = 0; i < localDimY; ++i)
            cpuToCpuSendBuffer[localDimY + i] = cpuData[(i+2)*(localDimX+2) -2];
        MPI_Isend(&cpuToCpuSendBuffer[localDimY], localDimY, MPI_DOUBLE, mpiRank+1, 0, MPI_COMM_WORLD, &cpuToCpuRightSendRequest);
        allCpuRequests.push_back(cpuToCpuRightSendRequest);
    }

    // Top
    if(haveTopBoundary) {
        for(int i = 0; i < localDimX; ++i)
            cpuToCpuSendBuffer[2*localDimY + i] = cpuData[(localDimX+2)*localDimY+1 + i];
        MPI_Isend(&cpuToCpuSendBuffer[2*localDimY], localDimX, MPI_DOUBLE, mpiRank+mpiNumX, 3, MPI_COMM_WORLD, &cpuToCpuTopSendRequest);
        allCpuRequests.push_back(cpuToCpuTopSendRequest);
    }

    // Bottom
    if(haveBottomBoundary) {
        for(int i = 0; i < localDimX; ++i)
            cpuToCpuSendBuffer[2*localDimY+localDimX + i] = cpuData[1+ localDimX+2 + i];
        MPI_Isend(&cpuToCpuSendBuffer[2*localDimY+localDimX], localDimX, MPI_DOUBLE, mpiRank-mpiNumX, 1, MPI_COMM_WORLD, &cpuToCpuBottomSendRequest);
        allCpuRequests.push_back(cpuToCpuBottomSendRequest);
    }

    if(haveLeftBoundary && haveBottomBoundary) {
        cpuToCpuSendBuffer[2*localDimX+2*localDimY] = cpuData[1+localDimX+2];
        MPI_Isend(&cpuToCpuSendBuffer[2*localDimX+2*localDimY], 1, MPI_DOUBLE, mpiRank-mpiNumX-1, 7, MPI_COMM_WORLD, &cpuToCpuBottomLeftSendRequest);
        allCpuRequests.push_back(cpuToCpuBottomLeftSendRequest);
    }
    if(haveRightBoundary && haveBottomBoundary) {
        cpuToCpuSendBuffer[2*localDimX+2*localDimY+1] = cpuData[localDimX+2 + localDimX];
        MPI_Isend(&cpuToCpuSendBuffer[2*localDimX+2*localDimY+1], 1, MPI_DOUBLE, mpiRank-mpiNumX+1, 6, MPI_COMM_WORLD, &cpuToCpuBottomRightSendRequest);
        allCpuRequests.push_back(cpuToCpuBottomRightSendRequest);
    }
    if(haveLeftBoundary && haveTopBoundary) {
        cpuToCpuSendBuffer[2*localDimX+2*localDimY+2] = cpuData[(localDimX+2)*localDimY + 1];
        MPI_Isend(&cpuToCpuSendBuffer[2*localDimX+2*localDimY+2], 1, MPI_DOUBLE, mpiRank+mpiNumX-1, 5, MPI_COMM_WORLD, &cpuToCpuTopLeftSendRequest);
        allCpuRequests.push_back(cpuToCpuTopLeftSendRequest);
    }
    if(haveRightBoundary && haveTopBoundary) {
        cpuToCpuSendBuffer[2*localDimX+2*localDimY+3] = cpuData[(localDimX+2)*localDimY + localDimX];
        MPI_Isend(&cpuToCpuSendBuffer[2*localDimX+2*localDimY+3], 1, MPI_DOUBLE, mpiRank+mpiNumX+1, 4, MPI_COMM_WORLD, &cpuToCpuTopRightSendRequest);
        allCpuRequests.push_back(cpuToCpuTopRightSendRequest);
    }

    if(gpuEnabled) {

        // left
        for(int i = 0; i < gpuHeight; ++ i)
            cpuToGpuSendBuffer[i] = cpuData[((localDimY-gpuHeight)/2 +i+1)*(localDimX+2) + (localDimX-gpuWidth)/2];
        MPI_Isend(&cpuToGpuSendBuffer[0], gpuHeight, MPI_DOUBLE, mpiRank, 14, MPI_COMM_WORLD, &cpuToGpuLeftSendRequest);
        allCpuRequests.push_back(cpuToGpuLeftSendRequest);
        // right
        for(int i = 0; i < gpuHeight; ++ i)
            cpuToGpuSendBuffer[gpuHeight + i] = cpuData[((localDimY-gpuHeight)/2 +i+1)*(localDimX+2) + (localDimX-gpuWidth)/2 + gpuWidth+1];
        MPI_Isend(&cpuToGpuSendBuffer[gpuHeight], gpuHeight, MPI_DOUBLE, mpiRank, 16, MPI_COMM_WORLD, &cpuToGpuRightSendRequest);
        allCpuRequests.push_back(cpuToGpuRightSendRequest);
        // top
        for(int i = 0; i < gpuWidth+2; ++ i)
            cpuToGpuSendBuffer[2*gpuHeight + i] = cpuData[((localDimY-gpuHeight)/2 +gpuHeight+1)*(localDimX+2) + (localDimX-gpuWidth)/2 + i];
        MPI_Isend(&cpuToGpuSendBuffer[2*gpuHeight], gpuWidth+2, MPI_DOUBLE, mpiRank, 15, MPI_COMM_WORLD, &cpuToGpuTopSendRequest);
        allCpuRequests.push_back(cpuToGpuTopSendRequest);
        // bottom
        for(int i = 0; i < gpuHeight+2; ++ i)
            cpuToGpuSendBuffer[2*gpuHeight+gpuWidth+2 + i] = cpuData[((localDimY-gpuHeight)/2)*(localDimX+2) + (localDimX-gpuWidth)/2 + i];
        MPI_Isend(&cpuToGpuSendBuffer[2*gpuHeight+gpuWidth+2], gpuWidth+2, MPI_DOUBLE, mpiRank, 17, MPI_COMM_WORLD, &cpuToGpuBottomSendRequest);
        allCpuRequests.push_back(cpuToGpuBottomSendRequest);

    }

}

void Tausch::startGpuTausch() {

    try {

        auto kernel_collectHalo = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(cl_programs, "collectHaloData");

        int globalSize = ((2*gpuWidth+2*gpuHeight)/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_collectHalo(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                           cl_gpuWidth, cl_gpuHeight, gpuData, cl_gpuToCpuSendBuffer);

        cl::copy(cl_queue, cl_gpuToCpuSendBuffer, &gpuToCpuSendBuffer[0], (&gpuToCpuSendBuffer[2*gpuWidth+2*gpuHeight-1])+1);

    } catch(cl::Error error) {
        std::cout << "[kernel collectHalo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    // left
    MPI_Isend(&gpuToCpuSendBuffer[0], gpuHeight, MPI_DOUBLE, mpiRank, 10, MPI_COMM_WORLD, &gpuToCpuLeftSendRequest);
    allGpuRequests.push_back(gpuToCpuLeftSendRequest);

    // right
    MPI_Isend(&gpuToCpuSendBuffer[gpuHeight], gpuHeight, MPI_DOUBLE, mpiRank, 12, MPI_COMM_WORLD, &gpuToCpuRightSendRequest);
    allGpuRequests.push_back(gpuToCpuRightSendRequest);

    // top
    MPI_Isend(&gpuToCpuSendBuffer[2*gpuHeight], gpuWidth, MPI_DOUBLE, mpiRank, 11, MPI_COMM_WORLD, &gpuToCpuTopSendRequest);
    allGpuRequests.push_back(gpuToCpuTopSendRequest);

    // bottom
    MPI_Isend(&gpuToCpuSendBuffer[2*gpuHeight + gpuWidth], gpuWidth, MPI_DOUBLE, mpiRank, 13, MPI_COMM_WORLD, &gpuToCpuBottomSendRequest);
    allGpuRequests.push_back(gpuToCpuBottomSendRequest);

}

void Tausch::completeCpuTausch() {

    // wait for all the local send/recvs to complete before moving on
    MPI_Waitall(allCpuRequests.size(), &allCpuRequests[0], MPI_STATUS_IGNORE);

    // distribute received data into halo regions

    // left
    if(haveLeftBoundary)
        for(int i = 0; i < localDimY; ++i)
            cpuData[(i+1)*(localDimX+2)] = cpuToCpuRecvBuffer[i];

    // right
    if(haveRightBoundary)
        for(int i = 0; i < localDimY; ++i)
            cpuData[(i+2)*(localDimX+2)-1] = cpuToCpuRecvBuffer[localDimY+i];

    // top
    if(haveTopBoundary)
        for(int i = 0; i < localDimX; ++i)
            cpuData[(localDimX+2)*(localDimY+1)+1 + i] = cpuToCpuRecvBuffer[2*localDimY+i];

    // bottom
    if(haveBottomBoundary)
        for(int i = 0; i < localDimX; ++i)
            cpuData[1+i] = cpuToCpuRecvBuffer[2*localDimY+localDimX+i];

    if(haveLeftBoundary && haveBottomBoundary)
        cpuData[0] = cpuToCpuRecvBuffer[2*localDimX+2*localDimY];
    if(haveRightBoundary && haveBottomBoundary)
        cpuData[localDimX+1] = cpuToCpuRecvBuffer[2*localDimX+2*localDimY + 1];
    if(haveLeftBoundary && haveTopBoundary)
        cpuData[(localDimX+2)*(localDimY+1)] = cpuToCpuRecvBuffer[2*localDimX+2*localDimY + 2];
    if(haveRightBoundary && haveTopBoundary)
        cpuData[(localDimX+2)*(localDimY+1) + localDimX+1] = cpuToCpuRecvBuffer[2*localDimX+2*localDimY + 3];


    if(gpuEnabled) {

        // left
        for(int i = 0; i < gpuHeight; ++ i)
            cpuData[((localDimY-gpuHeight)/2 +i+1)*(localDimX+2) + (localDimX-gpuWidth)/2 +1] = cpuToGpuRecvBuffer[i];
        // right
        for(int i = 0; i < gpuHeight; ++ i)
            cpuData[((localDimY-gpuHeight)/2 +i+1)*(localDimX+2) + (localDimX-gpuWidth)/2 + gpuWidth] = cpuToGpuRecvBuffer[gpuHeight + i];
        // top
        for(int i = 0; i < gpuWidth; ++ i)
            cpuData[((localDimY-gpuHeight)/2 +gpuHeight)*(localDimX+2) + (localDimX-gpuWidth)/2 + i+1] = cpuToGpuRecvBuffer[2*gpuHeight + i];
        // bottom
        for(int i = 0; i < gpuHeight; ++ i)
            cpuData[((localDimY-gpuHeight)/2 +1)*(localDimX+2) + (localDimX-gpuWidth)/2 + i+1] = cpuToGpuRecvBuffer[2*gpuHeight+gpuWidth + i];

    }

    // left
//    for(int i = 0; i < gpuWidth; ++i)

}

void Tausch::completeGpuTausch() {

    MPI_Waitall(allGpuRequests.size(), &allGpuRequests[0], MPI_STATUS_IGNORE);

    try {

        cl::copy(cl_queue, &gpuToCpuRecvBuffer[0], (&gpuToCpuRecvBuffer[2*(gpuWidth+2)+2*gpuHeight-1])+1, cl_gpuToCpuRecvBuffer);

        auto kernel_distributeHaloData = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(cl_programs, "distributeHaloData");

        int globalSize = ((2*gpuWidth + 2*gpuHeight)/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_distributeHaloData(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                                  cl_gpuWidth, cl_gpuHeight, gpuData, cl_gpuToCpuRecvBuffer);

    } catch(cl::Error error) {
        std::cout << "[dist halo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

// Create OpenCL context and choose a device (if multiple devices are available, the MPI ranks will split up evenly)
void Tausch::setupOpenCL() {

    gpuEnabled = true;
    cl_kernelLocalSize = 64;

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
        std::ifstream cl_file(std::string(SOURCEDIR) + "/kernels.cl");
        std::string str;
        cl_file.seekg(0, std::ios::end);
        str.reserve(cl_file.tellg());
        cl_file.seekg(0, std::ios::beg);
        str.assign((std::istreambuf_iterator<char>(cl_file)), std::istreambuf_iterator<char>());

        // Create program and build
        cl_programs = cl::Program(cl_context, str, false);
        cl_programs.build("");

    } catch(cl::Error error) {
        std::cout << "[setup] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        if(error.err() == -11) {
            std::string log = cl_programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_defaultDevice);
            std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** " << std::endl << std::endl;
        }
        exit(1);
    }

}
