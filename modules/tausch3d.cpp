#include "tausch3d.h"

Tausch3D::Tausch3D(int localDimX, int localDimY, int localDimZ, int mpiNumX, int mpiNumY, int mpiNumZ, int haloWidth, MPI_Comm comm) {

    MPI_Comm_dup(comm, &TAUSCH_COMM);

    // get MPI info
    MPI_Comm_rank(TAUSCH_COMM, &mpiRank);
    MPI_Comm_size(TAUSCH_COMM, &mpiSize);

    this->mpiNumX = mpiNumX;
    this->mpiNumY = mpiNumY;
    this->mpiNumZ = mpiNumZ;

    // store configuration
    this->localDimX = localDimX;
    this->localDimY = localDimY;
    this->localDimZ = localDimZ;

    this->haloWidth = haloWidth;

    // check if this rank has a boundary with another rank
    haveBoundary[Left] = (mpiRank%mpiNumX != 0);
    haveBoundary[Right] = ((mpiRank+1)%mpiNumX != 0);
    haveBoundary[Top] = (mpiRank%(mpiNumX*mpiNumY) < (mpiNumX*mpiNumY-mpiNumX));
    haveBoundary[Bottom] = (mpiRank%(mpiNumX*mpiNumY) > (mpiNumX-1));
    haveBoundary[Front] = (mpiRank > mpiNumX*mpiNumY-1);
    haveBoundary[Back] = mpiRank < mpiSize-mpiNumX*mpiNumY;

    // a send and recv buffer for the CPU-CPU communication
    cpuToCpuSendBuffer = new real_t*[6];
    cpuToCpuSendBuffer[Left] = new real_t[haloWidth*(localDimY+2*haloWidth)*(localDimZ+2*haloWidth)]{};
    cpuToCpuSendBuffer[Right] = new real_t[haloWidth*(localDimY+2*haloWidth)*(localDimZ+2*haloWidth)]{};
    cpuToCpuSendBuffer[Top] = new real_t[haloWidth*(localDimX+2*haloWidth)*(localDimZ+2*haloWidth)]{};
    cpuToCpuSendBuffer[Bottom] = new real_t[haloWidth*(localDimX+2*haloWidth)*(localDimZ+2*haloWidth)]{};
    cpuToCpuSendBuffer[Front] = new real_t[haloWidth*(localDimX+2*haloWidth)*(localDimY+2*haloWidth)]{};
    cpuToCpuSendBuffer[Back] = new real_t[haloWidth*(localDimX+2*haloWidth)*(localDimY+2*haloWidth)]{};
    cpuToCpuRecvBuffer = new real_t*[6];
    cpuToCpuRecvBuffer[Left] = new real_t[haloWidth*(localDimY+2*haloWidth)*(localDimZ+2*haloWidth)]{};
    cpuToCpuRecvBuffer[Right] = new real_t[haloWidth*(localDimY+2*haloWidth)*(localDimZ+2*haloWidth)]{};
    cpuToCpuRecvBuffer[Top] = new real_t[haloWidth*(localDimX+2*haloWidth)*(localDimZ+2*haloWidth)]{};
    cpuToCpuRecvBuffer[Bottom] = new real_t[haloWidth*(localDimX+2*haloWidth)*(localDimZ+2*haloWidth)]{};
    cpuToCpuRecvBuffer[Front] = new real_t[haloWidth*(localDimX+2*haloWidth)*(localDimY+2*haloWidth)]{};
    cpuToCpuRecvBuffer[Back] = new real_t[haloWidth*(localDimX+2*haloWidth)*(localDimY+2*haloWidth)]{};

    // whether the cpu/gpu pointers have been passed
    cpuInfoGiven = false;

    // cpu at beginning
    cpuRecvsPosted = false;

    // communication to neither edge has been started
    cpuStarted[Left] = false;
    cpuStarted[Right] = false;
    cpuStarted[Top] = false;
    cpuStarted[Bottom] = false;
    cpuStarted[Front] = false;
    cpuStarted[Back] = false;

#ifdef TAUSCH_OPENCL

    gpuInfoGiven = false;
    gpuEnabled = false;

    gpuToCpuStarted = false;
    cpuToGpuStarted = false;

    // used for syncing the CPU and GPU thread
    sync_counter[0].store(0);
    sync_counter[1].store(0);
    sync_lock[0].store(0);
    sync_lock[1].store(0);

#endif

}

Tausch3D::~Tausch3D() {

    // clean up memory
    for(int i = 0; i < 6; ++i) {
        delete[] cpuToCpuSendBuffer[i];
        delete[] cpuToCpuRecvBuffer[i];
    }
    delete[] cpuToCpuSendBuffer;
    delete[] cpuToCpuRecvBuffer;
#ifdef TAUSCH_OPENCL
    if(gpuEnabled) {
        delete[] cpuToGpuBuffer;
        delete[] gpuToCpuBuffer;
    }
#endif

}

// get a pointer to the CPU data
void Tausch3D::setCPUData(real_t *dat) {
    cpuInfoGiven = true;
    cpuData = dat;
}

// post the MPI_Irecv's for inter-rank communication
void Tausch3D::postCpuReceives() {

    if(!cpuInfoGiven) {
        std::cerr << "ERROR: You didn't tell me yet where to find the data! Abort..." << std::endl;
        exit(1);
    }

    cpuRecvsPosted = true;

    MPI_Datatype mpiDataType = ((sizeof(real_t) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT);

    if(haveBoundary[Left])
        MPI_Irecv(&cpuToCpuRecvBuffer[Left][0], haloWidth*(localDimY+2*haloWidth)*(localDimZ+2*haloWidth),
                mpiDataType, mpiRank-1, 0, TAUSCH_COMM, &cpuToCpuRecvRequest[Left]);
    if(haveBoundary[Right])
        MPI_Irecv(&cpuToCpuRecvBuffer[Right][0], haloWidth*(localDimY+2*haloWidth)*(localDimZ+2*haloWidth),
                mpiDataType, mpiRank+1, 2, TAUSCH_COMM, &cpuToCpuRecvRequest[Right]);
    if(haveBoundary[Top])
        MPI_Irecv(&cpuToCpuRecvBuffer[Top][0], haloWidth*(localDimX+2*haloWidth)*(localDimZ+2*haloWidth),
                mpiDataType, mpiRank+mpiNumX, 1, TAUSCH_COMM, &cpuToCpuRecvRequest[Top]);
    if(haveBoundary[Bottom])
        MPI_Irecv(&cpuToCpuRecvBuffer[Bottom][0], haloWidth*(localDimX+2*haloWidth)*(localDimZ+2*haloWidth),
                mpiDataType, mpiRank-mpiNumX, 3, TAUSCH_COMM, &cpuToCpuRecvRequest[Bottom]);
    if(haveBoundary[Front])
        MPI_Irecv(&cpuToCpuRecvBuffer[Front][0], haloWidth*(localDimX+2*haloWidth)*(localDimY+2*haloWidth),
                mpiDataType, mpiRank-mpiNumX*mpiNumY, 4, TAUSCH_COMM, &cpuToCpuRecvRequest[Front]);
    if(haveBoundary[Back])
        MPI_Irecv(&cpuToCpuRecvBuffer[Back][0], haloWidth*(localDimX+2*haloWidth)*(localDimY+2*haloWidth),
                mpiDataType, mpiRank+mpiNumX*mpiNumY, 5, TAUSCH_COMM, &cpuToCpuRecvRequest[Back]);

}

void Tausch3D::startCpuEdge(Edge edge) {

    if(!cpuRecvsPosted) {
        std::cerr << "ERROR: No CPU Recvs have been posted yet... Abort!" << std::endl;
        exit(1);
    }

    if(edge != Left && edge != Right && edge != Top && edge != Bottom && edge != Front && edge != Back) {
        std::cerr << "startCpuEdge(): ERROR: Invalid edge specified: " << edge << std::endl;
        exit(1);
    }

    cpuStarted[edge] = true;

    MPI_Datatype mpiDataType = ((sizeof(real_t) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT);

    if(edge == Left && haveBoundary[Left]) {
        for(int z = 0; z < localDimZ+2*haloWidth; ++z)
            for(int y = 0; y < localDimY+2*haloWidth; ++y)
                for(int x = 0; x < haloWidth; ++x)
                    cpuToCpuSendBuffer[Left][z*(localDimY+2*haloWidth)*haloWidth + y*haloWidth + x]
                            = cpuData[haloWidth + z*(localDimX+2*haloWidth)*(localDimY+2*haloWidth) + y*(localDimX+2*haloWidth)+x];
        MPI_Isend(&cpuToCpuSendBuffer[Left][0], haloWidth*(localDimY+2*haloWidth)*(localDimZ+2*haloWidth),
                mpiDataType, mpiRank-1, 2, TAUSCH_COMM, &cpuToCpuSendRequest[Left]);
    } else if(edge == Right && haveBoundary[Right]) {
        for(int z = 0; z < localDimZ+2*haloWidth; ++z)
            for(int y = 0; y < localDimY+2*haloWidth; ++y)
                for(int x = 0; x < haloWidth; ++x)
                    cpuToCpuSendBuffer[Right][z*(localDimY+2*haloWidth)*haloWidth + y*haloWidth + x]
                            = cpuData[localDimX + z*(localDimX+2*haloWidth)*(localDimY+2*haloWidth) + y*(localDimX+2*haloWidth)+x];
        MPI_Isend(&cpuToCpuSendBuffer[Right][0], haloWidth*(localDimY+2*haloWidth)*(localDimZ+2*haloWidth),
                mpiDataType, mpiRank+1, 0, TAUSCH_COMM, &cpuToCpuSendRequest[Right]);
    } else if(edge == Top && haveBoundary[Top]) {
        for(int z = 0; z < localDimZ+2*haloWidth; ++z)
            for(int y = 0; y < haloWidth; ++y)
                for(int x = 0; x < localDimX+2*haloWidth; ++x)
                    cpuToCpuSendBuffer[Top][z*(localDimX+2*haloWidth)*haloWidth + y*(localDimX+2*haloWidth) + x]
                            = cpuData[z*(localDimX+2*haloWidth)*(localDimY+2*haloWidth) + (y+localDimY)*(localDimX+2*haloWidth)+x];
        MPI_Isend(&cpuToCpuSendBuffer[Top][0], haloWidth*(localDimX+2*haloWidth)*(localDimZ+2*haloWidth),
                mpiDataType, mpiRank+mpiNumX, 3, TAUSCH_COMM, &cpuToCpuSendRequest[Top]);
    } else if(edge == Bottom && haveBoundary[Bottom]) {
        for(int z = 0; z < localDimZ+2*haloWidth; ++z)
            for(int y = 0; y < haloWidth; ++y)
                for(int x = 0; x < localDimX+2*haloWidth; ++x)
                    cpuToCpuSendBuffer[Bottom][z*(localDimX+2*haloWidth)*haloWidth + y*(localDimX+2*haloWidth) + x]
                            = cpuData[z*(localDimX+2*haloWidth)*(localDimY+2*haloWidth) + (y+haloWidth)*(localDimX+2*haloWidth)+x];
        MPI_Isend(&cpuToCpuSendBuffer[Bottom][0], haloWidth*(localDimX+2*haloWidth)*(localDimZ+2*haloWidth),
                mpiDataType, mpiRank-mpiNumX, 1, TAUSCH_COMM, &cpuToCpuSendRequest[Bottom]);
    } else if(edge == Front && haveBoundary[Front]) {
        for(int z = 0; z < haloWidth; ++z)
            for(int y = 0; y < localDimY+2*haloWidth; ++y)
                for(int x = 0; x < localDimX+2*haloWidth; ++x)
                    cpuToCpuSendBuffer[Front][z*(localDimY+2*haloWidth)*(localDimX+2*haloWidth) + y*(localDimX+2*haloWidth) + x]
                            = cpuData[(z+haloWidth)*(localDimX+2*haloWidth)*(localDimY+2*haloWidth) + y*(localDimX+2*haloWidth)+x];
        MPI_Isend(&cpuToCpuSendBuffer[Front][0], haloWidth*(localDimX+2*haloWidth)*(localDimY+2*haloWidth),
                mpiDataType, mpiRank-mpiNumX*mpiNumY, 5, TAUSCH_COMM, &cpuToCpuSendRequest[Front]);
    } else if(edge == Back && haveBoundary[Back]) {
        for(int z = 0; z < haloWidth; ++z)
            for(int y = 0; y < localDimY+2*haloWidth; ++y)
                for(int x = 0; x < localDimX+2*haloWidth; ++x)
                    cpuToCpuSendBuffer[Back][z*(localDimY+2*haloWidth)*(localDimX+2*haloWidth) + y*(localDimX+2*haloWidth) + x]
                            = cpuData[(z+localDimZ)*(localDimX+2*haloWidth)*(localDimY+2*haloWidth) + y*(localDimX+2*haloWidth)+x];
        MPI_Isend(&cpuToCpuSendBuffer[Back][0], haloWidth*(localDimX+2*haloWidth)*(localDimY+2*haloWidth),
                mpiDataType, mpiRank+mpiNumX*mpiNumY, 4, TAUSCH_COMM, &cpuToCpuSendRequest[Back]);
    }

}

// Complete CPU-CPU exchange to the left
void Tausch3D::completeCpuEdge(Edge edge) {

    if(edge != Left && edge != Right && edge != Top && edge != Bottom && edge != Front && edge != Back) {
        std::cerr << "completeCpuEdge(): ERROR: Invalid edge specified: " << edge << std::endl;
        exit(1);
    }

    if(!cpuStarted[edge]) {
        std::cerr << "ERROR: No edge #" << edge << " CPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    if(edge == Left && haveBoundary[Left]) {
        MPI_Wait(&cpuToCpuRecvRequest[Left], MPI_STATUS_IGNORE);
        for(int z = 0; z < localDimZ+2*haloWidth; ++z)
            for(int y = 0; y < localDimY+2*haloWidth; ++y)
                for(int hw = 0; hw < haloWidth; ++hw)
                    cpuData[z*(localDimX+2*haloWidth)*(localDimY+2*haloWidth) + y*(localDimX+2*haloWidth)+hw]
                            = cpuToCpuRecvBuffer[Left][z*(localDimY+2*haloWidth)*haloWidth + y*haloWidth + hw];
        MPI_Wait(&cpuToCpuSendRequest[Left], MPI_STATUS_IGNORE);
    } else if(edge == Right && haveBoundary[Right]) {
        MPI_Wait(&cpuToCpuRecvRequest[Right], MPI_STATUS_IGNORE);
        for(int z = 0; z < localDimZ+2*haloWidth; ++z)
            for(int y = 0; y < localDimY+2*haloWidth; ++y)
                for(int hw = 0; hw < haloWidth; ++hw)
                    cpuData[localDimX+haloWidth + z*(localDimX+2*haloWidth)*(localDimY+2*haloWidth) + y*(localDimX+2*haloWidth)+hw]
                            = cpuToCpuRecvBuffer[Right][z*(localDimY+2*haloWidth)*haloWidth + y*haloWidth + hw];
        MPI_Wait(&cpuToCpuSendRequest[Right], MPI_STATUS_IGNORE);
    } else if(edge == Top && haveBoundary[Top]) {
        MPI_Wait(&cpuToCpuRecvRequest[Top], MPI_STATUS_IGNORE);
        for(int z = 0; z < localDimZ+2*haloWidth; ++z)
            for(int y = 0; y < haloWidth; ++y)
                for(int x = 0; x < localDimX+2*haloWidth; ++x)
                    cpuData[z*(localDimX+2*haloWidth)*(localDimY+2*haloWidth) + (y+localDimY+haloWidth)*(localDimX+2*haloWidth)+x]
                            = cpuToCpuRecvBuffer[Top][z*(localDimX+2*haloWidth)*haloWidth + y*(localDimX+2*haloWidth) + x];
        MPI_Wait(&cpuToCpuSendRequest[Top], MPI_STATUS_IGNORE);
    } else if(edge == Bottom && haveBoundary[Bottom]) {
        MPI_Wait(&cpuToCpuRecvRequest[Bottom], MPI_STATUS_IGNORE);
        for(int z = 0; z < localDimZ+2*haloWidth; ++z)
            for(int y = 0; y < haloWidth; ++y)
                for(int x = 0; x < localDimX+2*haloWidth; ++x)
                    cpuData[z*(localDimX+2*haloWidth)*(localDimY+2*haloWidth) + y*(localDimX+2*haloWidth)+x]
                            = cpuToCpuRecvBuffer[Bottom][z*(localDimX+2*haloWidth)*haloWidth + y*(localDimX+2*haloWidth) + x];
        MPI_Wait(&cpuToCpuSendRequest[Bottom], MPI_STATUS_IGNORE);
    } else if(edge == Front && haveBoundary[Front]) {
        MPI_Wait(&cpuToCpuRecvRequest[Front], MPI_STATUS_IGNORE);
        for(int z = 0; z < haloWidth; ++z)
            for(int y = 0; y < localDimY+2*haloWidth; ++y)
                for(int x = 0; x < localDimX+2*haloWidth; ++x)
                    cpuData[z*(localDimX+2*haloWidth)*(localDimY+2*haloWidth) + y*(localDimX+2*haloWidth)+x]
                            = cpuToCpuRecvBuffer[Front][z*(localDimY+2*haloWidth)*(localDimY+2*haloWidth) + y*(localDimX+2*haloWidth) + x];
        MPI_Wait(&cpuToCpuSendRequest[Front], MPI_STATUS_IGNORE);
    } else if(edge == Back && haveBoundary[Back]) {
        MPI_Wait(&cpuToCpuRecvRequest[Back], MPI_STATUS_IGNORE);
        for(int z = 0; z < haloWidth; ++z)
            for(int y = 0; y < localDimY+2*haloWidth; ++y)
                for(int x = 0; x < localDimX+2*haloWidth; ++x)
                    cpuData[(z+haloWidth+localDimZ)*(localDimX+2*haloWidth)*(localDimY+2*haloWidth) + y*(localDimX+2*haloWidth)+x]
                            = cpuToCpuRecvBuffer[Back][z*(localDimY+2*haloWidth)*(localDimX+2*haloWidth) + y*(localDimX+2*haloWidth) + x];
        MPI_Wait(&cpuToCpuSendRequest[Back], MPI_STATUS_IGNORE);
    }

}

#ifdef TAUSCH_OPENCL

void Tausch3D::enableOpenCL(bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {

    // gpu disabled by default, only enabled if flag is set
    gpuEnabled = true;
    // local workgroup size
    cl_kernelLocalSize = clLocalWorkgroupSize;
    this->blockingSyncCpuGpu = blockingSyncCpuGpu;
    // Tausch creates its own OpenCL environment
    this->setupOpenCL(giveOpenCLDeviceName);

    cl_haloWidth = cl::Buffer(cl_context, &haloWidth, (&haloWidth)+1, true);

}

// If Tausch didn't set up OpenCL, the user needs to pass some OpenCL variables
void Tausch3D::enableOpenCL(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue, bool blockingSyncCpuGpu, int clLocalWorkgroupSize) {

    this->cl_defaultDevice = cl_defaultDevice;
    this->cl_context = cl_context;
    this->cl_queue = cl_queue;
    this->blockingSyncCpuGpu = blockingSyncCpuGpu;
    this->cl_kernelLocalSize = clLocalWorkgroupSize;

    cl_haloWidth = cl::Buffer(cl_context, &haloWidth, (&haloWidth)+1, true);

    gpuEnabled = true;

    compileKernels();

}

// get a pointer to the GPU buffer and its dimensions
void Tausch3D::setGPUData(cl::Buffer &dat, int gpuDimX, int gpuDimY, int gpuDimZ) {

    // check whether OpenCL has been set up
    if(!gpuEnabled) {
        std::cerr << "ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    gpuInfoGiven = true;

    // store parameters
    gpuData = dat;
    this->gpuDimX = gpuDimX;
    this->gpuDimY = gpuDimY;
    this->gpuDimZ = gpuDimZ;

    // store buffer to store the GPU and the CPU part of the halo.
    // We do not need two buffers each, as each thread has direct access to both arrays, no communication necessary
    cpuToGpuBuffer = new std::atomic<real_t>[2*haloWidth*(gpuDimX+2*haloWidth)*(gpuDimZ+2*haloWidth) + 2*haloWidth*gpuDimY*(gpuDimZ+2*haloWidth) +  2*haloWidth*gpuDimX*gpuDimY]{};
    gpuToCpuBuffer = new std::atomic<real_t>[2*haloWidth*gpuDimX*gpuDimZ + 2*haloWidth*gpuDimY*gpuDimZ + 2*haloWidth*gpuDimX*gpuDimY]{};

    // set up buffers on device
    try {
        cl_cpuToGpuBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (2*haloWidth*(gpuDimX+2*haloWidth)*(gpuDimZ+2*haloWidth) + 2*haloWidth*gpuDimY*(gpuDimZ+2*haloWidth) +  2*haloWidth*gpuDimX*gpuDimY)*sizeof(real_t));
        cl_gpuToCpuBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (2*haloWidth*gpuDimX*gpuDimZ + 2*haloWidth*gpuDimY*gpuDimZ + 2*haloWidth*gpuDimX*gpuDimY)*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_cpuToGpuBuffer, 0, 0, (2*haloWidth*(gpuDimX+2*haloWidth)*(gpuDimZ+2*haloWidth) + 2*haloWidth*gpuDimY*(gpuDimZ+2*haloWidth) +  2*haloWidth*gpuDimX*gpuDimY)*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_gpuToCpuBuffer, 0, 0, (2*haloWidth*gpuDimX*gpuDimZ + 2*haloWidth*gpuDimY*gpuDimZ + 2*haloWidth*gpuDimX*gpuDimY)*sizeof(real_t));
        cl_gpuDimX = cl::Buffer(cl_context, &gpuDimX, (&gpuDimX)+1, true);
        cl_gpuDimY = cl::Buffer(cl_context, &gpuDimY, (&gpuDimY)+1, true);
        cl_gpuDimY = cl::Buffer(cl_context, &gpuDimZ, (&gpuDimZ)+1, true);
    } catch(cl::Error error) {
        std::cout << "[setup send/recv buffer] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

// collect cpu side of cpu/gpu halo and store in buffer
void Tausch3D::startCpuToGpu() {
/*
    // check whether GPU is enabled
    if(!gpuEnabled) {
        std::cerr << "ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    cpuToGpuStarted = true;

    // left
    for(int i = 0; i < haloWidth*gpuDimY; ++ i) {
        int index = ((localDimY-gpuDimY)/2 +i/haloWidth+haloWidth)*(localDimX+2*haloWidth) + (localDimX-gpuDimX)/2+i%haloWidth;
        cpuToGpuBuffer[i].store(cpuData[index]);
    }
    // right
    for(int i = 0; i < haloWidth*gpuDimY; ++ i) {
        int index = ((localDimY-gpuDimY)/2 +haloWidth + i/haloWidth)*(localDimX+2*haloWidth) + (localDimX-gpuDimX)/2 + haloWidth + gpuDimX + i%haloWidth;
        cpuToGpuBuffer[haloWidth*gpuDimY + i].store(cpuData[index]);
    }
    // top
    for(int i = 0; i < haloWidth*(gpuDimX+2*haloWidth); ++ i) {
        int index = ((localDimY-gpuDimY)/2+gpuDimY+haloWidth + i/(gpuDimX+2*haloWidth))*(localDimX+2*haloWidth) + haloWidth + ((localDimX-gpuDimX)/2-haloWidth) +i%(gpuDimX+2*haloWidth);
        cpuToGpuBuffer[2*haloWidth*gpuDimY + i].store(cpuData[index]);
    }
    // bottom
    for(int i = 0; i < haloWidth*(gpuDimX+2*haloWidth); ++ i) {
        int index = ((localDimY-gpuDimY)/2 +i/(gpuDimX+2*haloWidth))*(localDimX+2*haloWidth) + (localDimX-gpuDimX)/2 + i%(gpuDimX+2*haloWidth);
        cpuToGpuBuffer[2*haloWidth*gpuDimY+haloWidth*(gpuDimX+2*haloWidth) + i].store(cpuData[index]);
    }
*/
}

// collect gpu side of cpu/gpu halo and download into buffer
void Tausch3D::startGpuToCpu() {
/*
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

        auto kernel_collectHalo = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(cl_programs, "collectHaloData");

        int globalSize = ((2*haloWidth*gpuDimX+2*haloWidth*(gpuDimY-2*haloWidth))/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_collectHalo(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                           cl_gpuDimX, cl_gpuDimY, cl_haloWidth, gpuData, cl_gpuToCpuBuffer);

        double *dat = new double[2*haloWidth*gpuDimX+2*haloWidth*(gpuDimY-2*haloWidth)];
        cl::copy(cl_queue, cl_gpuToCpuBuffer, &dat[0], (&dat[2*haloWidth*gpuDimX+2*haloWidth*(gpuDimY-2*haloWidth)-1])+1);
        for(int i = 0; i < 2*haloWidth*gpuDimX+2*haloWidth*(gpuDimY-2*haloWidth); ++i)
            gpuToCpuBuffer[i].store(dat[i]);

        delete[] dat;

    } catch(cl::Error error) {
        std::cout << "[kernel collectHalo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }
*/
}

// Complete CPU side of CPU/GPU halo exchange
void Tausch3D::completeCpuToGpu() {
/*
    if(!cpuToGpuStarted) {
        std::cerr << "ERROR: No CPU->GPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    // we need to wait for the GPU thread to arrive here
    if(blockingSyncCpuGpu)
        syncCpuAndGpu();

    // left
    for(int i = 0; i < haloWidth*(gpuDimY-2*haloWidth); ++ i) {
        int index = ((localDimY-gpuDimY)/2 +i/haloWidth +2*haloWidth)*(localDimX+2*haloWidth) + (localDimX-gpuDimX)/2 + haloWidth +i%haloWidth;
        cpuData[index] = gpuToCpuBuffer[i].load();
    }
    // right
    for(int i = 0; i < haloWidth*(gpuDimY-2*haloWidth); ++ i) {
        int index = ((localDimY-gpuDimY)/2 +2*haloWidth + i/haloWidth)*(localDimX+2*haloWidth) + (localDimX-gpuDimX)/2 + gpuDimX + i%haloWidth;
        cpuData[index] = gpuToCpuBuffer[haloWidth*(gpuDimY-2*haloWidth) + i].load();
    }
    // top
    for(int i = 0; i < haloWidth*gpuDimX; ++ i) {
        int index = ((localDimY-gpuDimY)/2+gpuDimY + i/(gpuDimX))*(localDimX+2*haloWidth) + 2*haloWidth + ((localDimX-gpuDimX)/2-haloWidth) +i%(gpuDimX);
        cpuData[index] = gpuToCpuBuffer[2*haloWidth*(gpuDimY-2*haloWidth) + i].load();
    }
    // bottom
    for(int i = 0; i < haloWidth*gpuDimX; ++ i) {
        int index = ((localDimY-gpuDimY)/2 +haloWidth +i/(gpuDimX))*(localDimX+2*haloWidth) + (localDimX-gpuDimX)/2 +haloWidth +i%(gpuDimX);
        cpuData[index] = gpuToCpuBuffer[2*haloWidth*(gpuDimY-2*haloWidth)+haloWidth*gpuDimX + i].load();
    }
*/
}

// Complete GPU side of CPU/GPU halo exchange
void Tausch3D::completeGpuToCpu() {
/*
    if(!gpuToCpuStarted) {
        std::cerr << "ERROR: No GPU->CPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    // we need to wait for the CPU thread to arrive here
    if(blockingSyncCpuGpu)
        syncCpuAndGpu();

    try {

        double *dat = new double[2*haloWidth*(gpuDimX+2*haloWidth)+2*haloWidth*gpuDimY];
        for(int i = 0; i < 2*haloWidth*(gpuDimX+2*haloWidth)+2*haloWidth*gpuDimY; ++i)
            dat[i] = cpuToGpuBuffer[i].load();

        cl::copy(cl_queue, &dat[0], (&dat[2*haloWidth*(gpuDimX+2*haloWidth)+2*haloWidth*gpuDimY-1])+1, cl_cpuToGpuBuffer);

        delete[] dat;

        auto kernel_distributeHaloData = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(cl_programs, "distributeHaloData");

        int globalSize = ((2*haloWidth*(gpuDimX+2*haloWidth) + 2*haloWidth*gpuDimY)/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_distributeHaloData(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                                  cl_gpuDimX, cl_gpuDimY, cl_haloWidth, gpuData, cl_cpuToGpuBuffer);

    } catch(cl::Error error) {
        std::cout << "[dist halo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

*/
}

// both the CPU and GPU have to arrive at this point before either can continue
void Tausch3D::syncCpuAndGpu() {

    // need to do this twice to prevent potential (though unlikely) deadlocks
    for(int i = 0; i < 2; ++i) {

        if(sync_lock[i].load() == 0)
            sync_lock[i].store(1);
        int val = sync_counter[i].fetch_add(1);
        if(val == 1) {
            sync_counter[i].store(0);
            sync_lock[i].store(0);
        }
        while(sync_lock[i].load() == 1);

    }

}

void Tausch3D::compileKernels() {

    std::string oclstr;
    std::ifstream cl_file(std::string(SOURCEDIR) + "/kernels3d.cl");
    cl_file.seekg(0, std::ios::end);
    oclstr.reserve(cl_file.tellg());
    cl_file.seekg(0, std::ios::beg);
    oclstr.assign((std::istreambuf_iterator<char>(cl_file)), std::istreambuf_iterator<char>());

/*
    // Tausch requires two kernels: One for collecting the halo data and one for distributing that data
    std::string oclstr = "typedef " + std::string((sizeof(real_t)==sizeof(double)) ? "double" : "float") + " real_t;\n";
    oclstr += R"d(
kernel void collectHaloData(global const int * restrict const dimX, global const int * restrict const dimY,
                            global const int * restrict const haloWidth,
                            global const real_t * restrict const vec, global real_t * sync) {
    unsigned int current = get_global_id(0);
    unsigned int maxNum = 2*(*haloWidth)*(*dimX) + 2*(*haloWidth)*(*dimY);

    if(current >= maxNum)
        return;
    // left
    if(current < (*haloWidth)*(*dimY-2*(*haloWidth))) {
        int index = (*haloWidth+current/(*haloWidth)+(*haloWidth))*(*dimX+2*(*haloWidth)) +(*haloWidth) + current%(*haloWidth);
        sync[current] = vec[index];
        return;
    }
    // right
    if(current < 2*(*haloWidth)*(*dimY-2*(*haloWidth))) {
        int touse = current-(*haloWidth)*(*dimY-2*(*haloWidth));
        int index = (1+(*haloWidth)+touse/(*haloWidth)+(*haloWidth))*(*dimX+2*(*haloWidth)) -2*(*haloWidth)+touse%(*haloWidth);
        sync[current] = vec[index];
        return;
    }
    // top
    if(current < 2*(*haloWidth)*(*dimY-2*(*haloWidth)) + (*haloWidth)*(*dimX)) {
        int touse = current-2*(*haloWidth)*(*dimY-2*(*haloWidth));
        int index = (*dimX+2*(*haloWidth))*(*dimY + touse/(*dimX)) + touse%(*dimX)+(*haloWidth);
        sync[current] = vec[index];
        return;
    }
    // bottom
    int touse = current - 2*(*haloWidth)*(*dimY-2*(*haloWidth)) - (*haloWidth)*(*dimX);
    int index = ((*haloWidth)+touse/(*dimX))*(*dimX+2*(*haloWidth))+touse%(*dimX)+(*haloWidth);
    sync[current] = vec[index];
}
kernel void distributeHaloData(global const int * restrict const dimX, global const int * restrict const dimY,
                               global const int * restrict const haloWidth,
                               global real_t * vec, global const real_t * restrict const sync) {
    unsigned int current = get_global_id(0);
    unsigned int maxNum = 2*(*haloWidth)*(*dimX+2*(*haloWidth)) + 2*(*haloWidth)*(*dimY);
    if(current >= maxNum)
        return;
    // left
    if(current < (*haloWidth)*(*dimY)) {
        vec[(*haloWidth+current/(*haloWidth))*(*dimX+2*(*haloWidth)) + current%(*haloWidth)] = sync[current];
        return;
    }
    // right
    if(current < 2*(*haloWidth)*(*dimY)) {
        int touse = current-(*haloWidth)*(*dimY);
        vec[((*haloWidth)+1+touse/(*haloWidth))*(*dimX+2*(*haloWidth)) -(*haloWidth)+touse%(*haloWidth)] = sync[current];
        return;
    }
    // top
    if(current < 2*(*haloWidth)*(*dimY)+(*haloWidth)*(*dimX+2*(*haloWidth))) {
        int touse = current-2*(*haloWidth)*(*dimY);
        vec[(*dimX+2*(*haloWidth))*(*dimY+(*haloWidth) + touse/(*dimX+2*(*haloWidth))) + touse%(*dimX+2*(*haloWidth))] = sync[current];
        return;
    }
    // bottom
    int touse = current - 2*(*haloWidth)*(*dimY) - (*haloWidth)*(*dimX+2*(*haloWidth));
    vec[(touse/(*dimX+2*(*haloWidth)))*(*dimX+2*(*haloWidth))+touse%(*dimX+2*(*haloWidth))] = sync[current];
}
                         )d";

    try {
        cl_programs = cl::Program(cl_context, oclstr, true);
    } catch(cl::Error error) {
        std::cout << "[kernel compile] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        if(error.err() == -11) {
            std::string log = cl_programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_defaultDevice);
            std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** " << std::endl << std::endl;
        }
    }
*/
}

// Create OpenCL context and choose a device (if multiple devices are available, the MPI ranks will split up evenly)
void Tausch3D::setupOpenCL(bool giveOpenCLDeviceName) {

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
            for(int iRank = 0; iRank < mpiSize; ++iRank){
                if(mpiRank == iRank)
                    std::cout << "Rank " << mpiRank << " using OpenCL platform #" << platform_num[mpiRank%num]
                              << " with device #" << device_num[mpiRank%num] << ": " << cl_defaultDevice.getInfo<CL_DEVICE_NAME>() << std::endl;
                MPI_Barrier(TAUSCH_COMM);
            }
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
#endif
