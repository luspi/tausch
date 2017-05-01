#include "tausch3d.h"

Tausch3D::Tausch3D(int localDimX, int localDimY, int localDimZ, int mpiNumX, int mpiNumY, int mpiNumZ, int cpuHaloWidth, MPI_Comm comm) {

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

    this->cpuHaloWidth = cpuHaloWidth;

    // check if this rank has a boundary with another rank
    haveBoundary[LEFT] = (mpiRank%mpiNumX != 0);
    haveBoundary[RIGHT] = ((mpiRank+1)%mpiNumX != 0);
    haveBoundary[TOP] = (mpiRank%(mpiNumX*mpiNumY) < (mpiNumX*mpiNumY-mpiNumX));
    haveBoundary[BOTTOM] = (mpiRank%(mpiNumX*mpiNumY) > (mpiNumX-1));
    haveBoundary[FRONT] = (mpiRank > mpiNumX*mpiNumY-1);
    haveBoundary[BACK] = mpiRank < mpiSize-mpiNumX*mpiNumY;

    // a send and recv buffer for the CPU-CPU communication
    cpuToCpuSendBuffer = new real_t*[6];
    cpuToCpuSendBuffer[LEFT] = new real_t[cpuHaloWidth*(localDimY+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth)]{};
    cpuToCpuSendBuffer[RIGHT] = new real_t[cpuHaloWidth*(localDimY+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth)]{};
    cpuToCpuSendBuffer[TOP] = new real_t[cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth)]{};
    cpuToCpuSendBuffer[BOTTOM] = new real_t[cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth)]{};
    cpuToCpuSendBuffer[FRONT] = new real_t[cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth)]{};
    cpuToCpuSendBuffer[BACK] = new real_t[cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth)]{};
    cpuToCpuRecvBuffer = new real_t*[6];
    cpuToCpuRecvBuffer[LEFT] = new real_t[cpuHaloWidth*(localDimY+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth)]{};
    cpuToCpuRecvBuffer[RIGHT] = new real_t[cpuHaloWidth*(localDimY+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth)]{};
    cpuToCpuRecvBuffer[TOP] = new real_t[cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth)]{};
    cpuToCpuRecvBuffer[BOTTOM] = new real_t[cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth)]{};
    cpuToCpuRecvBuffer[FRONT] = new real_t[cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth)]{};
    cpuToCpuRecvBuffer[BACK] = new real_t[cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth)]{};

    // whether the cpu/gpu pointers have been passed
    cpuInfoGiven = false;

    // cpu at beginning
    cpuRecvsPosted = false;

    // communication to neither edge has been started
    cpuStarted[LEFT] = false;
    cpuStarted[RIGHT] = false;
    cpuStarted[TOP] = false;
    cpuStarted[BOTTOM] = false;
    cpuStarted[FRONT] = false;
    cpuStarted[BACK] = false;

    MPI_Datatype mpiDataType = ((sizeof(real_t) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT);

    // Initialise the Recv/Send operations
    if(haveBoundary[LEFT]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[LEFT], cpuHaloWidth*(localDimY+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth),
                      mpiDataType, mpiRank-1, 0, TAUSCH_COMM, &cpuToCpuRecvRequest[LEFT]);
        MPI_Send_init(cpuToCpuSendBuffer[LEFT], cpuHaloWidth*(localDimY+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth),
                      mpiDataType, mpiRank-1, 2, TAUSCH_COMM, &cpuToCpuSendRequest[LEFT]);
    }
    if(haveBoundary[RIGHT]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[RIGHT], cpuHaloWidth*(localDimY+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth),
                      mpiDataType, mpiRank+1, 2, TAUSCH_COMM, &cpuToCpuRecvRequest[RIGHT]);
        MPI_Send_init(cpuToCpuSendBuffer[RIGHT], cpuHaloWidth*(localDimY+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth),
                      mpiDataType, mpiRank+1, 0, TAUSCH_COMM, &cpuToCpuSendRequest[RIGHT]);
    }
    if(haveBoundary[TOP]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[TOP], cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth),
                      mpiDataType, mpiRank+mpiNumX, 1, TAUSCH_COMM, &cpuToCpuRecvRequest[TOP]);
        MPI_Send_init(cpuToCpuSendBuffer[TOP], cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth),
                      mpiDataType, mpiRank+mpiNumX, 3, TAUSCH_COMM, &cpuToCpuSendRequest[TOP]);
    }
    if(haveBoundary[BOTTOM]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[BOTTOM], cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth),
                      mpiDataType, mpiRank-mpiNumX, 3, TAUSCH_COMM, &cpuToCpuRecvRequest[BOTTOM]);
        MPI_Send_init(cpuToCpuSendBuffer[BOTTOM], cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimZ+2*cpuHaloWidth),
                      mpiDataType, mpiRank-mpiNumX, 1, TAUSCH_COMM, &cpuToCpuSendRequest[BOTTOM]);
    }
    if(haveBoundary[FRONT]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[FRONT], cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth),
                      mpiDataType, mpiRank-mpiNumX*mpiNumY, 4, TAUSCH_COMM, &cpuToCpuRecvRequest[FRONT]);
        MPI_Send_init(cpuToCpuSendBuffer[FRONT], cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth),
                      mpiDataType, mpiRank-mpiNumX*mpiNumY, 5, TAUSCH_COMM, &cpuToCpuSendRequest[FRONT]);
    }
    if(haveBoundary[BACK]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[BACK], cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth),
                      mpiDataType, mpiRank+mpiNumX*mpiNumY, 5, TAUSCH_COMM, &cpuToCpuRecvRequest[BACK]);
        MPI_Send_init(cpuToCpuSendBuffer[BACK], cpuHaloWidth*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth),
                      mpiDataType, mpiRank+mpiNumX*mpiNumY, 4, TAUSCH_COMM, &cpuToCpuSendRequest[BACK]);
    }

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

    if(haveBoundary[LEFT])
        MPI_Start(&cpuToCpuRecvRequest[LEFT]);
    if(haveBoundary[RIGHT])
        MPI_Start(&cpuToCpuRecvRequest[RIGHT]);
    if(haveBoundary[TOP])
        MPI_Start(&cpuToCpuRecvRequest[TOP]);
    if(haveBoundary[BOTTOM])
        MPI_Start(&cpuToCpuRecvRequest[BOTTOM]);
    if(haveBoundary[FRONT])
        MPI_Start(&cpuToCpuRecvRequest[FRONT]);
    if(haveBoundary[BACK])
        MPI_Start(&cpuToCpuRecvRequest[BACK]);

}

void Tausch3D::startCpuEdge(Edge edge) {

    if(!cpuRecvsPosted) {
        std::cerr << "ERROR: No CPU Recvs have been posted yet... Abort!" << std::endl;
        exit(1);
    }

    if(edge != LEFT && edge != RIGHT && edge != TOP && edge != BOTTOM && edge != FRONT && edge != BACK) {
        std::cerr << "startCpuEdge(): ERROR: Invalid edge specified: " << edge << std::endl;
        exit(1);
    }

    cpuStarted[edge] = true;

    MPI_Datatype mpiDataType = ((sizeof(real_t) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT);

    if(edge == LEFT && haveBoundary[LEFT]) {
        for(int z = 0; z < localDimZ+2*cpuHaloWidth; ++z)
            for(int y = 0; y < localDimY+2*cpuHaloWidth; ++y)
                for(int x = 0; x < cpuHaloWidth; ++x)
                    cpuToCpuSendBuffer[LEFT][z*(localDimY+2*cpuHaloWidth)*cpuHaloWidth + y*cpuHaloWidth + x]
                            = cpuData[cpuHaloWidth + z*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) + y*(localDimX+2*cpuHaloWidth)+x];
        MPI_Start(&cpuToCpuSendRequest[LEFT]);
    } else if(edge == RIGHT && haveBoundary[RIGHT]) {
        for(int z = 0; z < localDimZ+2*cpuHaloWidth; ++z)
            for(int y = 0; y < localDimY+2*cpuHaloWidth; ++y)
                for(int x = 0; x < cpuHaloWidth; ++x)
                    cpuToCpuSendBuffer[RIGHT][z*(localDimY+2*cpuHaloWidth)*cpuHaloWidth + y*cpuHaloWidth + x]
                            = cpuData[localDimX + z*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) + y*(localDimX+2*cpuHaloWidth)+x];
        MPI_Start(&cpuToCpuSendRequest[RIGHT]);
    } else if(edge == TOP && haveBoundary[TOP]) {
        for(int z = 0; z < localDimZ+2*cpuHaloWidth; ++z)
            for(int y = 0; y < cpuHaloWidth; ++y)
                for(int x = 0; x < localDimX+2*cpuHaloWidth; ++x)
                    cpuToCpuSendBuffer[TOP][z*(localDimX+2*cpuHaloWidth)*cpuHaloWidth + y*(localDimX+2*cpuHaloWidth) + x]
                            = cpuData[z*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) + (y+localDimY)*(localDimX+2*cpuHaloWidth)+x];
        MPI_Start(&cpuToCpuSendRequest[TOP]);
    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {
        for(int z = 0; z < localDimZ+2*cpuHaloWidth; ++z)
            for(int y = 0; y < cpuHaloWidth; ++y)
                for(int x = 0; x < localDimX+2*cpuHaloWidth; ++x)
                    cpuToCpuSendBuffer[BOTTOM][z*(localDimX+2*cpuHaloWidth)*cpuHaloWidth + y*(localDimX+2*cpuHaloWidth) + x]
                            = cpuData[z*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) + (y+cpuHaloWidth)*(localDimX+2*cpuHaloWidth)+x];
        MPI_Start(&cpuToCpuSendRequest[BOTTOM]);
    } else if(edge == FRONT && haveBoundary[FRONT]) {
        for(int z = 0; z < cpuHaloWidth; ++z)
            for(int y = 0; y < localDimY+2*cpuHaloWidth; ++y)
                for(int x = 0; x < localDimX+2*cpuHaloWidth; ++x)
                    cpuToCpuSendBuffer[FRONT][z*(localDimY+2*cpuHaloWidth)*(localDimX+2*cpuHaloWidth) + y*(localDimX+2*cpuHaloWidth) + x]
                            = cpuData[(z+cpuHaloWidth)*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) + y*(localDimX+2*cpuHaloWidth)+x];
        MPI_Start(&cpuToCpuSendRequest[FRONT]);
    } else if(edge == BACK && haveBoundary[BACK]) {
        for(int z = 0; z < cpuHaloWidth; ++z)
            for(int y = 0; y < localDimY+2*cpuHaloWidth; ++y)
                for(int x = 0; x < localDimX+2*cpuHaloWidth; ++x)
                    cpuToCpuSendBuffer[BACK][z*(localDimY+2*cpuHaloWidth)*(localDimX+2*cpuHaloWidth) + y*(localDimX+2*cpuHaloWidth) + x]
                            = cpuData[(z+localDimZ)*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) + y*(localDimX+2*cpuHaloWidth)+x];
        MPI_Start(&cpuToCpuSendRequest[BACK]);
    }

}

// Complete CPU-CPU exchange to the left
void Tausch3D::completeCpuEdge(Edge edge) {

    if(edge != LEFT && edge != RIGHT && edge != TOP && edge != BOTTOM && edge != FRONT && edge != BACK) {
        std::cerr << "completeCpuEdge(): ERROR: Invalid edge specified: " << edge << std::endl;
        exit(1);
    }

    if(!cpuStarted[edge]) {
        std::cerr << "ERROR: No edge #" << edge << " CPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    if(edge == LEFT && haveBoundary[LEFT]) {
        MPI_Wait(&cpuToCpuRecvRequest[LEFT], MPI_STATUS_IGNORE);
        for(int z = 0; z < localDimZ+2*cpuHaloWidth; ++z)
            for(int y = 0; y < localDimY+2*cpuHaloWidth; ++y)
                for(int hw = 0; hw < cpuHaloWidth; ++hw)
                    cpuData[z*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) + y*(localDimX+2*cpuHaloWidth)+hw]
                            = cpuToCpuRecvBuffer[LEFT][z*(localDimY+2*cpuHaloWidth)*cpuHaloWidth + y*cpuHaloWidth + hw];
        MPI_Wait(&cpuToCpuSendRequest[LEFT], MPI_STATUS_IGNORE);
    } else if(edge == RIGHT && haveBoundary[RIGHT]) {
        MPI_Wait(&cpuToCpuRecvRequest[RIGHT], MPI_STATUS_IGNORE);
        for(int z = 0; z < localDimZ+2*cpuHaloWidth; ++z)
            for(int y = 0; y < localDimY+2*cpuHaloWidth; ++y)
                for(int hw = 0; hw < cpuHaloWidth; ++hw)
                    cpuData[localDimX+cpuHaloWidth + z*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) + y*(localDimX+2*cpuHaloWidth)+hw]
                            = cpuToCpuRecvBuffer[RIGHT][z*(localDimY+2*cpuHaloWidth)*cpuHaloWidth + y*cpuHaloWidth + hw];
        MPI_Wait(&cpuToCpuSendRequest[RIGHT], MPI_STATUS_IGNORE);
    } else if(edge == TOP && haveBoundary[TOP]) {
        MPI_Wait(&cpuToCpuRecvRequest[TOP], MPI_STATUS_IGNORE);
        for(int z = 0; z < localDimZ+2*cpuHaloWidth; ++z)
            for(int y = 0; y < cpuHaloWidth; ++y)
                for(int x = 0; x < localDimX+2*cpuHaloWidth; ++x)
                    cpuData[z*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) + (y+localDimY+cpuHaloWidth)*(localDimX+2*cpuHaloWidth)+x]
                            = cpuToCpuRecvBuffer[TOP][z*(localDimX+2*cpuHaloWidth)*cpuHaloWidth + y*(localDimX+2*cpuHaloWidth) + x];
        MPI_Wait(&cpuToCpuSendRequest[TOP], MPI_STATUS_IGNORE);
    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {
        MPI_Wait(&cpuToCpuRecvRequest[BOTTOM], MPI_STATUS_IGNORE);
        for(int z = 0; z < localDimZ+2*cpuHaloWidth; ++z)
            for(int y = 0; y < cpuHaloWidth; ++y)
                for(int x = 0; x < localDimX+2*cpuHaloWidth; ++x)
                    cpuData[z*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) + y*(localDimX+2*cpuHaloWidth)+x]
                            = cpuToCpuRecvBuffer[BOTTOM][z*(localDimX+2*cpuHaloWidth)*cpuHaloWidth + y*(localDimX+2*cpuHaloWidth) + x];
        MPI_Wait(&cpuToCpuSendRequest[BOTTOM], MPI_STATUS_IGNORE);
    } else if(edge == FRONT && haveBoundary[FRONT]) {
        MPI_Wait(&cpuToCpuRecvRequest[FRONT], MPI_STATUS_IGNORE);
        for(int z = 0; z < cpuHaloWidth; ++z)
            for(int y = 0; y < localDimY+2*cpuHaloWidth; ++y)
                for(int x = 0; x < localDimX+2*cpuHaloWidth; ++x)
                    cpuData[z*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) + y*(localDimX+2*cpuHaloWidth)+x]
                            = cpuToCpuRecvBuffer[FRONT][z*(localDimY+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) + y*(localDimX+2*cpuHaloWidth) + x];
        MPI_Wait(&cpuToCpuSendRequest[FRONT], MPI_STATUS_IGNORE);
    } else if(edge == BACK && haveBoundary[BACK]) {
        MPI_Wait(&cpuToCpuRecvRequest[BACK], MPI_STATUS_IGNORE);
        for(int z = 0; z < cpuHaloWidth; ++z)
            for(int y = 0; y < localDimY+2*cpuHaloWidth; ++y)
                for(int x = 0; x < localDimX+2*cpuHaloWidth; ++x)
                    cpuData[(z+cpuHaloWidth+localDimZ)*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) + y*(localDimX+2*cpuHaloWidth)+x]
                            = cpuToCpuRecvBuffer[BACK][z*(localDimY+2*cpuHaloWidth)*(localDimX+2*cpuHaloWidth) + y*(localDimX+2*cpuHaloWidth) + x];
        MPI_Wait(&cpuToCpuSendRequest[BACK], MPI_STATUS_IGNORE);
    }

}

#ifdef TAUSCH_OPENCL

void Tausch3D::enableOpenCL(int gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {

    // gpu disabled by default, only enabled if flag is set
    gpuEnabled = true;
    // local workgroup size
    cl_kernelLocalSize = clLocalWorkgroupSize;
    this->blockingSyncCpuGpu = blockingSyncCpuGpu;
    // The CPU/GPU halo width
    this->gpuHaloWidth = gpuHaloWidth;
    // Tausch creates its own OpenCL environment
    this->setupOpenCL(giveOpenCLDeviceName);

    cl_gpuHaloWidth = cl::Buffer(cl_context, &gpuHaloWidth, (&gpuHaloWidth)+1, true);

}

// If Tausch didn't set up OpenCL, the user needs to pass some OpenCL variables
void Tausch3D::enableOpenCL(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue, int gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize) {

    this->cl_defaultDevice = cl_defaultDevice;
    this->cl_context = cl_context;
    this->cl_queue = cl_queue;
    this->blockingSyncCpuGpu = blockingSyncCpuGpu;
    this->cl_kernelLocalSize = clLocalWorkgroupSize;
    this->gpuHaloWidth = gpuHaloWidth;

    cl_gpuHaloWidth = cl::Buffer(cl_context, &gpuHaloWidth, (&gpuHaloWidth)+1, true);

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
    cpuToGpuBuffer = new std::atomic<real_t>[2*gpuHaloWidth*(gpuDimY+2*gpuHaloWidth)*(gpuDimZ+2*gpuHaloWidth) +
                                             2*gpuHaloWidth*gpuDimX*(gpuDimZ+2*gpuHaloWidth) +
                                             2*gpuHaloWidth*gpuDimX*gpuDimY]{};
    gpuToCpuBuffer = new std::atomic<real_t>[2*gpuHaloWidth*gpuDimY*gpuDimZ +
                                             2*gpuHaloWidth*(gpuDimX-2*gpuHaloWidth)*gpuDimZ +
                                             2*gpuHaloWidth*(gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth)]{};

    // set up buffers on device
    try {
        cl_cpuToGpuBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (2*gpuHaloWidth*(gpuDimY+2*gpuHaloWidth)*(gpuDimZ+2*gpuHaloWidth) +
                                                                       2*gpuHaloWidth*gpuDimX*(gpuDimZ+2*gpuHaloWidth) +
                                                                       2*gpuHaloWidth*gpuDimX*gpuDimY)*sizeof(real_t));
        cl_gpuToCpuBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (2*gpuHaloWidth*gpuDimY*gpuDimZ +
                                                                       2*gpuHaloWidth*(gpuDimX-2*gpuHaloWidth)*gpuDimZ +
                                                                       2*gpuHaloWidth*(gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth))*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_cpuToGpuBuffer, 0, 0, (2*gpuHaloWidth*(gpuDimY+2*gpuHaloWidth)*(gpuDimZ+2*gpuHaloWidth) +
                                                             2*gpuHaloWidth*gpuDimX*(gpuDimZ+2*gpuHaloWidth) +
                                                             2*gpuHaloWidth*gpuDimX*gpuDimY)*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_gpuToCpuBuffer, 0, 0, (2*gpuHaloWidth*gpuDimY*gpuDimZ +
                                                             2*gpuHaloWidth*(gpuDimX-2*gpuHaloWidth)*gpuDimZ +
                                                             2*gpuHaloWidth*(gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth))*sizeof(real_t));
        cl_gpuDimX = cl::Buffer(cl_context, &gpuDimX, (&gpuDimX)+1, true);
        cl_gpuDimY = cl::Buffer(cl_context, &gpuDimY, (&gpuDimY)+1, true);
        cl_gpuDimZ = cl::Buffer(cl_context, &gpuDimZ, (&gpuDimZ)+1, true);
    } catch(cl::Error error) {
        std::cout << "[setup send/recv buffer] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

// collect cpu side of cpu/gpu halo and store in buffer
void Tausch3D::startCpuToGpu() {

    // check whether GPU is enabled
    if(!gpuEnabled) {
        std::cerr << "ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    cpuToGpuStarted.store(true);

   // left
    for(int z = 0; z < (gpuDimZ+2*gpuHaloWidth); ++z) {
        for(int y = 0; y < (gpuDimY+2*gpuHaloWidth); ++y) {
            for(int x = 0; x < gpuHaloWidth; ++x) {
                int index = (z+(localDimZ-gpuDimZ)/2)*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) +
                            (y+(localDimY-gpuDimY)/2)*(localDimX+2*cpuHaloWidth) +
                            x+(localDimX-gpuDimX)/2;
                std::cout << x << "/" << y << "/" << z << " :: " << index << " - " << cpuData[index] << std::endl;
                cpuToGpuBuffer[z*(gpuDimY+2*gpuHaloWidth)*gpuHaloWidth + y*gpuHaloWidth + x].store(cpuData[index]);
            }
        }
    }
    int offset = (gpuDimZ+2*gpuHaloWidth)*(gpuDimY+2*gpuHaloWidth)*gpuHaloWidth;
    // right
    for(int z = 0; z < (gpuDimZ+2*gpuHaloWidth); ++z) {
        for(int y = 0; y < (gpuDimY+2*gpuHaloWidth); ++y) {
            for(int x = 0; x < gpuHaloWidth; ++x) {
                int index = (z+(localDimZ-gpuDimZ)/2)*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) +
                            (y+(localDimY-gpuDimY)/2)*(localDimX+2*cpuHaloWidth) +
                            x+(localDimX-gpuDimX)/2+gpuDimX+gpuHaloWidth;
                cpuToGpuBuffer[offset + z*(gpuDimY+2*gpuHaloWidth)*gpuHaloWidth + y*gpuHaloWidth + x].store(cpuData[index]);
            }
        }
    }
    offset += (gpuDimZ+2*gpuHaloWidth)*(gpuDimY+2*gpuHaloWidth)*gpuHaloWidth;
    // top
    for(int z = 0; z < (gpuDimZ+2*gpuHaloWidth); ++z) {
        for(int y = 0; y < gpuHaloWidth; ++y) {
            for(int x = 0; x < gpuDimX; ++x) {
                int index = (z+(localDimZ-gpuDimZ)/2)*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) +
                            (y+(localDimY-gpuDimY)/2+gpuDimY+gpuHaloWidth)*(localDimX+2*cpuHaloWidth) +
                            x+(localDimX-gpuDimX)/2+gpuHaloWidth;
                cpuToGpuBuffer[offset + z*gpuHaloWidth*gpuDimX + y*gpuDimX + x].store(cpuData[index]);
            }
        }
    }
    offset += (gpuDimZ+2*gpuHaloWidth)*gpuHaloWidth*gpuDimX;
    // bottom
    for(int z = 0; z < (gpuDimZ+2*gpuHaloWidth); ++z) {
        for(int y = 0; y < gpuHaloWidth; ++y) {
            for(int x = 0; x < gpuDimX; ++x) {
                int index = (z+(localDimZ-gpuDimZ)/2)*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) +
                            (y+(localDimY-gpuDimY)/2)*(localDimX+2*cpuHaloWidth) +
                            x+(localDimX-gpuDimX)/2+gpuHaloWidth;
                cpuToGpuBuffer[offset + z*gpuHaloWidth*gpuDimX + y*gpuDimX + x].store(cpuData[index]);
            }
        }
    }
    offset += (gpuDimZ+2*gpuHaloWidth)*gpuHaloWidth*gpuDimX;
    // front
    for(int z = 0; z < gpuHaloWidth; ++z) {
        for(int y = 0; y < gpuDimY; ++y) {
            for(int x = 0; x < gpuDimX; ++x) {
                int index = (z+(localDimZ-gpuDimZ)/2)*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) +
                            (y+(localDimY-gpuDimY)/2+gpuHaloWidth)*(localDimX+2*cpuHaloWidth) +
                            x+(localDimX-gpuDimX)/2+gpuHaloWidth;
                cpuToGpuBuffer[offset + z*gpuDimY*gpuDimX + y*gpuDimX + x].store(cpuData[index]);
            }
        }
    }
    offset += gpuHaloWidth*gpuDimY*gpuDimX;
    // back
    for(int z = 0; z < gpuHaloWidth; ++z) {
        for(int y = 0; y < gpuDimY; ++y) {
            for(int x = 0; x < gpuDimX; ++x) {
                int index = (z+(localDimZ-gpuDimZ)/2+gpuDimZ+gpuHaloWidth)*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) +
                            (y+(localDimY-gpuDimY)/2+gpuHaloWidth)*(localDimX+2*cpuHaloWidth) +
                            x+(localDimX-gpuDimX)/2+gpuHaloWidth;
                cpuToGpuBuffer[offset + z*gpuDimY*gpuDimX + y*gpuDimX + x].store(cpuData[index]);
            }
        }
    }

}

// collect gpu side of cpu/gpu halo and download into buffer
void Tausch3D::startGpuToCpu() {

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

    gpuToCpuStarted.store(true);

    try {

        auto kernel_collectHalo = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(cl_programs, "collectHaloData");

        int globalSize = ((2*gpuHaloWidth*gpuDimY*gpuDimZ
                           + 2*gpuHaloWidth*(gpuDimX-2*gpuHaloWidth)*gpuDimZ
                           + 2*gpuHaloWidth*(gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth))/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_collectHalo(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                           cl_gpuDimX, cl_gpuDimY, cl_gpuDimZ, cl_gpuHaloWidth, gpuData, cl_gpuToCpuBuffer);

        double *dat = new double[2*gpuHaloWidth*gpuDimY*gpuDimZ +
                                 2*gpuHaloWidth*(gpuDimX-2*gpuHaloWidth)*gpuDimZ +
                                 2*gpuHaloWidth*(gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth)];
        cl::copy(cl_queue, cl_gpuToCpuBuffer, &dat[0], (&dat[2*gpuHaloWidth*gpuDimY*gpuDimZ +
                                                             2*gpuHaloWidth*(gpuDimX-2*gpuHaloWidth)*gpuDimZ +
                                                             2*gpuHaloWidth*(gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth)-1])+1);

        for(int i = 0; i < 2*gpuHaloWidth*gpuDimY*gpuDimZ + 2*gpuHaloWidth*(gpuDimX-2*gpuHaloWidth)*gpuDimZ + 2*gpuHaloWidth*(gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth); ++i)
            gpuToCpuBuffer[i].store(dat[i]);

        delete[] dat;

    } catch(cl::Error error) {
        std::cout << "[kernel collectHalo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

// Complete CPU side of CPU/GPU halo exchange
void Tausch3D::completeGpuToCpu() {

    // we need to wait for the GPU thread to arrive here
    if(blockingSyncCpuGpu)
        syncCpuAndGpu();

    if(!cpuToGpuStarted.load()) {
        std::cerr << "ERROR: No CPU->GPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    // left
    for(int i = 0; i < gpuHaloWidth*gpuDimY*gpuDimZ; ++i) {
        int index = ((i/(gpuHaloWidth*gpuDimY)) + gpuHaloWidth + (localDimZ-gpuDimZ)/2)*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) +
                    ((i%(gpuHaloWidth*gpuDimY))/gpuHaloWidth + cpuHaloWidth + (localDimY-gpuDimY)/2)*(localDimX+2*cpuHaloWidth) + (localDimX-gpuDimX)/2 +
                    (i%(gpuHaloWidth*gpuDimY))%gpuHaloWidth + cpuHaloWidth;
        cpuData[index] = gpuToCpuBuffer[i].load();
    }
    int offset = gpuHaloWidth*gpuDimY*gpuDimZ;
    // right
    for(int i = 0; i < gpuHaloWidth*gpuDimY*gpuDimZ; ++i) {
        int index = ((i/(gpuHaloWidth*gpuDimY)) + cpuHaloWidth + (localDimZ-gpuDimZ)/2)*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) +
                    ((i%(gpuHaloWidth*gpuDimY))/gpuHaloWidth + cpuHaloWidth + (localDimY-gpuDimY)/2)*(localDimX+2*cpuHaloWidth) + (localDimX-gpuDimX)/2 +
                    (i%(gpuHaloWidth*gpuDimY))%gpuHaloWidth + gpuDimX;
        cpuData[index] = gpuToCpuBuffer[offset+i].load();
    }
    offset += gpuHaloWidth*gpuDimY*gpuDimZ;
    // top
    for(int i = 0; i < (gpuDimX-2*gpuHaloWidth)*gpuHaloWidth*gpuDimZ; ++i) {
        int index = ((i/((gpuDimX-2*gpuHaloWidth)*gpuHaloWidth)) + cpuHaloWidth + (localDimZ-gpuDimZ)/2)*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) +
                    ((i%((gpuDimX-2*gpuHaloWidth)*gpuHaloWidth))/(gpuDimX-2*gpuHaloWidth) + gpuDimY + (localDimY-gpuDimY)/2)*(localDimX+2*cpuHaloWidth) + (localDimX-gpuDimX)/2 +
                    (i%((gpuDimX-2*gpuHaloWidth)*gpuHaloWidth))%(gpuDimX-2*gpuHaloWidth) + gpuHaloWidth+cpuHaloWidth;
        cpuData[index] = gpuToCpuBuffer[offset+i].load();
    }
    offset += (gpuDimX-2*gpuHaloWidth)*gpuHaloWidth*gpuDimZ;
    // bottom
    for(int i = 0; i < (gpuDimX-2*gpuHaloWidth)*gpuHaloWidth*gpuDimZ; ++i) {
        int index = ((i/((gpuDimX-2*gpuHaloWidth)*gpuHaloWidth)) + cpuHaloWidth + (localDimZ-gpuDimZ)/2)*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) +
                    ((i%((gpuDimX-2*gpuHaloWidth)*gpuHaloWidth))/(gpuDimX-2*gpuHaloWidth) + cpuHaloWidth + (localDimY-gpuDimY)/2)*(localDimX+2*cpuHaloWidth) + (localDimX-gpuDimX)/2 +
                    (i%((gpuDimX-2*gpuHaloWidth)*gpuHaloWidth))%(gpuDimX-2*gpuHaloWidth) + gpuHaloWidth+cpuHaloWidth;
        cpuData[index] = gpuToCpuBuffer[offset+i].load();
    }
    offset += (gpuDimX-2*gpuHaloWidth)*gpuHaloWidth*gpuDimZ;
    // front
    for(int i = 0; i < (gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth)*gpuHaloWidth; ++i) {
        int index = ((i/((gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth))) + cpuHaloWidth + (localDimZ-gpuDimZ)/2)*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) +
                    ((i%((gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth)))/(gpuDimX-2*gpuHaloWidth) + gpuHaloWidth+cpuHaloWidth + (localDimY-gpuDimY)/2)*(localDimX+2*cpuHaloWidth) + (localDimX-gpuDimX)/2 +
                    (i%((gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth)))%(gpuDimX-2*gpuHaloWidth) + gpuHaloWidth+cpuHaloWidth;
        cpuData[index] = gpuToCpuBuffer[offset+i].load();
    }
    offset += (gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth)*gpuHaloWidth;
    // back
    for(int i = 0; i < (gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth)*gpuHaloWidth; ++i) {
        int index = ((i/((gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth))) + gpuDimZ + (localDimZ-gpuDimZ)/2)*(localDimX+2*cpuHaloWidth)*(localDimY+2*cpuHaloWidth) +
                    ((i%((gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth)))/(gpuDimX-2*gpuHaloWidth) + gpuHaloWidth+cpuHaloWidth + (localDimY-gpuDimY)/2)*(localDimX+2*cpuHaloWidth) + (localDimX-gpuDimX)/2 +
                    (i%((gpuDimX-2*gpuHaloWidth)*(gpuDimY-2*gpuHaloWidth)))%(gpuDimX-2*gpuHaloWidth) + gpuHaloWidth+cpuHaloWidth;
        cpuData[index] = gpuToCpuBuffer[offset+i].load();
    }

}

// Complete GPU side of CPU/GPU halo exchange
void Tausch3D::completeCpuToGpu() {

    // we need to wait for the CPU thread to arrive here
    if(blockingSyncCpuGpu)
        syncCpuAndGpu();

    if(!gpuToCpuStarted.load()) {
        std::cerr << "ERROR: No GPU->CPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    try {

        double *dat = new double[2*gpuHaloWidth*(gpuDimY+2*gpuHaloWidth)*(gpuDimZ+2*gpuHaloWidth) + 2*gpuHaloWidth*gpuDimX*(gpuDimZ+2*gpuHaloWidth) + 2*gpuHaloWidth*gpuDimX*gpuDimY];
        for(int i = 0; i < 2*gpuHaloWidth*(gpuDimY+2*gpuHaloWidth)*(gpuDimZ+2*gpuHaloWidth) + 2*gpuHaloWidth*gpuDimX*(gpuDimZ+2*gpuHaloWidth) + 2*gpuHaloWidth*gpuDimX*gpuDimY; ++i)
            dat[i] = cpuToGpuBuffer[i].load();

        cl::copy(cl_queue, &dat[0], (&dat[2*gpuHaloWidth*(gpuDimY+2*gpuHaloWidth)*(gpuDimZ+2*gpuHaloWidth) + 2*gpuHaloWidth*gpuDimX*(gpuDimZ+2*gpuHaloWidth) + 2*gpuHaloWidth*gpuDimX*gpuDimY-1])+1, cl_cpuToGpuBuffer);

        delete[] dat;

        auto kernel_distributeHaloData = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(cl_programs, "distributeHaloData");

        int globalSize = ((2*gpuHaloWidth*(gpuDimY+2*gpuHaloWidth)*(gpuDimZ+2*gpuHaloWidth) + 2*gpuHaloWidth*gpuDimX*(gpuDimZ+2*gpuHaloWidth) + 2*gpuHaloWidth*gpuDimX*gpuDimY)/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_distributeHaloData(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                                  cl_gpuDimX, cl_gpuDimY, cl_gpuDimZ, cl_gpuHaloWidth, gpuData, cl_cpuToGpuBuffer);

    } catch(cl::Error error) {
        std::cout << "[dist halo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }


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

    // Tausch requires two kernels: One for collecting the halo data and one for distributing that data
    std::string oclstr = "typedef " + std::string((sizeof(real_t)==sizeof(double)) ? "double" : "float") + " real_t;\n";

    oclstr += R"d(
kernel void collectHaloData(global const int * restrict const dimX, global const int * restrict const dimY,
                            global const int * restrict const dimZ, global const int * restrict const haloWidth,
                            global const real_t * restrict const vec, global real_t * sync) {

    unsigned int current = get_global_id(0);
    unsigned int maxNum = 2*(*haloWidth)*(*dimY)*(*dimZ) +
                          2*(*haloWidth)*(*dimX-2*(*haloWidth))*(*dimZ) +
                          2*(*haloWidth)*(*dimX-2*(*haloWidth))*((*dimY)-2*(*haloWidth));

    if(current >= maxNum)
        return;

    // left
    if(current < (*haloWidth)*(*dimY)*(*dimZ)) {
        int index = (*haloWidth + current/((*haloWidth)*(*dimY))) * ((*dimX+2*(*haloWidth))*(*dimY+2*(*haloWidth))) +
                    (*haloWidth + (current%((*haloWidth)*(*dimY)))/(*haloWidth)) * (*dimX+2*(*haloWidth)) + current%(*haloWidth) + *haloWidth;
        sync[current] = vec[index];
        return;
    }
    int offset = (*haloWidth)*(*dimY)*(*dimZ);
    // right
    if(current < offset+(*haloWidth)*(*dimY)*(*dimZ)) {
        current -= offset;
        int index = (*haloWidth + current/((*haloWidth)*(*dimY))) * ((*dimX+2*(*haloWidth))*(*dimY+2*(*haloWidth))) +
                    (*haloWidth + (current%((*haloWidth)*(*dimY)))/(*haloWidth)) * (*dimX+2*(*haloWidth)) + current%(*haloWidth) + *dimX;
        sync[offset+current] = vec[index];
        return;
    }
    // top
    offset += (*haloWidth)*(*dimY)*(*dimZ);
    if(current < offset + (*dimX - 2*(*haloWidth))*(*haloWidth)*(*dimZ)) {
        current -= offset;
        int index = (*haloWidth + current/((*dimX - 2*(*haloWidth))*(*haloWidth))) * ((*dimX+2*(*haloWidth))*(*dimY+2*(*haloWidth))) +
                    ((current%((*dimX - 2*(*haloWidth))*(*haloWidth)))/(*dimX - 2*(*haloWidth)) + *dimY) * (*dimX+2*(*haloWidth)) +
                    current%(*dimX - 2*(*haloWidth)) + 2*(*haloWidth);
        sync[offset+current] = vec[index];
        return;
    }
    // bottom
    offset += (*dimX - 2*(*haloWidth))*(*haloWidth)*(*dimZ);
    if(current < offset + (*dimX - 2*(*haloWidth))*(*haloWidth)*(*dimZ)) {
        current -= offset;
        int index = ((*haloWidth) + current/((*dimX - 2*(*haloWidth))*(*haloWidth))) * ((*dimX+2*(*haloWidth))*(*dimY+2*(*haloWidth))) +
                    ((current%((*dimX - 2*(*haloWidth))*(*haloWidth)))/(*dimX - 2*(*haloWidth)) + *haloWidth) * (*dimX+2*(*haloWidth)) +
                    current%(*dimX - 2*(*haloWidth)) + 2*(*haloWidth);
        sync[offset+current] = vec[index];
        return;
    }
    // front
    offset += (*dimX - 2*(*haloWidth))*(*haloWidth)*(*dimZ);
    if(current < offset + (*dimX - 2*(*haloWidth))*(*dimY - 2*(*haloWidth))*(*haloWidth)) {
        current -= offset;
        int index = ((*haloWidth) + current/((*dimX - 2*(*haloWidth))*(*dimY - 2*(*haloWidth)))) * ((*dimX+2*(*haloWidth))*(*dimY+2*(*haloWidth))) +
                    ((*haloWidth) + (current%((*dimX - 2*(*haloWidth))*(*dimY - 2*(*haloWidth))))/(*dimX - 2*(*haloWidth)) + *haloWidth) * (*dimX+2*(*haloWidth)) +
                    current%(*dimX - 2*(*haloWidth)) + 2*(*haloWidth);
        sync[offset+current] = vec[index];
        return;
    }
    // back
    offset += (*dimX - 2*(*haloWidth))*(*dimY - 2*(*haloWidth))*(*haloWidth);
    if(current < offset + (*dimX - 2*(*haloWidth))*(*dimY - 2*(*haloWidth))*(*haloWidth)) {
        current -= offset;
        int index = ((*dimZ) + current/((*dimX - 2*(*haloWidth))*(*dimY - 2*(*haloWidth)))) * ((*dimX+2*(*haloWidth))*(*dimY+2*(*haloWidth))) +
                    ((*haloWidth) + (current%((*dimX - 2*(*haloWidth))*(*dimY - 2*(*haloWidth))))/(*dimX - 2*(*haloWidth)) + *haloWidth) * (*dimX+2*(*haloWidth)) +
                    current%(*dimX - 2*(*haloWidth)) + 2*(*haloWidth);
        sync[offset+current] = vec[index];
        return;
    }
}

kernel void distributeHaloData(global const int * restrict const dimX, global const int * restrict const dimY,
                               global const int * restrict const dimZ, global const int * restrict const haloWidth,
                               global real_t * vec, global const real_t * restrict const sync) {

    unsigned int current = get_global_id(0);
    int maxNum = 2*(*haloWidth)*(*dimY+2*(*haloWidth))*(*dimZ+2*(*haloWidth)) + 2*(*haloWidth)*(*dimX)*(*dimZ+2*(*haloWidth)) + 2*(*haloWidth)*(*dimX)*(*dimY);

    if(current >= maxNum)
        return;

    // left
    if(current < (*haloWidth)*(*dimY+2*(*haloWidth))*(*dimZ+2*(*haloWidth))) {
        int index = ( current/(*haloWidth*(*dimY+2*(*haloWidth))) ) * (*dimX+2*(*haloWidth)) * (*dimY+2*(*haloWidth)) +
                    ( (current%(*haloWidth*(*dimY+2*(*haloWidth)))) / (*haloWidth) ) * (*dimX+2*(*haloWidth)) +
                    current%(*haloWidth);
        vec[index] = sync[current];
        return;
    }
    int offset = (*haloWidth)*(*dimY+2*(*haloWidth))*(*dimZ+2*(*haloWidth));
    // right
    if(current < offset + (*haloWidth)*(*dimY+2*(*haloWidth))*(*dimZ+2*(*haloWidth))) {
        current -= offset;
        int index = ( current/(*haloWidth*(*dimY+2*(*haloWidth))) ) * (*dimX+2*(*haloWidth)) * (*dimY+2*(*haloWidth)) +
                    ( (current%(*haloWidth*(*dimY+2*(*haloWidth)))) / (*haloWidth) ) * (*dimX+2*(*haloWidth)) +
                    current%(*haloWidth) + *dimX+*haloWidth;
        vec[index] = sync[offset+current];
        return;
    }
    // top
    offset += (*haloWidth)*(*dimY+2*(*haloWidth))*(*dimZ+2*(*haloWidth));
    if(current < offset + (*haloWidth)*(*dimX)*(*dimZ+2*(*haloWidth))) {
        current -= offset;
        int index = ( current/(*haloWidth*(*dimX))) * (*dimX+2*(*haloWidth)) * (*dimY+2*(*haloWidth)) +
                    ( (current%(*haloWidth*(*dimX))) / (*dimX)  + *dimY+*haloWidth) * (*dimX+2*(*haloWidth)) +
                    current%(*dimX) + *haloWidth;
        vec[index] = sync[offset+current];
        return;
    }
    // bottom
    offset += (*haloWidth)*(*dimX)*(*dimZ+2*(*haloWidth));
    if(current < offset + (*haloWidth)*(*dimX)*(*dimZ+2*(*haloWidth))) {
        current -= offset;
        int index = ( current/(*haloWidth*(*dimX))) * (*dimX+2*(*haloWidth)) * (*dimY+2*(*haloWidth)) +
                    ( (current%(*haloWidth*(*dimX))) / (*dimX) ) * (*dimX+2*(*haloWidth)) +
                    current%(*dimX) + *haloWidth;
        vec[index] = sync[offset+current];
        return;
    }
    // front
    offset += (*haloWidth)*(*dimX)*(*dimZ+2*(*haloWidth));
    if(current < offset + (*haloWidth)*(*dimX)*(*dimY)) {
        current -= offset;
        int index = ( current/((*dimX)*(*dimY))) * (*dimX+2*(*haloWidth)) * (*dimY+2*(*haloWidth)) +
                    ( (current%((*dimX)*(*dimY))) / (*dimX) + *haloWidth) * (*dimX+2*(*haloWidth)) +
                    current%(*dimX) + *haloWidth;
        vec[index] = sync[offset+current];
        return;
    }
    // back
    offset += (*haloWidth)*(*dimX)*(*dimY);
    current -= offset;
    int index = ( current/((*dimX)*(*dimY)) + (*dimZ) + (*haloWidth)) * (*dimX+2*(*haloWidth)) * (*dimY+2*(*haloWidth)) +
                ( (current%((*dimX)*(*dimY))) / (*dimX) + *haloWidth) * (*dimX+2*(*haloWidth)) +
                current%(*dimX) + *haloWidth;
    vec[index] = sync[offset+current];

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
