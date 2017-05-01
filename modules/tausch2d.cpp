#include "tausch2d.h"

Tausch2D::Tausch2D(int localDimX, int localDimY, int mpiNumX, int mpiNumY, int cpuHaloWidth, MPI_Comm comm) {

    MPI_Comm_dup(comm, &TAUSCH_COMM);

    // get MPI info
    MPI_Comm_rank(TAUSCH_COMM, &mpiRank);
    MPI_Comm_size(TAUSCH_COMM, &mpiSize);

    this->mpiNumX = mpiNumX;
    this->mpiNumY = mpiNumY;

    // store configuration
    this->localDimX = localDimX;
    this->localDimY = localDimY;

    this->cpuHaloWidth = cpuHaloWidth;

    // check if this rank has a boundary with another rank
    haveBoundary[LEFT] = (mpiRank%mpiNumX != 0);
    haveBoundary[RIGHT] = ((mpiRank+1)%mpiNumX != 0);
    haveBoundary[TOP] = (mpiRank < mpiSize-mpiNumX);
    haveBoundary[BOTTOM] = (mpiRank > mpiNumX-1);

    // whether the cpu/gpu pointers have been passed
    cpuInfoGiven = false;

    stencilInfoGiven = false;

    // cpu at beginning
    cpuRecvsPosted = false;

    stencilRecvsPosted = false;

    // communication to neither edge has been started
    cpuStarted[LEFT] = false;
    cpuStarted[RIGHT] = false;
    cpuStarted[TOP] = false;
    cpuStarted[BOTTOM] = false;

    cpuStencilStarted[LEFT] = false;
    cpuStencilStarted[RIGHT] = false;
    cpuStencilStarted[TOP] = false;
    cpuStencilStarted[BOTTOM] = false;

    mpiDataType = ((sizeof(real_t) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT);

#ifdef TAUSCH_OPENCL

    gpuDataInfoGiven = false;
    gpuStencilInfoGiven = false;
    gpuEnabled = false;

    gpuToCpuDataStarted = false;
    cpuToGpuDataStarted = false;

    // used for syncing the CPU and GPU thread
    sync_counter[0].store(0);
    sync_counter[1].store(0);
    sync_lock[0].store(0);
    sync_lock[1].store(0);

#endif

}

Tausch2D::~Tausch2D() {
    // clean up memory
    for(int i = 0; i < 4; ++i) {
        delete[] cpuToCpuSendBuffer[i];
        delete[] cpuToCpuRecvBuffer[i];
    }
    delete[] cpuToCpuSendBuffer;
    delete[] cpuToCpuRecvBuffer;
#ifdef TAUSCH_OPENCL
    if(gpuEnabled) {
        delete[] cpuToGpuDataBuffer;
        delete[] gpuToCpuDataBuffer;
        delete[] cpuToGpuStencilBuffer;
        delete[] gpuToCpuStencilBuffer;
    }
#endif
}

// get a pointer to the CPU data
void Tausch2D::setCPUData(real_t *data) {

    cpuInfoGiven = true;
    cpuData = data;

    // a send and recv buffer for the CPU-CPU communication
    cpuToCpuSendBuffer = new real_t*[4];
    cpuToCpuSendBuffer[LEFT] = new real_t[cpuHaloWidth*(localDimY+2*cpuHaloWidth)]{};
    cpuToCpuSendBuffer[RIGHT] = new real_t[cpuHaloWidth*(localDimY+2*cpuHaloWidth)]{};
    cpuToCpuSendBuffer[TOP] = new real_t[cpuHaloWidth*(localDimX+2*cpuHaloWidth)]{};
    cpuToCpuSendBuffer[BOTTOM] = new real_t[cpuHaloWidth*(localDimX+2*cpuHaloWidth)]{};
    cpuToCpuRecvBuffer = new real_t*[4];
    cpuToCpuRecvBuffer[LEFT] = new real_t[cpuHaloWidth*(localDimY+2*cpuHaloWidth)]{};
    cpuToCpuRecvBuffer[RIGHT] = new real_t[cpuHaloWidth*(localDimY+2*cpuHaloWidth)]{};
    cpuToCpuRecvBuffer[TOP] = new real_t[cpuHaloWidth*(localDimX+2*cpuHaloWidth)]{};
    cpuToCpuRecvBuffer[BOTTOM] = new real_t[cpuHaloWidth*(localDimX+2*cpuHaloWidth)]{};

    if(haveBoundary[LEFT]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[LEFT], cpuHaloWidth*(localDimY+2*cpuHaloWidth),
                      mpiDataType, mpiRank-1, 0, TAUSCH_COMM, &cpuToCpuRecvRequest[LEFT]);
        MPI_Send_init(cpuToCpuSendBuffer[LEFT], cpuHaloWidth*(localDimY+2*cpuHaloWidth),
                      mpiDataType, mpiRank-1, 2, TAUSCH_COMM, &cpuToCpuSendRequest[LEFT]);
    }
    if(haveBoundary[RIGHT]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[RIGHT], cpuHaloWidth*(localDimY+2*cpuHaloWidth),
                      mpiDataType, mpiRank+1, 2, TAUSCH_COMM, &cpuToCpuRecvRequest[RIGHT]);
        MPI_Send_init(cpuToCpuSendBuffer[RIGHT], cpuHaloWidth*(localDimY+2*cpuHaloWidth),
                      mpiDataType, mpiRank+1, 0, TAUSCH_COMM, &cpuToCpuSendRequest[RIGHT]);
    }
    if(haveBoundary[TOP]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[TOP], cpuHaloWidth*(localDimX+2*cpuHaloWidth),
                      mpiDataType, mpiRank+mpiNumX, 1, TAUSCH_COMM, &cpuToCpuRecvRequest[TOP]);
        MPI_Send_init(cpuToCpuSendBuffer[TOP], cpuHaloWidth*(localDimX+2*cpuHaloWidth),
                      mpiDataType, mpiRank+mpiNumX, 3, TAUSCH_COMM, &cpuToCpuSendRequest[TOP]);
    }
    if(haveBoundary[BOTTOM]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[BOTTOM], cpuHaloWidth*(localDimX+2*cpuHaloWidth),
                      mpiDataType, mpiRank-mpiNumX, 3, TAUSCH_COMM, &cpuToCpuRecvRequest[BOTTOM]);
        MPI_Send_init(cpuToCpuSendBuffer[BOTTOM], cpuHaloWidth*(localDimX+2*cpuHaloWidth),
                      mpiDataType, mpiRank-mpiNumX, 1, TAUSCH_COMM, &cpuToCpuSendRequest[BOTTOM]);
    }

}

void Tausch2D::setCPUStencil(real_t *stencil, int stencilNumPoints) {

    stencilInfoGiven = true;
    cpuStencil = stencil;
    this->stencilNumPoints = stencilNumPoints;

    cpuToCpuStencilSendBuffer = new real_t*[4];
    cpuToCpuStencilSendBuffer[LEFT] = new real_t[stencilNumPoints*cpuHaloWidth*(localDimY+2*cpuHaloWidth)]{};
    cpuToCpuStencilSendBuffer[RIGHT] = new real_t[stencilNumPoints*cpuHaloWidth*(localDimY+2*cpuHaloWidth)]{};
    cpuToCpuStencilSendBuffer[TOP] = new real_t[stencilNumPoints*cpuHaloWidth*(localDimX+2*cpuHaloWidth)]{};
    cpuToCpuStencilSendBuffer[BOTTOM] = new real_t[stencilNumPoints*cpuHaloWidth*(localDimX+2*cpuHaloWidth)]{};
    cpuToCpuStencilRecvBuffer = new real_t*[4];
    cpuToCpuStencilRecvBuffer[LEFT] = new real_t[stencilNumPoints*cpuHaloWidth*(localDimY+2*cpuHaloWidth)]{};
    cpuToCpuStencilRecvBuffer[RIGHT] = new real_t[stencilNumPoints*cpuHaloWidth*(localDimY+2*cpuHaloWidth)]{};
    cpuToCpuStencilRecvBuffer[TOP] = new real_t[stencilNumPoints*cpuHaloWidth*(localDimX+2*cpuHaloWidth)]{};
    cpuToCpuStencilRecvBuffer[BOTTOM] = new real_t[stencilNumPoints*cpuHaloWidth*(localDimX+2*cpuHaloWidth)]{};

    if(haveBoundary[LEFT]) {
        MPI_Recv_init(cpuToCpuStencilRecvBuffer[LEFT], stencilNumPoints*cpuHaloWidth*(localDimY+2*cpuHaloWidth),
                      mpiDataType, mpiRank-1, 0, TAUSCH_COMM, &cpuToCpuStencilRecvRequest[LEFT]);
        MPI_Send_init(cpuToCpuStencilSendBuffer[LEFT], stencilNumPoints*cpuHaloWidth*(localDimY+2*cpuHaloWidth),
                      mpiDataType, mpiRank-1, 2, TAUSCH_COMM, &cpuToCpuStencilSendRequest[LEFT]);
    }
    if(haveBoundary[RIGHT]) {
        MPI_Recv_init(cpuToCpuStencilRecvBuffer[RIGHT], stencilNumPoints*cpuHaloWidth*(localDimY+2*cpuHaloWidth),
                      mpiDataType, mpiRank+1, 2, TAUSCH_COMM, &cpuToCpuStencilRecvRequest[RIGHT]);
        MPI_Send_init(cpuToCpuStencilSendBuffer[RIGHT], stencilNumPoints*cpuHaloWidth*(localDimY+2*cpuHaloWidth),
                      mpiDataType, mpiRank+1, 0, TAUSCH_COMM, &cpuToCpuStencilSendRequest[RIGHT]);
    }
    if(haveBoundary[TOP]) {
        MPI_Recv_init(cpuToCpuStencilRecvBuffer[TOP], stencilNumPoints*cpuHaloWidth*(localDimX+2*cpuHaloWidth),
                      mpiDataType, mpiRank+mpiNumX, 1, TAUSCH_COMM, &cpuToCpuStencilRecvRequest[TOP]);
        MPI_Send_init(cpuToCpuStencilSendBuffer[TOP], stencilNumPoints*cpuHaloWidth*(localDimX+2*cpuHaloWidth),
                      mpiDataType, mpiRank+mpiNumX, 3, TAUSCH_COMM, &cpuToCpuStencilSendRequest[TOP]);
    }
    if(haveBoundary[BOTTOM]) {
        MPI_Recv_init(cpuToCpuStencilRecvBuffer[BOTTOM], stencilNumPoints*cpuHaloWidth*(localDimX+2*cpuHaloWidth),
                      mpiDataType, mpiRank-mpiNumX, 3, TAUSCH_COMM, &cpuToCpuStencilRecvRequest[BOTTOM]);
        MPI_Send_init(cpuToCpuStencilSendBuffer[BOTTOM], stencilNumPoints*cpuHaloWidth*(localDimX+2*cpuHaloWidth),
                      mpiDataType, mpiRank-mpiNumX, 1, TAUSCH_COMM, &cpuToCpuStencilSendRequest[BOTTOM]);
    }

}

// post the MPI_Irecv's for inter-rank communication
void Tausch2D::postCpuDataReceives() {

    if(!cpuInfoGiven) {
        std::cerr << "Tausch2D :: ERROR: You didn't tell me yet where to find the data! Abort..." << std::endl;
        exit(1);
    }

    cpuRecvsPosted = true;

    if(haveBoundary[LEFT])
        MPI_Start(&cpuToCpuRecvRequest[LEFT]);
    if(haveBoundary[RIGHT])
        MPI_Start(&cpuToCpuRecvRequest[RIGHT]);
    if(haveBoundary[TOP])
        MPI_Start(&cpuToCpuRecvRequest[TOP]);
    if(haveBoundary[BOTTOM])
        MPI_Start(&cpuToCpuRecvRequest[BOTTOM]);

}

// post the MPI_Irecv's for inter-rank communication
void Tausch2D::postCpuStencilReceives() {

    if(!cpuInfoGiven) {
        std::cerr << "Tausch2D :: ERROR: You didn't tell me yet where to find the data! Abort..." << std::endl;
        exit(1);
    }

    stencilRecvsPosted = true;

    if(haveBoundary[LEFT])
        MPI_Start(&cpuToCpuStencilRecvRequest[LEFT]);
    if(haveBoundary[RIGHT])
        MPI_Start(&cpuToCpuStencilRecvRequest[RIGHT]);
    if(haveBoundary[TOP])
        MPI_Start(&cpuToCpuStencilRecvRequest[TOP]);
    if(haveBoundary[BOTTOM])
        MPI_Start(&cpuToCpuStencilRecvRequest[BOTTOM]);

}

void Tausch2D::startCpuDataEdge(Edge edge) {

    if(!cpuRecvsPosted) {
        std::cerr << "Tausch2D :: ERROR: No CPU Recvs have been posted yet... Abort!" << std::endl;
        exit(1);
    }

    if(edge != LEFT && edge != RIGHT && edge != TOP && edge != BOTTOM) {
        std::cerr << "Tausch2D :: startCpuDataEdge(): ERROR: Invalid edge specified: " << edge << std::endl;
        exit(1);
    }

    cpuStarted[edge] = true;

    MPI_Datatype mpiDataType = ((sizeof(real_t) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT);

    if(edge == LEFT && haveBoundary[LEFT]) {
        for(int i = 0; i < cpuHaloWidth*(localDimY+2*cpuHaloWidth); ++i)
            cpuToCpuSendBuffer[LEFT][i] = cpuData[cpuHaloWidth+ (i/cpuHaloWidth)*(localDimX+2*cpuHaloWidth)+i%cpuHaloWidth];
        MPI_Start(&cpuToCpuSendRequest[LEFT]);
    } else if(edge == RIGHT && haveBoundary[RIGHT]) {
        for(int i = 0; i < cpuHaloWidth*(localDimY+2*cpuHaloWidth); ++i)
            cpuToCpuSendBuffer[RIGHT][i] = cpuData[(i/cpuHaloWidth+1)*(localDimX+2*cpuHaloWidth) -(2*cpuHaloWidth)+i%cpuHaloWidth];
        MPI_Start(&cpuToCpuSendRequest[RIGHT]);
    } else if(edge == TOP && haveBoundary[TOP]) {
        for(int i = 0; i < cpuHaloWidth*(localDimX+2*cpuHaloWidth); ++i)
            cpuToCpuSendBuffer[TOP][i] = cpuData[(localDimX+2*cpuHaloWidth)*(localDimY) + i];
        MPI_Start(&cpuToCpuSendRequest[TOP]);
    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {
        for(int i = 0; i < cpuHaloWidth*(localDimX+2*cpuHaloWidth); ++i)
            cpuToCpuSendBuffer[BOTTOM][i] = cpuData[cpuHaloWidth*(localDimX+2*cpuHaloWidth) + i];
        MPI_Start(&cpuToCpuSendRequest[BOTTOM]);
    }

}

void Tausch2D::startCpuStencilEdge(Edge edge) {

    if(!stencilRecvsPosted) {
        std::cerr << "Tausch2D :: ERROR: No CPU stencil recvs have been posted yet... Abort!" << std::endl;
        exit(1);
    }

    if(edge != LEFT && edge != RIGHT && edge != TOP && edge != BOTTOM) {
        std::cerr << "Tausch2D :: startCpuStencilEdge(): ERROR: Invalid edge specified: " << edge << std::endl;
        exit(1);
    }

    cpuStencilStarted[edge] = true;

    if(edge == LEFT && haveBoundary[LEFT]) {
        for(int i = 0; i < cpuHaloWidth*(localDimY+2*cpuHaloWidth); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuToCpuStencilSendBuffer[LEFT][stencilNumPoints*i + j] = cpuStencil[stencilNumPoints*(cpuHaloWidth+ (i/cpuHaloWidth)*(localDimX+2*cpuHaloWidth)+i%cpuHaloWidth) + j];
        MPI_Start(&cpuToCpuStencilSendRequest[LEFT]);
    } else if(edge == RIGHT && haveBoundary[RIGHT]) {
        for(int i = 0; i < cpuHaloWidth*(localDimY+2*cpuHaloWidth); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuToCpuStencilSendBuffer[RIGHT][stencilNumPoints*i + j] = cpuStencil[stencilNumPoints*((i/cpuHaloWidth+1)*(localDimX+2*cpuHaloWidth) -(2*cpuHaloWidth)+i%cpuHaloWidth) + j];
        MPI_Start(&cpuToCpuStencilSendRequest[RIGHT]);
    } else if(edge == TOP && haveBoundary[TOP]) {
        for(int i = 0; i < cpuHaloWidth*(localDimX+2*cpuHaloWidth); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuToCpuStencilSendBuffer[TOP][stencilNumPoints*i + j] = cpuStencil[stencilNumPoints*((localDimX+2*cpuHaloWidth)*(localDimY) + i) + j];
        MPI_Start(&cpuToCpuStencilSendRequest[TOP]);
    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {
        for(int i = 0; i < cpuHaloWidth*(localDimX+2*cpuHaloWidth); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuToCpuStencilSendBuffer[BOTTOM][stencilNumPoints*i + j] = cpuStencil[stencilNumPoints*(cpuHaloWidth*(localDimX+2*cpuHaloWidth) + i) + j];
        MPI_Start(&cpuToCpuStencilSendRequest[BOTTOM]);
    }

}

// Complete CPU-CPU exchange to the left
void Tausch2D::completeCpuDataEdge(Edge edge) {

    if(edge != LEFT && edge != RIGHT && edge != TOP && edge != BOTTOM) {
        std::cerr << "Tausch2D :: completeCpuDataEdge(): ERROR: Invalid edge specified: " << edge << std::endl;
        exit(1);
    }

    if(!cpuStarted[edge]) {
        std::cerr << "Tausch2D :: ERROR: No edge #" << edge << " CPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    if(edge == LEFT && haveBoundary[LEFT]) {
        MPI_Wait(&cpuToCpuRecvRequest[LEFT], MPI_STATUS_IGNORE);
        for(int i = 0; i < cpuHaloWidth*(localDimY+2*cpuHaloWidth); ++i)
            cpuData[(i/cpuHaloWidth)*(localDimX+2*cpuHaloWidth)+i%cpuHaloWidth] = cpuToCpuRecvBuffer[LEFT][i];
        MPI_Wait(&cpuToCpuSendRequest[LEFT], MPI_STATUS_IGNORE);
    } else if(edge == RIGHT && haveBoundary[RIGHT]) {
        MPI_Wait(&cpuToCpuRecvRequest[RIGHT], MPI_STATUS_IGNORE);
        for(int i = 0; i < cpuHaloWidth*(localDimY+2*cpuHaloWidth); ++i)
            cpuData[(i/cpuHaloWidth+1)*(localDimX+2*cpuHaloWidth)-cpuHaloWidth+i%cpuHaloWidth] = cpuToCpuRecvBuffer[RIGHT][i];
        MPI_Wait(&cpuToCpuSendRequest[RIGHT], MPI_STATUS_IGNORE);
    } else if(edge == TOP && haveBoundary[TOP]) {
        MPI_Wait(&cpuToCpuRecvRequest[TOP], MPI_STATUS_IGNORE);
        for(int i = 0; i < cpuHaloWidth*(localDimX+2*cpuHaloWidth); ++i)
            cpuData[(localDimX+2*cpuHaloWidth)*(localDimY+cpuHaloWidth) + i] = cpuToCpuRecvBuffer[TOP][i];
        MPI_Wait(&cpuToCpuSendRequest[TOP], MPI_STATUS_IGNORE);
    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {
        MPI_Wait(&cpuToCpuRecvRequest[BOTTOM], MPI_STATUS_IGNORE);
        for(int i = 0; i < cpuHaloWidth*(localDimX+2*cpuHaloWidth); ++i)
            cpuData[i] = cpuToCpuRecvBuffer[BOTTOM][i];
        MPI_Wait(&cpuToCpuSendRequest[BOTTOM], MPI_STATUS_IGNORE);
    }

}

// Complete CPU-CPU exchange to the left
void Tausch2D::completeCpuStencilEdge(Edge edge) {

    if(edge != LEFT && edge != RIGHT && edge != TOP && edge != BOTTOM) {
        std::cerr << "Tausch2D :: completeCpuStencilEdge(): ERROR: Invalid edge specified: " << edge << std::endl;
        exit(1);
    }

    if(!cpuStencilStarted[edge]) {
        std::cerr << "Tausch2D :: ERROR: No edge #" << edge << " CPU stencil exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    if(edge == LEFT && haveBoundary[LEFT]) {
        MPI_Wait(&cpuToCpuStencilRecvRequest[LEFT], MPI_STATUS_IGNORE);
        for(int i = 0; i < cpuHaloWidth*(localDimY+2*cpuHaloWidth); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuStencil[stencilNumPoints*((i/cpuHaloWidth)*(localDimX+2*cpuHaloWidth)+i%cpuHaloWidth) + j] = cpuToCpuStencilRecvBuffer[LEFT][stencilNumPoints*i + j];
        MPI_Wait(&cpuToCpuStencilSendRequest[LEFT], MPI_STATUS_IGNORE);
    } else if(edge == RIGHT && haveBoundary[RIGHT]) {
        MPI_Wait(&cpuToCpuStencilRecvRequest[RIGHT], MPI_STATUS_IGNORE);
        for(int i = 0; i < cpuHaloWidth*(localDimY+2*cpuHaloWidth); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuStencil[stencilNumPoints*((i/cpuHaloWidth+1)*(localDimX+2*cpuHaloWidth)-cpuHaloWidth+i%cpuHaloWidth) + j] = cpuToCpuStencilRecvBuffer[RIGHT][stencilNumPoints*i + j];
        MPI_Wait(&cpuToCpuStencilSendRequest[RIGHT], MPI_STATUS_IGNORE);
    } else if(edge == TOP && haveBoundary[TOP]) {
        MPI_Wait(&cpuToCpuStencilRecvRequest[TOP], MPI_STATUS_IGNORE);
        for(int i = 0; i < cpuHaloWidth*(localDimX+2*cpuHaloWidth); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuStencil[stencilNumPoints*((localDimX+2*cpuHaloWidth)*(localDimY+cpuHaloWidth) + i) + j] = cpuToCpuStencilRecvBuffer[TOP][stencilNumPoints*i + j];
        MPI_Wait(&cpuToCpuStencilSendRequest[TOP], MPI_STATUS_IGNORE);
    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {
        MPI_Wait(&cpuToCpuStencilRecvRequest[BOTTOM], MPI_STATUS_IGNORE);
        for(int i = 0; i < cpuHaloWidth*(localDimX+2*cpuHaloWidth); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuStencil[stencilNumPoints*i + j] = cpuToCpuStencilRecvBuffer[BOTTOM][stencilNumPoints*i + j];
        MPI_Wait(&cpuToCpuStencilSendRequest[BOTTOM], MPI_STATUS_IGNORE);
    }

}

#ifdef TAUSCH_OPENCL

void Tausch2D::enableOpenCL(int gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {

    // gpu disabled by default, only enabled if flag is set
    gpuEnabled = true;
    // local workgroup size
    cl_kernelLocalSize = clLocalWorkgroupSize;
    this->blockingSyncCpuGpu = blockingSyncCpuGpu;
    // the GPU halo width
    this->gpuHaloWidth = gpuHaloWidth;
    // Tausch creates its own OpenCL environment
    this->setupOpenCL(giveOpenCLDeviceName);

    cl_gpuHaloWidth = cl::Buffer(cl_context, &gpuHaloWidth, (&gpuHaloWidth)+1, true);

}

// If Tausch didn't set up OpenCL, the user needs to pass some OpenCL variables
void Tausch2D::enableOpenCL(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue, int gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize) {

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
void Tausch2D::setGPUData(cl::Buffer &dat, int gpuDimX, int gpuDimY) {

    // check whether OpenCL has been set up
    if(!gpuEnabled) {
        std::cerr << "Tausch2D :: ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    gpuDataInfoGiven = true;

    // store parameters
    gpuData = dat;
    this->gpuDimX = gpuDimX;
    this->gpuDimY = gpuDimY;

    // store buffer to store the GPU and the CPU part of the halo.
    // We do not need two buffers each, as each thread has direct access to both arrays, no MPI communication necessary
    cpuToGpuDataBuffer = new std::atomic<real_t>[2*gpuHaloWidth*(gpuDimX+2*gpuHaloWidth) + 2*gpuHaloWidth*gpuDimY]{};
    gpuToCpuDataBuffer = new std::atomic<real_t>[2*gpuHaloWidth*gpuDimX + 2*gpuHaloWidth*gpuDimY]{};

    // set up buffers on device
    try {
        cl_gpuToCpuDataBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (2*gpuHaloWidth*gpuDimX+2*gpuHaloWidth*gpuDimY)*sizeof(real_t));
        cl_cpuToGpuDataBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, (2*gpuHaloWidth*(gpuDimX+2*gpuHaloWidth)+2*gpuHaloWidth*gpuDimY)*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_gpuToCpuDataBuffer, 0, 0, (2*gpuHaloWidth*gpuDimX + 2*gpuHaloWidth*gpuDimY)*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_cpuToGpuDataBuffer, 0, 0, (2*gpuHaloWidth*(gpuDimX+2) + 2*gpuHaloWidth*gpuDimY)*sizeof(real_t));
        cl_gpuDimX = cl::Buffer(cl_context, &gpuDimX, (&gpuDimX)+1, true);
        cl_gpuDimY = cl::Buffer(cl_context, &gpuDimY, (&gpuDimY)+1, true);
    } catch(cl::Error error) {
        std::cout << "Tausch2D :: [setup send/recv buffer] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

void Tausch2D::setGPUStencil(cl::Buffer &stencil, int stencilNumPoints, int stencilDimX, int stencilDimY) {

    // check whether OpenCL has been set up
    if(!gpuEnabled) {
        std::cerr << "Tausch2D :: ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    gpuStencilInfoGiven = true;

    // store parameters
    gpuStencil = stencil;
    this->stencilNumPoints = stencilNumPoints;
    this->stencilDimX = (stencilDimX == 0 ? gpuDimX : stencilDimX);
    this->stencilDimY = (stencilDimY == 0 ? gpuDimY : stencilDimY);
    stencilDimX = this->stencilDimX;
    stencilDimY = this->stencilDimY;

    // store buffer to store the GPU and the CPU part of the halo.
    // We do not need two buffers each, as each thread has direct access to both arrays, no MPI communication necessary
    cpuToGpuStencilBuffer = new std::atomic<real_t>[stencilNumPoints*(2*gpuHaloWidth*(stencilDimX+2*gpuHaloWidth) + 2*gpuHaloWidth*stencilDimY)]{};
    gpuToCpuStencilBuffer = new std::atomic<real_t>[stencilNumPoints*(2*gpuHaloWidth*stencilDimX + 2*gpuHaloWidth*stencilDimY)]{};

    // set up buffers on device
    try {
        cl_gpuToCpuStencilBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, stencilNumPoints*(2*gpuHaloWidth*stencilDimX+2*gpuHaloWidth*stencilDimY)*sizeof(real_t));
        cl_cpuToGpuStencilBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, stencilNumPoints*(2*gpuHaloWidth*(stencilDimX+2*gpuHaloWidth)+2*gpuHaloWidth*stencilDimY)*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_gpuToCpuStencilBuffer, 0, 0, stencilNumPoints*(2*gpuHaloWidth*stencilDimX + 2*gpuHaloWidth*stencilDimY)*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_cpuToGpuStencilBuffer, 0, 0, stencilNumPoints*(2*gpuHaloWidth*(stencilDimX+2) + 2*gpuHaloWidth*stencilDimY)*sizeof(real_t));
        cl_stencilDimX = cl::Buffer(cl_context, &stencilDimX, (&stencilDimX)+1, true);
        cl_stencilDimY = cl::Buffer(cl_context, &stencilDimY, (&stencilDimY)+1, true);
        cl_stencilNumPoints = cl::Buffer(cl_context, &stencilNumPoints, (&stencilNumPoints)+1, true);
    } catch(cl::Error error) {
        std::cout << "Tausch2D :: [setup send/recv buffer] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

// collect cpu side of cpu/gpu halo and store in buffer
void Tausch2D::startCpuToGpuData() {

    // check whether GPU is enabled
    if(!gpuEnabled) {
        std::cerr << "Tausch2D :: ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    cpuToGpuDataStarted.store(true);

    // left
    for(int i = 0; i < gpuHaloWidth*gpuDimY; ++ i) {
        int index = ((localDimY-gpuDimY)/2 +i/gpuHaloWidth+cpuHaloWidth)*(localDimX+2*cpuHaloWidth) +
                    (localDimX-gpuDimX)/2+i%gpuHaloWidth + cpuHaloWidth-gpuHaloWidth;
        cpuToGpuDataBuffer[i].store(cpuData[index]);
    }
    // right
    for(int i = 0; i < gpuHaloWidth*gpuDimY; ++ i) {
        int index = ((localDimY-gpuDimY)/2 +cpuHaloWidth + i/gpuHaloWidth)*(localDimX+2*cpuHaloWidth) +
                    (localDimX-gpuDimX)/2 + cpuHaloWidth + gpuDimX + i%gpuHaloWidth;
        cpuToGpuDataBuffer[gpuHaloWidth*gpuDimY + i].store(cpuData[index]);
    }
    // top
    for(int i = 0; i < gpuHaloWidth*(gpuDimX+2*gpuHaloWidth); ++ i) {
        int index = ((localDimY-gpuDimY)/2+gpuDimY+cpuHaloWidth + i/(gpuDimX+2*gpuHaloWidth))*(localDimX+2*cpuHaloWidth) +
                    cpuHaloWidth + ((localDimX-gpuDimX)/2-gpuHaloWidth) +i%(gpuDimX+2*gpuHaloWidth);
        cpuToGpuDataBuffer[2*gpuHaloWidth*gpuDimY + i].store(cpuData[index]);
    }
    // bottom
    for(int i = 0; i < gpuHaloWidth*(gpuDimX+2*gpuHaloWidth); ++ i) {
        int index = (i/(gpuDimX+2*gpuHaloWidth) + cpuHaloWidth + (localDimY-gpuDimY)/2 - gpuHaloWidth)*(localDimX+2*cpuHaloWidth) +
                    cpuHaloWidth + (localDimX-gpuDimX)/2 - gpuHaloWidth + i%(gpuDimX+2*gpuHaloWidth);
        cpuToGpuDataBuffer[2*gpuHaloWidth*gpuDimY+gpuHaloWidth*(gpuDimX+2*gpuHaloWidth) + i].store(cpuData[index]);
    }

}

// collect cpu side of cpu/gpu halo and store in buffer
void Tausch2D::startCpuToGpuStencil() {

    // check whether GPU is enabled
    if(!gpuEnabled) {
        std::cerr << "Tausch2D :: ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    cpuToGpuStencilStarted.store(true);

    // left
    for(int i = 0; i < gpuHaloWidth*stencilDimY; ++i) {
        int index = ((localDimY-stencilDimY)/2 +i/gpuHaloWidth+cpuHaloWidth)*(localDimX+2*cpuHaloWidth) +
                    (localDimX-stencilDimX)/2+i%gpuHaloWidth + cpuHaloWidth-gpuHaloWidth;
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuToGpuStencilBuffer[stencilNumPoints*i + j].store(cpuStencil[stencilNumPoints*index + j]);
    }
    // right
    for(int i = 0; i < gpuHaloWidth*stencilDimY; ++i) {
        int index = ((localDimY-stencilDimY)/2 +cpuHaloWidth + i/gpuHaloWidth)*(localDimX+2*cpuHaloWidth) +
                    (localDimX-stencilDimX)/2 + cpuHaloWidth + stencilDimX + i%gpuHaloWidth;
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuToGpuStencilBuffer[stencilNumPoints*(gpuHaloWidth*stencilDimY + i) + j].store(cpuStencil[stencilNumPoints*index + j]);
    }
    // top
    for(int i = 0; i < gpuHaloWidth*(stencilDimX+2*gpuHaloWidth); ++i) {
        int index = ((localDimY-stencilDimY)/2+stencilDimY+cpuHaloWidth + i/(stencilDimX+2*gpuHaloWidth))*(localDimX+2*cpuHaloWidth) +
                    cpuHaloWidth + ((localDimX-stencilDimX)/2-gpuHaloWidth) +i%(stencilDimX+2*gpuHaloWidth);
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuToGpuStencilBuffer[stencilNumPoints*(2*gpuHaloWidth*stencilDimY + i) + j].store(cpuStencil[stencilNumPoints*index + j]);
    }
    // bottom
    for(int i = 0; i < gpuHaloWidth*(stencilDimX+2*gpuHaloWidth); ++i) {
        int index = (i/(stencilDimX+2*gpuHaloWidth) + cpuHaloWidth + (localDimY-stencilDimY)/2 - gpuHaloWidth)*(localDimX+2*cpuHaloWidth) +
                    cpuHaloWidth + (localDimX-stencilDimX)/2 - gpuHaloWidth + i%(stencilDimX+2*gpuHaloWidth);
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuToGpuStencilBuffer[stencilNumPoints*(2*gpuHaloWidth*stencilDimY+gpuHaloWidth*(stencilDimX+2*gpuHaloWidth) + i) + j].store(cpuStencil[stencilNumPoints*index + j]);
    }

}

// collect gpu side of cpu/gpu halo and download into buffer
void Tausch2D::startGpuToCpuData() {

    // check whether GPU is enabled
    if(!gpuEnabled) {
        std::cerr << "Tausch2D :: ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }
    // check whether GPU info was given
    if(!gpuDataInfoGiven) {
        std::cerr << "Tausch2D :: ERROR: GPU info not available! Did you call setGPUData()? Abort..." << std::endl;
        exit(1);
    }

    gpuToCpuDataStarted.store(true);

    try {

        auto kernel_collectHalo = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(cl_programs, "collectHaloData");

        int globalSize = ((2*gpuHaloWidth*gpuDimX+2*gpuHaloWidth*(gpuDimY-2*gpuHaloWidth))/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_collectHalo(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                           cl_gpuDimX, cl_gpuDimY, cl_gpuHaloWidth, gpuData, cl_gpuToCpuDataBuffer);

        double *dat = new double[2*gpuHaloWidth*gpuDimX+2*gpuHaloWidth*(gpuDimY-2*gpuHaloWidth)];
        cl::copy(cl_queue, cl_gpuToCpuDataBuffer, &dat[0], (&dat[2*gpuHaloWidth*gpuDimX+2*gpuHaloWidth*(gpuDimY-2*gpuHaloWidth)-1])+1);
        for(int i = 0; i < 2*gpuHaloWidth*gpuDimX+2*gpuHaloWidth*(gpuDimY-2*gpuHaloWidth); ++i)
            gpuToCpuDataBuffer[i].store(dat[i]);

        delete[] dat;

    } catch(cl::Error error) {
        std::cout << "Tausch2D :: [kernel collectHalo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

// collect gpu side of cpu/gpu halo and download into buffer
void Tausch2D::startGpuToCpuStencil() {

    // check whether GPU is enabled
    if(!gpuEnabled) {
        std::cerr << "Tausch2D :: ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }
    // check whether GPU info was given
    if(!gpuStencilInfoGiven) {
        std::cerr << "Tausch2D :: ERROR: GPU info not available! Did you call setGPUStencil()? Abort..." << std::endl;
        exit(1);
    }

    gpuToCpuStencilStarted.store(true);

    try {

        auto kernel_collectHalo = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(cl_programs, "collectHaloStencil");

        int globalSize = ((stencilNumPoints*(2*gpuHaloWidth*stencilDimX+2*gpuHaloWidth*(stencilDimY-2*gpuHaloWidth)))/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_collectHalo(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                           cl_stencilDimX, cl_stencilDimY, cl_gpuHaloWidth, gpuStencil, cl_stencilNumPoints, cl_gpuToCpuStencilBuffer);

        double *dat = new double[stencilNumPoints*(2*gpuHaloWidth*stencilDimX+2*gpuHaloWidth*(stencilDimY-2*gpuHaloWidth))];
        cl::copy(cl_queue, cl_gpuToCpuStencilBuffer, &dat[0], (&dat[stencilNumPoints*(2*gpuHaloWidth*stencilDimX+2*gpuHaloWidth*(stencilDimY-2*gpuHaloWidth))-1])+1);
        for(int i = 0; i < stencilNumPoints*(2*gpuHaloWidth*stencilDimX+2*gpuHaloWidth*(stencilDimY-2*gpuHaloWidth)); ++i)
            gpuToCpuStencilBuffer[i].store(dat[i]);

        delete[] dat;

    } catch(cl::Error error) {
        std::cout << "Tausch2D :: [kernel collectHalo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

// Complete CPU side of CPU/GPU halo exchange
void Tausch2D::completeGpuToCpuData() {

    // we need to wait for the GPU thread to arrive here
    if(blockingSyncCpuGpu)
        syncCpuAndGpu(false);

    if(!gpuToCpuDataStarted.load()) {
        std::cerr << "Tausch2D :: ERROR: No CPU->GPU data exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    // left
    for(int i = 0; i < gpuHaloWidth*(gpuDimY-2*gpuHaloWidth); ++ i) {
        int index = ((localDimY-gpuDimY)/2 +i/gpuHaloWidth+cpuHaloWidth+gpuHaloWidth)*(localDimX+2*cpuHaloWidth) +
                    (localDimX-gpuDimX)/2+i%gpuHaloWidth + cpuHaloWidth;
        cpuData[index] = gpuToCpuDataBuffer[i].load();
    }
    // right
    for(int i = 0; i < gpuHaloWidth*(gpuDimY-2*gpuHaloWidth); ++ i) {
        int index = ((localDimY-gpuDimY)/2 +cpuHaloWidth + i/gpuHaloWidth + gpuHaloWidth)*(localDimX+2*cpuHaloWidth) +
                    (localDimX-gpuDimX)/2 + cpuHaloWidth-gpuHaloWidth + gpuDimX + i%gpuHaloWidth;
        cpuData[index] = gpuToCpuDataBuffer[gpuHaloWidth*(gpuDimY-2*gpuHaloWidth) + i].load();
    }
    // top
    for(int i = 0; i < gpuHaloWidth*gpuDimX; ++ i) {
        int index = ((localDimY-gpuDimY)/2+gpuDimY+cpuHaloWidth-gpuHaloWidth + i/gpuDimX)*(localDimX+2*cpuHaloWidth) +
                    cpuHaloWidth + ((localDimX-gpuDimX)/2-gpuHaloWidth) +i%gpuDimX + gpuHaloWidth;
        cpuData[index] = gpuToCpuDataBuffer[2*gpuHaloWidth*(gpuDimY-2*gpuHaloWidth) + i].load();
    }
    // bottom
    for(int i = 0; i < gpuHaloWidth*gpuDimX; ++ i) {
        int index = (i/gpuDimX + cpuHaloWidth + (localDimY-gpuDimY)/2)*(localDimX+2*cpuHaloWidth) +
                    cpuHaloWidth + (localDimX-gpuDimX)/2 + i%gpuDimX;
        cpuData[index] = gpuToCpuDataBuffer[2*gpuHaloWidth*(gpuDimY-2*gpuHaloWidth)+gpuHaloWidth*gpuDimX + i].load();
    }

}

// Complete CPU side of CPU/GPU halo exchange
void Tausch2D::completeGpuToCpuStencil() {

    // we need to wait for the GPU thread to arrive here
    if(blockingSyncCpuGpu)
        syncCpuAndGpu(true);

    if(!gpuToCpuStencilStarted.load()) {
        std::cerr << "Tausch2D :: ERROR: No CPU->GPU stencil exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    // left
    for(int i = 0; i < gpuHaloWidth*(stencilDimY-2*gpuHaloWidth); ++ i) {
        int index = ((localDimY-stencilDimY)/2 +i/gpuHaloWidth+cpuHaloWidth+gpuHaloWidth)*(localDimX+2*cpuHaloWidth) +
                    (localDimX-stencilDimX)/2+i%gpuHaloWidth + cpuHaloWidth;
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuStencil[stencilNumPoints*index + j] = gpuToCpuStencilBuffer[stencilNumPoints*i + j].load();
    }
    // right
    for(int i = 0; i < gpuHaloWidth*(stencilDimY-2*gpuHaloWidth); ++ i) {
        int index = ((localDimY-stencilDimY)/2 +cpuHaloWidth + i/gpuHaloWidth + gpuHaloWidth)*(localDimX+2*cpuHaloWidth) +
                    (localDimX-stencilDimX)/2 + cpuHaloWidth-gpuHaloWidth + stencilDimX + i%gpuHaloWidth;
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuStencil[stencilNumPoints*index + j] = gpuToCpuStencilBuffer[stencilNumPoints*(gpuHaloWidth*(stencilDimY-2*gpuHaloWidth) + i) + j].load();
    }
    // top
    for(int i = 0; i < gpuHaloWidth*stencilDimX; ++ i) {
        int index = ((localDimY-stencilDimY)/2+stencilDimY+cpuHaloWidth-gpuHaloWidth + i/stencilDimX)*(localDimX+2*cpuHaloWidth) +
                    cpuHaloWidth + ((localDimX-stencilDimX)/2-gpuHaloWidth) +i%stencilDimX + gpuHaloWidth;
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuStencil[stencilNumPoints*index + j] = gpuToCpuStencilBuffer[stencilNumPoints*(2*gpuHaloWidth*(stencilDimY-2*gpuHaloWidth) + i) + j].load();
    }
    // bottom
    for(int i = 0; i < gpuHaloWidth*stencilDimX; ++ i) {
        int index = (i/stencilDimX + cpuHaloWidth + (localDimY-stencilDimY)/2)*(localDimX+2*cpuHaloWidth) +
                    cpuHaloWidth + (localDimX-stencilDimX)/2 + i%stencilDimX;
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuStencil[stencilNumPoints*index + j] = gpuToCpuStencilBuffer[stencilNumPoints*(2*gpuHaloWidth*(stencilDimY-2*gpuHaloWidth)+gpuHaloWidth*stencilDimX + i) + j].load();
    }

}

// Complete GPU side of CPU/GPU halo exchange
void Tausch2D::completeCpuToGpuData() {

    // we need to wait for the CPU thread to arrive here
    if(blockingSyncCpuGpu)
        syncCpuAndGpu(false);

    if(!cpuToGpuDataStarted.load()) {
        std::cerr << "Tausch2D :: ERROR: No GPU->CPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    try {

        double *dat = new double[2*gpuHaloWidth*(gpuDimX+2*gpuHaloWidth)+2*gpuHaloWidth*gpuDimY];
        for(int i = 0; i < 2*gpuHaloWidth*(gpuDimX+2*gpuHaloWidth)+2*gpuHaloWidth*gpuDimY; ++i)
            dat[i] = cpuToGpuDataBuffer[i].load();

        cl::copy(cl_queue, &dat[0], (&dat[2*gpuHaloWidth*(gpuDimX+2*gpuHaloWidth)+2*gpuHaloWidth*gpuDimY-1])+1, cl_cpuToGpuDataBuffer);

        delete[] dat;

        auto kernel_distributeHaloData = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(cl_programs, "distributeHaloData");

        int globalSize = ((2*gpuHaloWidth*(gpuDimX+2*gpuHaloWidth) + 2*gpuHaloWidth*gpuDimY)/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_distributeHaloData(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                                  cl_gpuDimX, cl_gpuDimY, cl_gpuHaloWidth, gpuData, cl_cpuToGpuDataBuffer);

    } catch(cl::Error error) {
        std::cout << "Tausch2D :: [dist halo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

// Complete GPU side of CPU/GPU halo exchange
void Tausch2D::completeCpuToGpuStencil() {

    // we need to wait for the CPU thread to arrive here
    if(blockingSyncCpuGpu)
        syncCpuAndGpu(true);

    if(!cpuToGpuStencilStarted.load()) {
        std::cerr << "Tausch2D :: ERROR: No GPU->CPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    try {

        double *dat = new double[stencilNumPoints*(2*gpuHaloWidth*(gpuDimX+2*gpuHaloWidth)+2*gpuHaloWidth*gpuDimY)];
        for(int i = 0; i < stencilNumPoints*(2*gpuHaloWidth*(gpuDimX+2*gpuHaloWidth)+2*gpuHaloWidth*gpuDimY); ++i)
            dat[i] = cpuToGpuStencilBuffer[i].load();

        cl::copy(cl_queue, &dat[0], (&dat[stencilNumPoints*(2*gpuHaloWidth*(gpuDimX+2*gpuHaloWidth)+2*gpuHaloWidth*gpuDimY)-1])+1, cl_cpuToGpuStencilBuffer);

        delete[] dat;

        auto kernel_distributeHaloData = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(cl_programs, "distributeHaloStencil");

        int globalSize = ((stencilNumPoints*(2*gpuHaloWidth*(gpuDimX+2*gpuHaloWidth) + 2*gpuHaloWidth*gpuDimY))/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_distributeHaloData(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                                  cl_gpuDimX, cl_gpuDimY, cl_gpuHaloWidth, gpuStencil, cl_stencilNumPoints, cl_cpuToGpuStencilBuffer);

    } catch(cl::Error error) {
        std::cout << "Tausch2D :: [dist halo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

// both the CPU and GPU have to arrive at this point before either can continue
void Tausch2D::syncCpuAndGpu(bool offsetByTwo) {

    int starti = (offsetByTwo ? 2 : 0);
    int endi = (offsetByTwo ? 4 : 2);

    // need to do this twice to prevent potential (though unlikely) deadlocks
    for(int i = starti; i < endi; ++i) {

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

void Tausch2D::compileKernels() {

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
kernel void collectHaloStencil(global const int * restrict const dimX, global const int * restrict const dimY,
                               global const int * restrict const haloWidth, global const real_t * restrict const vec,
                               global const int * restrict const stencilNumPoints, global real_t * sync) {
    unsigned int current = get_global_id(0);
    unsigned int maxNum = (*stencilNumPoints)*(2*(*haloWidth)*(*dimX) + 2*(*haloWidth)*(*dimY));
    if(current >= maxNum)
        return;
    // left
    if(current < (*stencilNumPoints)*(*haloWidth)*(*dimY-2*(*haloWidth))) {
        int currentStencil = current%(*stencilNumPoints);
        int currentIndex = current/(*stencilNumPoints);
        int index = (*stencilNumPoints)*((*haloWidth+currentIndex/(*haloWidth)+(*haloWidth))*(*dimX+2*(*haloWidth)) +(*haloWidth) + currentIndex%(*haloWidth)) + currentStencil;
        sync[current] = vec[index];
        return;
    }
    // right
    if(current < (*stencilNumPoints)*2*(*haloWidth)*(*dimY-2*(*haloWidth))) {
        int touse = current-(*stencilNumPoints)*(*haloWidth)*(*dimY-2*(*haloWidth));
        int currentStencil = touse%(*stencilNumPoints);
        int currentIndex = touse/(*stencilNumPoints);
        int index = (*stencilNumPoints)*((1+(*haloWidth)+currentIndex/(*haloWidth)+(*haloWidth))*(*dimX+2*(*haloWidth)) -2*(*haloWidth)+currentIndex%(*haloWidth)) + currentStencil;
        sync[current] = vec[index];
        return;
    }
    // top
    if(current < (*stencilNumPoints)*(2*(*haloWidth)*(*dimY-2*(*haloWidth)) + (*haloWidth)*(*dimX))) {
        int touse = current-2*(*stencilNumPoints)*(*haloWidth)*(*dimY-2*(*haloWidth));
        int currentStencil = touse%(*stencilNumPoints);
        int currentIndex = touse/(*stencilNumPoints);
        int index = (*stencilNumPoints)*((*dimX+2*(*haloWidth))*(*dimY + currentIndex/(*dimX)) + currentIndex%(*dimX)+(*haloWidth)) + currentStencil;
        sync[current] = vec[index];
        return;
    }
    // bottom
    int touse = current - 2*(*stencilNumPoints)*(*haloWidth)*(*dimY-2*(*haloWidth)) - (*stencilNumPoints)*(*haloWidth)*(*dimX);
    int currentStencil = touse%(*stencilNumPoints);
    int currentIndex = touse/(*stencilNumPoints);
    int index = (*stencilNumPoints)*(((*haloWidth)+currentIndex/(*dimX))*(*dimX+2*(*haloWidth))+currentIndex%(*dimX)+(*haloWidth)) + currentStencil;
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
kernel void distributeHaloStencil(global const int * restrict const dimX, global const int * restrict const dimY,
                                  global const int * restrict const haloWidth, global real_t * vec,
                                  global const int * restrict const stencilNumPoints, global const real_t * restrict const sync) {
    unsigned int current = get_global_id(0);
    unsigned int maxNum = (*stencilNumPoints)*(2*(*haloWidth)*(*dimX+2*(*haloWidth)) + 2*(*haloWidth)*(*dimY));
    if(current >= maxNum)
        return;
    // left
    if(current < (*stencilNumPoints)*(*haloWidth)*(*dimY)) {
        int currentStencil = current%(*stencilNumPoints);
        int currentIndex = current/(*stencilNumPoints);
        int index = (*stencilNumPoints)*((*haloWidth+currentIndex/(*haloWidth))*(*dimX+2*(*haloWidth))) + currentStencil;
        vec[index] = sync[current];
        return;
    }
    // right
    if(current < (*stencilNumPoints)*2*(*haloWidth)*(*dimY)) {
        int touse = current-(*stencilNumPoints)*(*haloWidth)*(*dimY);
        int currentStencil = touse%(*stencilNumPoints);
        int currentIndex = touse/(*stencilNumPoints);
        int index = (*stencilNumPoints)*(((*haloWidth)+1+currentIndex/(*haloWidth))*(*dimX+2*(*haloWidth)) -(*haloWidth)+currentIndex%(*haloWidth)) + currentStencil;
        vec[index] = sync[current];
        return;
    }
    // top
    if(current < (*stencilNumPoints)*(2*(*haloWidth)*(*dimY)+(*haloWidth)*(*dimX+2*(*haloWidth)))) {
        int touse = current-2*(*stencilNumPoints)*(*haloWidth)*(*dimY);
        int currentStencil = touse%(*stencilNumPoints);
        int currentIndex = touse/(*stencilNumPoints);
        int index = (*stencilNumPoints)*((*dimX+2*(*haloWidth))*(*dimY+(*haloWidth) + currentIndex/(*dimX+2*(*haloWidth))) + currentIndex%(*dimX+2*(*haloWidth))) + currentStencil;
        vec[index] = sync[current];
        return;
    }
    // bottom
    int touse = current - (*stencilNumPoints)*(2*(*haloWidth)*(*dimY)+(*haloWidth)*(*dimX+2*(*haloWidth)));
    int currentStencil = touse%(*stencilNumPoints);
    int currentIndex = touse/(*stencilNumPoints);
    int index = (*stencilNumPoints)*((currentIndex/(*dimX+2*(*haloWidth)))*(*dimX+2*(*haloWidth))+currentIndex%(*dimX+2*(*haloWidth))) + currentStencil;
    vec[index] = sync[current];
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
void Tausch2D::setupOpenCL(bool giveOpenCLDeviceName) {

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
