#include "tausch2d.h"

Tausch2D::Tausch2D(int *localDim, int *mpiNum, int cpuHaloWidth, MPI_Comm comm) {
    int useHaloWidth[4] = {cpuHaloWidth, cpuHaloWidth, cpuHaloWidth, cpuHaloWidth};
    _constructor(localDim, mpiNum, useHaloWidth, comm);
}

Tausch2D::Tausch2D(int *localDim, int *mpiNum, int *cpuHaloWidth, MPI_Comm comm) {
    _constructor(localDim, mpiNum, cpuHaloWidth, comm);
}

void Tausch2D::_constructor(int *localDim, int *mpiNum, int *cpuHaloWidth, MPI_Comm comm) {

    MPI_Comm_dup(comm, &TAUSCH_COMM);

    // get MPI info
    MPI_Comm_rank(TAUSCH_COMM, &mpiRank);
    MPI_Comm_size(TAUSCH_COMM, &mpiSize);

    this->mpiNum[X] = mpiNum[X];
    this->mpiNum[Y] = mpiNum[Y];

    // store configuration
    this->localDim[X] = localDim[X];
    this->localDim[Y] = localDim[Y];

    this->cpuHaloWidth[0] = cpuHaloWidth[0];
    this->cpuHaloWidth[1] = cpuHaloWidth[1];
    this->cpuHaloWidth[2] = cpuHaloWidth[2];
    this->cpuHaloWidth[3] = cpuHaloWidth[3];

    // check if this rank has a boundary with another rank
    haveBoundary[LEFT] = (mpiRank%mpiNum[X] != 0);
    haveBoundary[RIGHT] = ((mpiRank+1)%mpiNum[X] != 0);
    haveBoundary[TOP] = (mpiRank < mpiSize-mpiNum[X]);
    haveBoundary[BOTTOM] = (mpiRank > mpiNum[X]-1);

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
    for(int i = 0; i < 4; ++i) {
        sync_counter[i].store(0);
        sync_lock[i].store(0);
    }

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
void Tausch2D::setCpuData(real_t *data) {

    cpuInfoGiven = true;
    cpuData = data;

    // a send and recv buffer for the CPU-CPU communication
    cpuToCpuSendBuffer = new real_t*[4];
    cpuToCpuSendBuffer[LEFT] = new real_t[cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM])]{};
    cpuToCpuSendBuffer[RIGHT] = new real_t[cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM])]{};
    cpuToCpuSendBuffer[TOP] = new real_t[cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])]{};
    cpuToCpuSendBuffer[BOTTOM] = new real_t[cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])]{};
    cpuToCpuRecvBuffer = new real_t*[4];
    cpuToCpuRecvBuffer[LEFT] = new real_t[cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM])]{};
    cpuToCpuRecvBuffer[RIGHT] = new real_t[cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM])]{};
    cpuToCpuRecvBuffer[TOP] = new real_t[cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])]{};
    cpuToCpuRecvBuffer[BOTTOM] = new real_t[cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])]{};

    if(haveBoundary[LEFT]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[LEFT], cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM]),
                      mpiDataType, mpiRank-1, 0, TAUSCH_COMM, &cpuToCpuRecvRequest[LEFT]);
        MPI_Send_init(cpuToCpuSendBuffer[LEFT], cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM]),
                      mpiDataType, mpiRank-1, 2, TAUSCH_COMM, &cpuToCpuSendRequest[LEFT]);
    }
    if(haveBoundary[RIGHT]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[RIGHT], cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM]),
                      mpiDataType, mpiRank+1, 2, TAUSCH_COMM, &cpuToCpuRecvRequest[RIGHT]);
        MPI_Send_init(cpuToCpuSendBuffer[RIGHT], cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM]),
                      mpiDataType, mpiRank+1, 0, TAUSCH_COMM, &cpuToCpuSendRequest[RIGHT]);
    }
    if(haveBoundary[TOP]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[TOP], cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]),
                      mpiDataType, mpiRank+mpiNum[X], 1, TAUSCH_COMM, &cpuToCpuRecvRequest[TOP]);
        MPI_Send_init(cpuToCpuSendBuffer[TOP], cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]),
                      mpiDataType, mpiRank+mpiNum[X], 3, TAUSCH_COMM, &cpuToCpuSendRequest[TOP]);
    }
    if(haveBoundary[BOTTOM]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[BOTTOM], cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]),
                      mpiDataType, mpiRank-mpiNum[X], 3, TAUSCH_COMM, &cpuToCpuRecvRequest[BOTTOM]);
        MPI_Send_init(cpuToCpuSendBuffer[BOTTOM], cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]),
                      mpiDataType, mpiRank-mpiNum[X], 1, TAUSCH_COMM, &cpuToCpuSendRequest[BOTTOM]);
    }

}

void Tausch2D::setCpuStencil(real_t *stencil, int stencilNumPoints) {

    stencilInfoGiven = true;
    cpuStencil = stencil;
    this->stencilNumPoints = stencilNumPoints;

    cpuToCpuStencilSendBuffer = new real_t*[4];
    cpuToCpuStencilSendBuffer[LEFT] = new real_t[stencilNumPoints*cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM])]{};
    cpuToCpuStencilSendBuffer[RIGHT] = new real_t[stencilNumPoints*cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM])]{};
    cpuToCpuStencilSendBuffer[TOP] = new real_t[stencilNumPoints*cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])]{};
    cpuToCpuStencilSendBuffer[BOTTOM] = new real_t[stencilNumPoints*cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])]{};
    cpuToCpuStencilRecvBuffer = new real_t*[4];
    cpuToCpuStencilRecvBuffer[LEFT] = new real_t[stencilNumPoints*cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM])]{};
    cpuToCpuStencilRecvBuffer[RIGHT] = new real_t[stencilNumPoints*cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM])]{};
    cpuToCpuStencilRecvBuffer[TOP] = new real_t[stencilNumPoints*cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])]{};
    cpuToCpuStencilRecvBuffer[BOTTOM] = new real_t[stencilNumPoints*cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])]{};

    if(haveBoundary[LEFT]) {
        MPI_Recv_init(cpuToCpuStencilRecvBuffer[LEFT], stencilNumPoints*cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM]),
                      mpiDataType, mpiRank-1, 0, TAUSCH_COMM, &cpuToCpuStencilRecvRequest[LEFT]);
        MPI_Send_init(cpuToCpuStencilSendBuffer[LEFT], stencilNumPoints*cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM]),
                      mpiDataType, mpiRank-1, 2, TAUSCH_COMM, &cpuToCpuStencilSendRequest[LEFT]);
    }
    if(haveBoundary[RIGHT]) {
        MPI_Recv_init(cpuToCpuStencilRecvBuffer[RIGHT], stencilNumPoints*cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM]),
                      mpiDataType, mpiRank+1, 2, TAUSCH_COMM, &cpuToCpuStencilRecvRequest[RIGHT]);
        MPI_Send_init(cpuToCpuStencilSendBuffer[RIGHT], stencilNumPoints*cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM]),
                      mpiDataType, mpiRank+1, 0, TAUSCH_COMM, &cpuToCpuStencilSendRequest[RIGHT]);
    }
    if(haveBoundary[TOP]) {
        MPI_Recv_init(cpuToCpuStencilRecvBuffer[TOP], stencilNumPoints*cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]),
                      mpiDataType, mpiRank+mpiNum[X], 1, TAUSCH_COMM, &cpuToCpuStencilRecvRequest[TOP]);
        MPI_Send_init(cpuToCpuStencilSendBuffer[TOP], stencilNumPoints*cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]),
                      mpiDataType, mpiRank+mpiNum[X], 3, TAUSCH_COMM, &cpuToCpuStencilSendRequest[TOP]);
    }
    if(haveBoundary[BOTTOM]) {
        MPI_Recv_init(cpuToCpuStencilRecvBuffer[BOTTOM], stencilNumPoints*cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]),
                      mpiDataType, mpiRank-mpiNum[X], 3, TAUSCH_COMM, &cpuToCpuStencilRecvRequest[BOTTOM]);
        MPI_Send_init(cpuToCpuStencilSendBuffer[BOTTOM], stencilNumPoints*cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]),
                      mpiDataType, mpiRank-mpiNum[X], 1, TAUSCH_COMM, &cpuToCpuStencilSendRequest[BOTTOM]);
    }

}

// post the MPI_Irecv's for inter-rank communication
void Tausch2D::postCpuDataReceives() {

    if(!cpuInfoGiven) {
        std::cerr << "Tausch2D :: ERROR: You didn't tell me yet where to find the data! Abort..." << std::endl;
        exit(1);
    }

    cpuRecvsPosted = true;

    for(int i  0; i < 4; ++i)
        if(haveBoundary[i])
            MPI_Start(&cpuToCpuRecvRequest[LEFT]);

}

// post the MPI_Irecv's for inter-rank communication
void Tausch2D::postCpuStencilReceives() {

    if(!cpuInfoGiven) {
        std::cerr << "Tausch2D :: ERROR: You didn't tell me yet where to find the data! Abort..." << std::endl;
        exit(1);
    }

    stencilRecvsPosted = true;

    for(int i  0; i < 4; ++i)
        if(haveBoundary[i])
            MPI_Start(&cpuToCpuStencilRecvRequest[LEFT]);

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
        for(int i = 0; i < cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM]); ++i)
            cpuToCpuSendBuffer[LEFT][i] = cpuData[cpuHaloWidth[LEFT]+ (i/cpuHaloWidth[RIGHT])*(localDim[X]+cpuHaloWidth[LEFT]+
                                                  cpuHaloWidth[RIGHT])+i%cpuHaloWidth[RIGHT]];
        MPI_Start(&cpuToCpuSendRequest[LEFT]);
    } else if(edge == RIGHT && haveBoundary[RIGHT]) {
        for(int i = 0; i < cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM]); ++i)
            cpuToCpuSendBuffer[RIGHT][i] = cpuData[(i/cpuHaloWidth[LEFT]+1)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) -
                                                   (cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+i%cpuHaloWidth[LEFT]];
        MPI_Start(&cpuToCpuSendRequest[RIGHT]);
    } else if(edge == TOP && haveBoundary[TOP]) {
        for(int i = 0; i < cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]); ++i)
            cpuToCpuSendBuffer[TOP][i] = cpuData[(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]) + i];
        MPI_Start(&cpuToCpuSendRequest[TOP]);
    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {
        for(int i = 0; i < cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]); ++i)
            cpuToCpuSendBuffer[BOTTOM][i] = cpuData[cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + i];
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
        for(int i = 0; i < cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuToCpuStencilSendBuffer[LEFT][stencilNumPoints*i + j] =
                        cpuStencil[stencilNumPoints*(cpuHaloWidth[LEFT]+ (i/cpuHaloWidth[RIGHT])*(localDim[X]+cpuHaloWidth[LEFT]+
                                                     cpuHaloWidth[RIGHT])+i%cpuHaloWidth[RIGHT]) + j];
        MPI_Start(&cpuToCpuStencilSendRequest[LEFT]);
    } else if(edge == RIGHT && haveBoundary[RIGHT]) {
        for(int i = 0; i < cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM]); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuToCpuStencilSendBuffer[RIGHT][stencilNumPoints*i + j] =
                        cpuStencil[stencilNumPoints*((i/cpuHaloWidth[LEFT]+1)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) -
                                                     (cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+i%cpuHaloWidth[LEFT]) + j];
        MPI_Start(&cpuToCpuStencilSendRequest[RIGHT]);
    } else if(edge == TOP && haveBoundary[TOP]) {
        for(int i = 0; i < cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuToCpuStencilSendBuffer[TOP][stencilNumPoints*i + j] =
                        cpuStencil[stencilNumPoints*((localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]) + i) + j];
        MPI_Start(&cpuToCpuStencilSendRequest[TOP]);
    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {
        for(int i = 0; i < cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuToCpuStencilSendBuffer[BOTTOM][stencilNumPoints*i + j] =
                        cpuStencil[stencilNumPoints*(cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + i) + j];
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
        for(int i = 0; i < cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM]); ++i)
            cpuData[(i/cpuHaloWidth[LEFT])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+i%cpuHaloWidth[LEFT]] = cpuToCpuRecvBuffer[LEFT][i];
        MPI_Wait(&cpuToCpuSendRequest[LEFT], MPI_STATUS_IGNORE);
    } else if(edge == RIGHT && haveBoundary[RIGHT]) {
        MPI_Wait(&cpuToCpuRecvRequest[RIGHT], MPI_STATUS_IGNORE);
        for(int i = 0; i < cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM]); ++i)
            cpuData[(i/cpuHaloWidth[RIGHT]+1)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])-cpuHaloWidth[RIGHT]+i%cpuHaloWidth[RIGHT]] = cpuToCpuRecvBuffer[RIGHT][i];
        MPI_Wait(&cpuToCpuSendRequest[RIGHT], MPI_STATUS_IGNORE);
    } else if(edge == TOP && haveBoundary[TOP]) {
        MPI_Wait(&cpuToCpuRecvRequest[TOP], MPI_STATUS_IGNORE);
        for(int i = 0; i < cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]); ++i)
            cpuData[(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]) + i] = cpuToCpuRecvBuffer[TOP][i];
        MPI_Wait(&cpuToCpuSendRequest[TOP], MPI_STATUS_IGNORE);
    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {
        MPI_Wait(&cpuToCpuRecvRequest[BOTTOM], MPI_STATUS_IGNORE);
        for(int i = 0; i < cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]); ++i)
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
        for(int i = 0; i < cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM]); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuStencil[stencilNumPoints*((i/cpuHaloWidth[LEFT])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+
                                             i%cpuHaloWidth[LEFT]) + j] = cpuToCpuStencilRecvBuffer[LEFT][stencilNumPoints*i + j];
        MPI_Wait(&cpuToCpuStencilSendRequest[LEFT], MPI_STATUS_IGNORE);
    } else if(edge == RIGHT && haveBoundary[RIGHT]) {
        MPI_Wait(&cpuToCpuStencilRecvRequest[RIGHT], MPI_STATUS_IGNORE);
        for(int i = 0; i < cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[TOP]+cpuHaloWidth[BOTTOM]); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuStencil[stencilNumPoints*((i/cpuHaloWidth[RIGHT]+1)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])-
                                             cpuHaloWidth[RIGHT]+i%cpuHaloWidth[RIGHT]) + j] = cpuToCpuStencilRecvBuffer[RIGHT][stencilNumPoints*i + j];
        MPI_Wait(&cpuToCpuStencilSendRequest[RIGHT], MPI_STATUS_IGNORE);
    } else if(edge == TOP && haveBoundary[TOP]) {
        MPI_Wait(&cpuToCpuStencilRecvRequest[TOP], MPI_STATUS_IGNORE);
        for(int i = 0; i < cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuStencil[stencilNumPoints*((localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]) + i) + j] =
                        cpuToCpuStencilRecvBuffer[TOP][stencilNumPoints*i + j];
        MPI_Wait(&cpuToCpuStencilSendRequest[TOP], MPI_STATUS_IGNORE);
    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {
        MPI_Wait(&cpuToCpuStencilRecvRequest[BOTTOM], MPI_STATUS_IGNORE);
        for(int i = 0; i < cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]); ++i)
            for(int j = 0; j < stencilNumPoints; ++j)
                cpuStencil[stencilNumPoints*i + j] = cpuToCpuStencilRecvBuffer[BOTTOM][stencilNumPoints*i + j];
        MPI_Wait(&cpuToCpuStencilSendRequest[BOTTOM], MPI_STATUS_IGNORE);
    }

}

#ifdef TAUSCH_OPENCL

void Tausch2D::enableOpenCL(int *gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {

    // gpu disabled by default, only enabled if flag is set
    gpuEnabled = true;
    // local workgroup size
    cl_kernelLocalSize = clLocalWorkgroupSize;
    this->blockingSyncCpuGpu = blockingSyncCpuGpu;
    // the GPU halo widths
    for(int i = 0; i < 4; ++i)
        this->gpuHaloWidth[i] = gpuHaloWidth[i];
    // Tausch creates its own OpenCL environment
    this->setupOpenCL(giveOpenCLDeviceName);

    try {
        cl_gpuHaloWidth = cl::Buffer(cl_context, &gpuHaloWidth[0], (&gpuHaloWidth[3])+1, true);
    } catch(cl::Error error) {
        std::cout << "Tausch2D :: [setup gpuHaloWidth buffer] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

void Tausch2D::enableOpenCL(int gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {
    int useHaloWidth[4] = {gpuHaloWidth, gpuHaloWidth, gpuHaloWidth, gpuHaloWidth};
    enableOpenCL(useHaloWidth, blockingSyncCpuGpu, clLocalWorkgroupSize, giveOpenCLDeviceName);
}

// If Tausch didn't set up OpenCL, the user needs to pass some OpenCL variables
void Tausch2D::enableOpenCL(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue,
                            int *gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize) {

    this->cl_defaultDevice = cl_defaultDevice;
    this->cl_context = cl_context;
    this->cl_queue = cl_queue;
    this->blockingSyncCpuGpu = blockingSyncCpuGpu;
    this->cl_kernelLocalSize = clLocalWorkgroupSize;
    for(int i = 0; i < 4; ++i)
        this->gpuHaloWidth[i] = gpuHaloWidth[i];

    try {
        cl_gpuHaloWidth = cl::Buffer(cl_context, &gpuHaloWidth[0], (&gpuHaloWidth[3])+1, true);
    } catch(cl::Error error) {
        std::cout << "Tausch2D :: [setup gpuHaloWidth buffer] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    gpuEnabled = true;

    compileKernels();

}

void Tausch2D::enableOpenCL(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue,
                            int gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize) {
    int useHaloWidth[4] = {gpuHaloWidth, gpuHaloWidth, gpuHaloWidth, gpuHaloWidth};
    enableOpenCL(cl_defaultDevice, cl_context, cl_queue, useHaloWidth, blockingSyncCpuGpu, clLocalWorkgroupSize);
}

// get a pointer to the GPU buffer and its dimensions
void Tausch2D::setGpuData(cl::Buffer &dat, int *gpuDim) {

    // check whether OpenCL has been set up
    if(!gpuEnabled) {
        std::cerr << "Tausch2D :: ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    gpuDataInfoGiven = true;

    // store parameters
    gpuData = dat;
    this->gpuDim[X] = gpuDim[X];
    this->gpuDim[Y] = gpuDim[Y];

    // store buffer to store the GPU and the CPU part of the halo.
    // We do not need two buffers each, as each thread has direct access to both arrays, no MPI communication necessary
    int cTg = gpuHaloWidth[TOP]*(gpuDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]) +
              gpuHaloWidth[BOTTOM]*(gpuDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]) +
              gpuHaloWidth[LEFT]*gpuDim[Y] +
              gpuHaloWidth[RIGHT]*gpuDim[Y];
    int gTc = gpuHaloWidth[TOP]*gpuDim[X] + gpuHaloWidth[BOTTOM]*gpuDim[X] +
              gpuHaloWidth[LEFT]*gpuDim[Y] + gpuHaloWidth[RIGHT]*gpuDim[Y];
    cpuToGpuDataBuffer = new std::atomic<real_t>[cTg]{};
    gpuToCpuDataBuffer = new std::atomic<real_t>[gTc]{};

    // set up buffers on device
    try {
        cl_gpuToCpuDataBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, gTc*sizeof(real_t));
        cl_cpuToGpuDataBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, cTg*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_gpuToCpuDataBuffer, 0, 0, gTc*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_cpuToGpuDataBuffer, 0, 0, cTg*sizeof(real_t));
        cl_gpuDim[X] = cl::Buffer(cl_context, &gpuDim[X], (&gpuDim[X])+1, true);
        cl_gpuDim[Y] = cl::Buffer(cl_context, &gpuDim[Y], (&gpuDim[Y])+1, true);
    } catch(cl::Error error) {
        std::cout << "Tausch2D :: [setup send/recv buffer] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

void Tausch2D::setGpuStencil(cl::Buffer &stencil, int stencilNumPoints, int *stencilDim) {

    // check whether OpenCL has been set up
    if(!gpuEnabled) {
        std::cerr << "Tausch2D :: ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    gpuStencilInfoGiven = true;

    // store parameters
    gpuStencil = stencil;
    this->stencilNumPoints = stencilNumPoints;
    this->stencilDim[X] = ((stencilDim==nullptr || stencilDim[X] == 0) ? gpuDim[X] : stencilDim[X]);
    this->stencilDim[Y] = ((stencilDim==nullptr || stencilDim[Y] == 0) ? gpuDim[Y] : stencilDim[Y]);

    // store buffer to store the GPU and the CPU part of the halo.
    // We do not need two buffers each, as each thread has direct access to both arrays, no MPI communication necessary
    int cTg = gpuHaloWidth[TOP]*(this->stencilDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]) +
              gpuHaloWidth[BOTTOM]*(this->stencilDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]) +
              gpuHaloWidth[LEFT]*this->stencilDim[Y] +
              gpuHaloWidth[RIGHT]*this->stencilDim[Y];
    int gTc = gpuHaloWidth[TOP]*this->stencilDim[X] + gpuHaloWidth[BOTTOM]*this->stencilDim[X] +
              gpuHaloWidth[LEFT]*this->stencilDim[Y] + gpuHaloWidth[RIGHT]*this->stencilDim[Y];
    cpuToGpuStencilBuffer = new std::atomic<real_t>[stencilNumPoints*cTg]{};
    gpuToCpuStencilBuffer = new std::atomic<real_t>[stencilNumPoints*gTc]{};

    // set up buffers on device
    try {
        cl_gpuToCpuStencilBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, stencilNumPoints*gTc*sizeof(real_t));
        cl_cpuToGpuStencilBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, stencilNumPoints*cTg*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_gpuToCpuStencilBuffer, 0, 0, stencilNumPoints*gTc*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_cpuToGpuStencilBuffer, 0, 0, stencilNumPoints*cTg*sizeof(real_t));
        cl_stencilDim[X] = cl::Buffer(cl_context, &this->stencilDim[X], (&this->stencilDim[X])+1, true);
        cl_stencilDim[Y] = cl::Buffer(cl_context, &this->stencilDim[Y], (&this->stencilDim[Y])+1, true);
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
    for(int i = 0; i < gpuHaloWidth[LEFT]*gpuDim[Y]; ++i) {
        int index = ((localDim[Y]-gpuDim[Y])/2 +i/gpuHaloWidth[LEFT]+cpuHaloWidth[BOTTOM])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    (localDim[X]-gpuDim[X])/2+i%gpuHaloWidth[LEFT] + cpuHaloWidth[LEFT]-gpuHaloWidth[LEFT];
        cpuToGpuDataBuffer[i].store(cpuData[index]);
    }
    // right
    for(int i = 0; i < gpuHaloWidth[RIGHT]*gpuDim[Y]; ++i) {
        int index = ((localDim[Y]-gpuDim[Y])/2 +cpuHaloWidth[BOTTOM] + i/gpuHaloWidth[RIGHT])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    (localDim[X]-gpuDim[X])/2 + cpuHaloWidth[LEFT] + gpuDim[X] + i%gpuHaloWidth[RIGHT];
        cpuToGpuDataBuffer[gpuHaloWidth[LEFT]*gpuDim[Y] + i].store(cpuData[index]);
    }
    // top
    for(int i = 0; i < gpuHaloWidth[TOP]*(gpuDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]); ++i) {
        int index = ((localDim[Y]-gpuDim[Y])/2+gpuDim[Y]+cpuHaloWidth[BOTTOM] + i/(gpuDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]))*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    cpuHaloWidth[LEFT] + ((localDim[X]-gpuDim[X])/2-gpuHaloWidth[LEFT]) +i%(gpuDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]);
        cpuToGpuDataBuffer[gpuHaloWidth[LEFT]*gpuDim[Y] + gpuHaloWidth[RIGHT]*gpuDim[Y] + i].store(cpuData[index]);
    }
    // bottom
    for(int i = 0; i < gpuHaloWidth[BOTTOM]*(gpuDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]); ++i) {
        int index = (i/(gpuDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]) +
                     cpuHaloWidth[BOTTOM] + (localDim[Y]-gpuDim[Y])/2 - gpuHaloWidth[BOTTOM])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    cpuHaloWidth[LEFT] + (localDim[X]-gpuDim[X])/2 - gpuHaloWidth[LEFT] + i%(gpuDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]);
        cpuToGpuDataBuffer[gpuHaloWidth[LEFT]*gpuDim[Y] + gpuHaloWidth[RIGHT]*gpuDim[Y]+gpuHaloWidth[TOP]*(gpuDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]) + i].store(cpuData[index]);
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
    for(int i = 0; i < gpuHaloWidth[LEFT]*stencilDim[Y]; ++i) {
        int index = ((localDim[Y]-stencilDim[Y])/2 +i/gpuHaloWidth[LEFT]+cpuHaloWidth[BOTTOM])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    (localDim[X]-stencilDim[X])/2+i%gpuHaloWidth[LEFT] + cpuHaloWidth[LEFT]-gpuHaloWidth[LEFT];
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuToGpuStencilBuffer[stencilNumPoints*i + j].store(cpuStencil[stencilNumPoints*index + j]);
    }
    // right
    for(int i = 0; i < gpuHaloWidth[RIGHT]*stencilDim[Y]; ++i) {
        int index = ((localDim[Y]-stencilDim[Y])/2 +cpuHaloWidth[BOTTOM] + i/gpuHaloWidth[RIGHT])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    (localDim[X]-stencilDim[X])/2 + cpuHaloWidth[LEFT] + stencilDim[X] + i%gpuHaloWidth[RIGHT];
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuToGpuStencilBuffer[stencilNumPoints*(gpuHaloWidth[LEFT]*stencilDim[Y] + i) + j].store(cpuStencil[stencilNumPoints*index + j]);
    }
    // top
    for(int i = 0; i < gpuHaloWidth[TOP]*(stencilDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]); ++i) {
        int index = ((localDim[Y]-stencilDim[Y])/2+stencilDim[Y]+cpuHaloWidth[BOTTOM] +
                     i/(stencilDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]))*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    cpuHaloWidth[LEFT] + ((localDim[X]-stencilDim[X])/2-gpuHaloWidth[LEFT]) +
                    i%(stencilDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]);
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuToGpuStencilBuffer[stencilNumPoints*(gpuHaloWidth[LEFT]*stencilDim[Y] + gpuHaloWidth[RIGHT]*stencilDim[Y] + i) + j].store(cpuStencil[stencilNumPoints*index + j]);
    }
    // bottom
    for(int i = 0; i < gpuHaloWidth[BOTTOM]*(stencilDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]); ++i) {
        int index = (i/(stencilDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]) + cpuHaloWidth[BOTTOM] +
                     (localDim[Y]-stencilDim[Y])/2 - gpuHaloWidth[BOTTOM])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    cpuHaloWidth[LEFT] + (localDim[X]-stencilDim[X])/2 - gpuHaloWidth[LEFT] +
                    i%(stencilDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]);
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuToGpuStencilBuffer[stencilNumPoints*(gpuHaloWidth[LEFT]*stencilDim[Y] + gpuHaloWidth[RIGHT]*stencilDim[Y]+gpuHaloWidth[TOP]*(stencilDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]) + i) + j].store(cpuStencil[stencilNumPoints*index + j]);
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

        int gTc = gpuHaloWidth[TOP]*gpuDim[X]+gpuHaloWidth[BOTTOM]*gpuDim[X]+
                  gpuHaloWidth[LEFT]*(gpuDim[Y]-(gpuHaloWidth[TOP]+gpuHaloWidth[BOTTOM]))+gpuHaloWidth[RIGHT]*(gpuDim[Y]-(gpuHaloWidth[TOP]+gpuHaloWidth[BOTTOM]));

        int globalSize = (gTc/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_collectHalo(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                           cl_gpuDim[X], cl_gpuDim[Y], cl_gpuHaloWidth, gpuData, cl_gpuToCpuDataBuffer);

        double *dat = new double[gTc];
        cl::copy(cl_queue, cl_gpuToCpuDataBuffer, &dat[0], (&dat[gTc-1])+1);
        for(int i = 0; i < gTc; ++i)
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

        int gTc = gpuHaloWidth[TOP]*stencilDim[X]+gpuHaloWidth[BOTTOM]*stencilDim[X]+
                  gpuHaloWidth[LEFT]*(stencilDim[Y]-(gpuHaloWidth[TOP]+gpuHaloWidth[BOTTOM]))+gpuHaloWidth[RIGHT]*(stencilDim[Y]-(gpuHaloWidth[TOP]+gpuHaloWidth[BOTTOM]));

        int globalSize = ((stencilNumPoints*gTc)/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_collectHalo(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                           cl_stencilDim[X], cl_stencilDim[Y], cl_gpuHaloWidth, gpuStencil, cl_stencilNumPoints, cl_gpuToCpuStencilBuffer);

        double *dat = new double[stencilNumPoints*gTc];
        cl::copy(cl_queue, cl_gpuToCpuStencilBuffer, &dat[0], (&dat[stencilNumPoints*gTc-1])+1);
        for(int i = 0; i < stencilNumPoints*gTc; ++i)
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
    for(int i = 0; i < gpuHaloWidth[LEFT]*(gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM]); ++ i) {
        int index = ((localDim[Y]-gpuDim[Y])/2 +i/gpuHaloWidth[LEFT]+cpuHaloWidth[BOTTOM]+gpuHaloWidth[BOTTOM])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    (localDim[X]-gpuDim[X])/2+i%gpuHaloWidth[LEFT] + cpuHaloWidth[LEFT];
        cpuData[index] = gpuToCpuDataBuffer[i].load();
    }
    int offset = gpuHaloWidth[LEFT]*(gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM]);
    // right
    for(int i = 0; i < gpuHaloWidth[RIGHT]*(gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM]); ++ i) {
        int index = ((localDim[Y]-gpuDim[Y])/2 +cpuHaloWidth[BOTTOM] + i/gpuHaloWidth[RIGHT] + gpuHaloWidth[BOTTOM])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    (localDim[X]-gpuDim[X])/2 + cpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT] + gpuDim[X] + i%gpuHaloWidth[RIGHT];
        cpuData[index] = gpuToCpuDataBuffer[offset + i].load();
    }
    offset += gpuHaloWidth[RIGHT]*(gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM]);
    // top
    for(int i = 0; i < gpuHaloWidth[TOP]*gpuDim[X]; ++ i) {
        int index = ((localDim[Y]-gpuDim[Y])/2+gpuDim[Y]+cpuHaloWidth[BOTTOM]-gpuHaloWidth[TOP] + i/gpuDim[X])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    cpuHaloWidth[LEFT] + (localDim[X]-gpuDim[X])/2 +i%gpuDim[X];
        cpuData[index] = gpuToCpuDataBuffer[offset + i].load();
    }
    offset += gpuHaloWidth[TOP]*gpuDim[X];
    // bottom
    for(int i = 0; i < gpuHaloWidth[BOTTOM]*gpuDim[X]; ++ i) {
        int index = (i/gpuDim[X] + cpuHaloWidth[BOTTOM] + (localDim[Y]-gpuDim[Y])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    cpuHaloWidth[LEFT] + (localDim[X]-gpuDim[X])/2 + i%gpuDim[X];
        cpuData[index] = gpuToCpuDataBuffer[offset + i].load();
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
    for(int i = 0; i < gpuHaloWidth[LEFT]*(stencilDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM]); ++ i) {
        int index = ((localDim[Y]-stencilDim[Y])/2 +i/gpuHaloWidth[LEFT]+cpuHaloWidth[BOTTOM]+gpuHaloWidth[BOTTOM])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    (localDim[X]-stencilDim[X])/2+i%gpuHaloWidth[LEFT] + cpuHaloWidth[LEFT];
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuStencil[stencilNumPoints*index + j] = gpuToCpuStencilBuffer[stencilNumPoints*i + j].load();
    }
    int offset = gpuHaloWidth[LEFT]*(stencilDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM]);
    // right
    for(int i = 0; i < gpuHaloWidth[RIGHT]*(stencilDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM]); ++ i) {
        int index = ((localDim[Y]-stencilDim[Y])/2 +cpuHaloWidth[BOTTOM] + i/gpuHaloWidth[RIGHT] + gpuHaloWidth[BOTTOM])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    (localDim[X]-stencilDim[X])/2 + cpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT] + stencilDim[X] + i%gpuHaloWidth[RIGHT];
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuStencil[stencilNumPoints*index + j] = gpuToCpuStencilBuffer[stencilNumPoints*(offset+i) + j].load();
    }
    offset += gpuHaloWidth[RIGHT]*(stencilDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM]);
    // top
    for(int i = 0; i < gpuHaloWidth[TOP]*stencilDim[X]; ++ i) {
        int index = ((localDim[Y]-stencilDim[Y])/2+stencilDim[Y]+cpuHaloWidth[BOTTOM]-gpuHaloWidth[TOP] + i/stencilDim[X])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    cpuHaloWidth[LEFT] + (localDim[X]-stencilDim[X])/2 +i%stencilDim[X];
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuStencil[stencilNumPoints*index + j] = gpuToCpuStencilBuffer[stencilNumPoints*(offset + i) + j].load();
    }
    offset += gpuHaloWidth[TOP]*stencilDim[X];
    // bottom
    for(int i = 0; i < gpuHaloWidth[BOTTOM]*stencilDim[X]; ++ i) {
        int index = (i/stencilDim[X] + cpuHaloWidth[BOTTOM] + (localDim[Y]-stencilDim[Y])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    cpuHaloWidth[LEFT] + (localDim[X]-stencilDim[X])/2 + i%stencilDim[X];
        for(int j = 0; j < stencilNumPoints; ++j)
            cpuStencil[stencilNumPoints*index + j] = gpuToCpuStencilBuffer[stencilNumPoints*(offset + i) + j].load();
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

        int cTg = gpuHaloWidth[TOP]*(stencilDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]) +
                  gpuHaloWidth[BOTTOM]*(stencilDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]) +
                  gpuHaloWidth[LEFT]*stencilDim[Y] + gpuHaloWidth[RIGHT]*stencilDim[Y];

        double *dat = new double[cTg];
        for(int i = 0; i < cTg; ++i)
            dat[i] = cpuToGpuDataBuffer[i].load();

        cl::copy(cl_queue, &dat[0], (&dat[cTg-1])+1, cl_cpuToGpuDataBuffer);

        delete[] dat;

        auto kernel_distributeHaloData = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(cl_programs, "distributeHaloData");

        int globalSize = (cTg/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_distributeHaloData(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                                  cl_stencilDim[X], cl_stencilDim[Y], cl_gpuHaloWidth, gpuData, cl_cpuToGpuDataBuffer);

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

        int cTg = gpuHaloWidth[TOP]*(stencilDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]) +
                  gpuHaloWidth[BOTTOM]*(stencilDim[X]+gpuHaloWidth[LEFT]+gpuHaloWidth[RIGHT]) +
                  gpuHaloWidth[LEFT]*stencilDim[Y] + gpuHaloWidth[RIGHT]*stencilDim[Y];

        double *dat = new double[stencilNumPoints*cTg];
        for(int i = 0; i < stencilNumPoints*cTg; ++i)
            dat[i] = cpuToGpuStencilBuffer[i].load();

        cl::copy(cl_queue, &dat[0], (&dat[stencilNumPoints*cTg-1])+1, cl_cpuToGpuStencilBuffer);

        delete[] dat;

        auto kernel_distributeHaloData = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(cl_programs, "distributeHaloStencil");

        int globalSize = ((stencilNumPoints*cTg)/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_distributeHaloData(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                                  cl_stencilDim[X], cl_stencilDim[Y], cl_gpuHaloWidth, gpuStencil, cl_stencilNumPoints, cl_cpuToGpuStencilBuffer);

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
enum { LEFT, RIGHT, TOP, BOTTOM };
kernel void collectHaloData(global const int * restrict const dimX, global const int * restrict const dimY,
                            global const int * restrict const haloWidth,
                            global const real_t * restrict const vec, global real_t * sync) {
    int currentIndex = get_global_id(0);
    int current = currentIndex;
    int maxNum = haloWidth[TOP]*(*dimX) + haloWidth[BOTTOM]*(*dimX) +
    haloWidth[LEFT]*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM]) + haloWidth[RIGHT]*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM]);
    if(current >= maxNum)
        return;
    // left
    if(current < haloWidth[LEFT]*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])) {
        int index = (2*haloWidth[BOTTOM]+current/haloWidth[LEFT])*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +haloWidth[LEFT] +
        current%haloWidth[LEFT];
        sync[currentIndex] = vec[index];
        return;
    }
    current -= haloWidth[LEFT]*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM]);
    // right
    if(current < haloWidth[RIGHT]*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])) {
        int index = (1+2*haloWidth[BOTTOM]+current/haloWidth[RIGHT])*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) -2*haloWidth[RIGHT]+
        current%haloWidth[RIGHT];
        sync[currentIndex] = vec[index];
        return;
    }
    current -= haloWidth[RIGHT]*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM]);
    // top
    if(current < haloWidth[TOP]*(*dimX)) {
        int index = (*dimY+haloWidth[BOTTOM]-haloWidth[TOP] + current/(*dimX))*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
        current%(*dimX)+haloWidth[LEFT];
        sync[currentIndex] = vec[index];
        return;
    }
    current -= haloWidth[TOP]*(*dimX);
    // bottom
    int index = (haloWidth[BOTTOM]+current/(*dimX))*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT])+current%(*dimX)+haloWidth[LEFT];
    sync[currentIndex] = vec[index];
}
kernel void collectHaloStencil(global const int * restrict const dimX, global const int * restrict const dimY,
                               global const int * restrict const haloWidth, global const real_t * restrict const vec,
                               global const int * restrict const stencilNumPoints, global real_t * sync) {
    int currentIndex = get_global_id(0);
    int current = currentIndex;
    int maxNum = (*stencilNumPoints)*(haloWidth[TOP]*(*dimX) + haloWidth[BOTTOM]*(*dimX) +
    haloWidth[LEFT]*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM]) + haloWidth[RIGHT]*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM]));
    if(current >= maxNum)
        return;
    // left
    if(current < (*stencilNumPoints)*haloWidth[LEFT]*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])) {
        int current_stencil = current%(*stencilNumPoints);
        int current_index = current/(*stencilNumPoints);
        int index = (*stencilNumPoints)*((2*haloWidth[BOTTOM]+current_index/haloWidth[LEFT])*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
        haloWidth[LEFT] + current_index%haloWidth[LEFT]) + current_stencil;
        sync[currentIndex] = vec[index];
        return;
    }
    current -= (*stencilNumPoints)*haloWidth[LEFT]*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM]);
    // right
    if(current < (*stencilNumPoints)*haloWidth[RIGHT]*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])) {
        int current_stencil = current%(*stencilNumPoints);
        int current_index = current/(*stencilNumPoints);
        int index = (*stencilNumPoints)*((1+2*haloWidth[BOTTOM]+current_index/haloWidth[RIGHT])*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) -
        2*haloWidth[RIGHT]+current_index%haloWidth[RIGHT]) + current_stencil;
        sync[currentIndex] = vec[index];
        return;
    }
    current -= (*stencilNumPoints)*haloWidth[LEFT]*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM]);
    // top
    if(current < (*stencilNumPoints)*haloWidth[BOTTOM]*(*dimX)) {
        int current_stencil = current%(*stencilNumPoints);
        int current_index = current/(*stencilNumPoints);
        int index = (*stencilNumPoints)*((*dimY+haloWidth[BOTTOM]-haloWidth[TOP] + current_index/(*dimX))*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
        current_index%(*dimX)+haloWidth[LEFT]) + current_stencil;
        sync[currentIndex] = vec[index];
        return;
    }
    // bottom
    current -= (*stencilNumPoints)*haloWidth[BOTTOM]*(*dimX);
    int current_stencil = current%(*stencilNumPoints);
    int current_index = current/(*stencilNumPoints);
    int index = (*stencilNumPoints)*((haloWidth[BOTTOM]+current_index/(*dimX))*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT])+
    current_index%(*dimX)+haloWidth[LEFT]) + current_stencil;
    sync[currentIndex] = vec[index];
}
kernel void distributeHaloData(global const int * restrict const dimX, global const int * restrict const dimY,
                               global const int * restrict const haloWidth,
                               global real_t * vec, global const real_t * restrict const sync) {
    int syncIndex = get_global_id(0);
    int current = syncIndex;
    int maxNum = haloWidth[TOP]*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) + haloWidth[BOTTOM]*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
    haloWidth[LEFT]*(*dimY) + haloWidth[RIGHT]*(*dimY);
    if(current >= maxNum)
        return;
    // left
    if(current < haloWidth[LEFT]*(*dimY)) {
        int index = (haloWidth[BOTTOM]+current/haloWidth[LEFT])*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) + current%haloWidth[LEFT];
        vec[index] = sync[syncIndex];
        return;
    }
    current -= haloWidth[LEFT]*(*dimY);
    // right
    if(current < haloWidth[RIGHT]*(*dimY)) {
        int index = (haloWidth[BOTTOM]+1+current/haloWidth[RIGHT])*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) -
        haloWidth[RIGHT]+current%haloWidth[RIGHT];
        vec[index] = sync[syncIndex];
        return;
    }
    current -= haloWidth[RIGHT]*(*dimY);
    // top
    if(current < haloWidth[TOP]*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT])) {
        int index = (*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[BOTTOM] + current/(*dimX+haloWidth[LEFT]+haloWidth[RIGHT])) +
        current%(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]);
        vec[index] = sync[syncIndex];
        return;
    }
    current -= haloWidth[TOP]*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]);
    // bottom
    int index = (current/(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]))*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT])+
    current%(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]);
    vec[index] = sync[syncIndex];
}
kernel void distributeHaloStencil(global const int * restrict const dimX, global const int * restrict const dimY,
                                  global const int * restrict const haloWidth, global real_t * vec,
                                  global const int * restrict const stencilNumPoints, global const real_t * restrict const sync) {
    int syncIndex = get_global_id(0);
    int current = syncIndex;
    int maxNum = (*stencilNumPoints)*(haloWidth[TOP]*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
    haloWidth[BOTTOM]*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
    haloWidth[LEFT]*(*dimY) + haloWidth[RIGHT]*(*dimY));
    if(current >= maxNum)
        return;
    // left
    if(current < (*stencilNumPoints)*haloWidth[LEFT]*(*dimY)) {
        int currentStencil = current%(*stencilNumPoints);
        int currentIndex = current/(*stencilNumPoints);
        int index = (*stencilNumPoints)*((haloWidth[BOTTOM]+currentIndex/haloWidth[LEFT])*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
        currentIndex%haloWidth[LEFT]) + currentStencil;
        vec[index] = sync[syncIndex];
        return;
    }
    current -= (*stencilNumPoints)*haloWidth[LEFT]*(*dimY);
    // right
    if(current < (*stencilNumPoints)*haloWidth[RIGHT]*(*dimY)) {
        int currentStencil = current%(*stencilNumPoints);
        int currentIndex = current/(*stencilNumPoints);
        int index = (*stencilNumPoints)*((haloWidth[BOTTOM]+1+currentIndex/haloWidth[RIGHT])*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) -
        haloWidth[RIGHT]+currentIndex%haloWidth[RIGHT]) + currentStencil;
        vec[index] = sync[syncIndex];
        return;
    }
    current -= (*stencilNumPoints)*haloWidth[RIGHT]*(*dimY);
    // top
    if(current < (*stencilNumPoints)*haloWidth[TOP]*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT])) {
        int currentStencil = current%(*stencilNumPoints);
        int currentIndex = current/(*stencilNumPoints);
        int index = (*stencilNumPoints)*((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[BOTTOM] +
        currentIndex/(*dimX+haloWidth[LEFT]+haloWidth[RIGHT])) + currentIndex%(*dimX+haloWidth[LEFT]+haloWidth[RIGHT])) +
        currentStencil;
        vec[index] = sync[syncIndex];
        return;
    }
    current -= (*stencilNumPoints)*haloWidth[TOP]*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]);
    // bottom
    int currentStencil = current%(*stencilNumPoints);
    int currentIndex = current/(*stencilNumPoints);
    int index = (*stencilNumPoints)*((currentIndex/(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]))*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT])+
    currentIndex%(*dimX+haloWidth[LEFT]+haloWidth[RIGHT])) + currentStencil;
    vec[index] = sync[syncIndex];
}
        )d";

    try {
        cl_programs = cl::Program(cl_context, oclstr, true);
    } catch(cl::Error error) {
        std::cout << "[kernel compile] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        if(error.err() == -11) {
            std::string log = cl_programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_defaultDevice);
            std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl
                      << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** "
                      << std::endl << std::endl;
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
