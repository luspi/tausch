#include "tausch3d.h"

Tausch3D::Tausch3D(int *localDim, int *mpiNum, int *cpuHaloWidth, MPI_Comm comm) {

    MPI_Comm_dup(comm, &TAUSCH_COMM);

    // get MPI info
    MPI_Comm_rank(TAUSCH_COMM, &mpiRank);
    MPI_Comm_size(TAUSCH_COMM, &mpiSize);

    this->mpiNum[X] = mpiNum[X];
    this->mpiNum[Y] = mpiNum[Y];
    this->mpiNum[Z] = mpiNum[Z];

    // store configuration
    this->localDim[X] = localDim[X];
    this->localDim[Y] = localDim[Y];
    this->localDim[Z] = localDim[Z];

    for(int i = 0; i < 6; ++i)
        this->cpuHaloWidth[i] = cpuHaloWidth[i];

    // check if this rank has a boundary with another rank
    haveBoundary[LEFT] = (mpiRank%mpiNum[X] != 0);
    haveBoundary[RIGHT] = ((mpiRank+1)%mpiNum[X] != 0);
    haveBoundary[TOP] = (mpiRank%(mpiNum[X]*mpiNum[Y]) < (mpiNum[X]*mpiNum[Y]-mpiNum[X]));
    haveBoundary[BOTTOM] = (mpiRank%(mpiNum[X]*mpiNum[Y]) > (mpiNum[X]-1));
    haveBoundary[FRONT] = (mpiRank > mpiNum[X]*mpiNum[Y]-1);
    haveBoundary[BACK] = mpiRank < mpiSize-mpiNum[X]*mpiNum[Y];

    // whether the cpu/gpu pointers have been passed
    cpuInfoGiven = false;
    stencilInfoGiven = false;

    // cpu at beginning
    cpuRecvsPosted = false;
    stencilRecvsPosted = false;

    // communication to neither edge has been started
    for(int i = 0; i < 6; ++i)
        cpuStarted[i] = false;
    for(int i = 0; i < 6; ++i)
        cpuStencilStarted[i] = false;

    mpiDataType = ((sizeof(real_t) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT);

#ifdef TAUSCH_OPENCL

    gpuInfoGiven = false;
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

Tausch3D::~Tausch3D() {

    // clean up memory

    if(cpuInfoGiven) {
        for(int i = 0; i < 6; ++i) {
            if(haveBoundary[i]) {
                delete[] cpuToCpuSendBuffer[i];
                delete[] cpuToCpuRecvBuffer[i];
            }
        }
        delete[] cpuToCpuSendBuffer;
        delete[] cpuToCpuRecvBuffer;
    }
    if(stencilInfoGiven) {
        for(int i = 0; i < 6; ++i) {
            if(haveBoundary[i]) {
                delete[] cpuToCpuStencilSendBuffer[i];
                delete[] cpuToCpuStencilRecvBuffer[i];
            }
        }
        delete[] cpuToCpuStencilSendBuffer;
        delete[] cpuToCpuStencilRecvBuffer;
    }
#ifdef TAUSCH_OPENCL
    if(gpuEnabled) {
        if(gpuInfoGiven) {
            delete[] cpuToGpuDataBuffer;
            delete[] gpuToCpuDataBuffer;
        }
    }
#endif

}

// get a pointer to the CPU data
void Tausch3D::setCpuData(real_t *dat) {

    cpuInfoGiven = true;
    cpuData = dat;

    int sendBufferSizes[6] = {
        // left
        cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // right
        cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // top
        cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // bottom
        cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // front
        cpuHaloWidth[BACK]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]),
        // back
        cpuHaloWidth[FRONT]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])
    };

    int recvBufferSizes[6] = {
        // left
        cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // right
        cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // top
        cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // bottom
        cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // front
        cpuHaloWidth[FRONT]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]),
        // back
        cpuHaloWidth[BACK]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])
    };

    int ranks[6] = {mpiRank-1, mpiRank+1, mpiRank+mpiNum[X], mpiRank-mpiNum[X], mpiRank-mpiNum[X]*mpiNum[Y], mpiRank+mpiNum[X]*mpiNum[Y]};
    int recvTags[6] = {0, 2, 1, 3, 4, 5};
    int sendTags[6] = {2, 0, 3, 1, 5, 4};

    // six buffers for each of the MPI send/recv operations
    cpuToCpuSendBuffer = new real_t*[6];
    cpuToCpuRecvBuffer = new real_t*[6];

    for(int i = 0; i < 6; ++i) {

        if(haveBoundary[i]) {

            cpuToCpuSendBuffer[i] = new real_t[sendBufferSizes[i]]{};
            cpuToCpuRecvBuffer[i] = new real_t[recvBufferSizes[i]]{};

            MPI_Recv_init(cpuToCpuRecvBuffer[i], recvBufferSizes[i], mpiDataType, ranks[i], recvTags[i], TAUSCH_COMM, &cpuToCpuRecvRequest[i]);
            MPI_Send_init(cpuToCpuSendBuffer[i], sendBufferSizes[i], mpiDataType, ranks[i], sendTags[i], TAUSCH_COMM, &cpuToCpuSendRequest[i]);

        }
    }

}

// get a pointer to the CPU data
void Tausch3D::setCpuStencil(real_t *stencil, int stencilNumPoints) {

    stencilInfoGiven = true;
    cpuStencil = stencil;
    this->stencilNumPoints = stencilNumPoints;

    int sendBufferSizes[6] = {
        // left
        cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // right
        cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // top
        cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // bottom
        cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // front
        cpuHaloWidth[BACK]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]),
        // back
        cpuHaloWidth[FRONT]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])
    };

    int recvBufferSizes[6] = {
        // left
        cpuHaloWidth[LEFT]*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // right
        cpuHaloWidth[RIGHT]*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // top
        cpuHaloWidth[TOP]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // bottom
        cpuHaloWidth[BOTTOM]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
        // front
        cpuHaloWidth[FRONT]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]),
        // back
        cpuHaloWidth[BACK]*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])
    };

    int ranks[6] = {mpiRank-1, mpiRank+1, mpiRank+mpiNum[X], mpiRank-mpiNum[X], mpiRank-mpiNum[X]*mpiNum[Y], mpiRank+mpiNum[X]*mpiNum[Y]};
    int recvTags[6] = {0, 2, 1, 3, 4, 5};
    int sendTags[6] = {2, 0, 3, 1, 5, 4};

    // six buffers for each edge's MPI send/recv operations
    cpuToCpuStencilSendBuffer = new real_t*[6];
    cpuToCpuStencilRecvBuffer = new real_t*[6];

    for(int i = 0; i < 6; ++i) {

        if(haveBoundary[i]) {

            cpuToCpuStencilSendBuffer[i] = new real_t[stencilNumPoints*sendBufferSizes[i]]{};
            cpuToCpuStencilRecvBuffer[i] = new real_t[stencilNumPoints*recvBufferSizes[i]]{};

            MPI_Recv_init(cpuToCpuStencilRecvBuffer[i], stencilNumPoints*recvBufferSizes[i], mpiDataType,
                          ranks[i], recvTags[i], TAUSCH_COMM, &cpuToCpuStencilRecvRequest[i]);
            MPI_Send_init(cpuToCpuStencilSendBuffer[i], stencilNumPoints*sendBufferSizes[i], mpiDataType,
                          ranks[i], sendTags[i], TAUSCH_COMM, &cpuToCpuStencilSendRequest[i]);

        }

    }
}

// post the MPI_Irecv's for inter-rank communication
void Tausch3D::postCpuDataReceives() {

    if(!cpuInfoGiven) {
        std::cerr << "ERROR: You didn't tell me yet where to find the data! Abort..." << std::endl;
        exit(1);
    }

    cpuRecvsPosted = true;

    for(int i = 0; i < 6; ++i)
        if(haveBoundary[i])
            MPI_Start(&cpuToCpuRecvRequest[i]);

}

void Tausch3D::postCpuStencilReceives() {

    if(!stencilInfoGiven) {
        std::cerr << "ERROR: You didn't tell me yet where to find the stencil data! Abort..." << std::endl;
        exit(1);
    }

    stencilRecvsPosted = true;

    for(int i = 0; i < 6; ++i)
        if(haveBoundary[i])
            MPI_Start(&cpuToCpuStencilRecvRequest[i]);

}

void Tausch3D::startCpuDataEdge(Edge edge) {

    if(!cpuRecvsPosted) {
        std::cerr << "ERROR: No CPU Recvs have been posted yet... Abort!" << std::endl;
        exit(1);
    }

    if(edge != LEFT && edge != RIGHT && edge != TOP && edge != BOTTOM && edge != FRONT && edge != BACK) {
        std::cerr << "startCpuDataEdge(): ERROR: Invalid edge specified: " << edge << std::endl;
        exit(1);
    }

    cpuStarted[edge] = true;

    if(edge == LEFT && haveBoundary[LEFT]) {

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < cpuHaloWidth[RIGHT]; ++x)

                    cpuToCpuSendBuffer[LEFT][z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*cpuHaloWidth[RIGHT]+y*cpuHaloWidth[RIGHT] + x]
                            = cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                      y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x+cpuHaloWidth[LEFT]];

        MPI_Start(&cpuToCpuSendRequest[LEFT]);

    } else if(edge == RIGHT && haveBoundary[RIGHT]) {

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < cpuHaloWidth[LEFT]; ++x)

                    cpuToCpuSendBuffer[RIGHT][z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*cpuHaloWidth[LEFT]+y*cpuHaloWidth[LEFT] + x]
                            = cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                      y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +localDim[X]+x];

        MPI_Start(&cpuToCpuSendRequest[RIGHT]);

    } else if(edge == TOP && haveBoundary[TOP]) {

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < cpuHaloWidth[BOTTOM]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    cpuToCpuSendBuffer[TOP][z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*cpuHaloWidth[BOTTOM]+
                                            y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x]
                            = cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                     (y+localDim[Y])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x];

        MPI_Start(&cpuToCpuSendRequest[TOP]);

    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    cpuToCpuSendBuffer[BOTTOM][z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*cpuHaloWidth[TOP]+
                                               y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x]
                            = cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                     (y+cpuHaloWidth[BOTTOM])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x];

        MPI_Start(&cpuToCpuSendRequest[BOTTOM]);

    } else if(edge == FRONT && haveBoundary[FRONT]) {

        for(int z = 0; z < cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    cpuToCpuSendBuffer[FRONT][z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                                (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                                              y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x]
                            = cpuData[(z+cpuHaloWidth[FRONT])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                     (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) + y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x];

        MPI_Start(&cpuToCpuSendRequest[FRONT]);

    } else if(edge == BACK && haveBoundary[BACK]) {

        for(int z = 0; z < cpuHaloWidth[FRONT]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    cpuToCpuSendBuffer[BACK][z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                               (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                                             y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x]
                            = cpuData[(z+localDim[Z])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                      (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                       y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x];

        MPI_Start(&cpuToCpuSendRequest[BACK]);

    }
}

void Tausch3D::startCpuStencilEdge(Edge edge) {

    if(!stencilRecvsPosted) {
        std::cerr << "ERROR: No CPU stencil Recvs have been posted yet... Abort!" << std::endl;
        exit(1);
    }

    if(edge != LEFT && edge != RIGHT && edge != TOP && edge != BOTTOM && edge != FRONT && edge != BACK) {
        std::cerr << "startCpuStencilEdge(): ERROR: Invalid edge specified: " << edge << std::endl;
        exit(1);
    }

    cpuStencilStarted[edge] = true;

    mpiDataType = ((sizeof(real_t) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT);

    if(edge == LEFT && haveBoundary[LEFT]) {

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < cpuHaloWidth[RIGHT]; ++x)

                    for(int st = 0; st < stencilNumPoints; ++st)

                        cpuToCpuStencilSendBuffer[LEFT][stencilNumPoints*(z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*cpuHaloWidth[RIGHT]+
                                                                          y*cpuHaloWidth[RIGHT] + x) + st]
                                = cpuStencil[stencilNumPoints*(z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                               (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                                              y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x+cpuHaloWidth[LEFT]) + st];

        MPI_Start(&cpuToCpuStencilSendRequest[LEFT]);

    } else if(edge == RIGHT && haveBoundary[RIGHT]) {

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < cpuHaloWidth[LEFT]; ++x)

                    for(int st = 0; st < stencilNumPoints; ++st)

                        cpuToCpuStencilSendBuffer[RIGHT][stencilNumPoints*(z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*cpuHaloWidth[LEFT]+
                                                                           y*cpuHaloWidth[LEFT] + x) + st]
                                = cpuStencil[stencilNumPoints*(z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                               (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                                               y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + localDim[X]+x) + st];

        MPI_Start(&cpuToCpuStencilSendRequest[RIGHT]);

    } else if(edge == TOP && haveBoundary[TOP]) {

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < cpuHaloWidth[BOTTOM]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    for(int st = 0; st < stencilNumPoints; ++st)

                        cpuToCpuStencilSendBuffer[TOP][stencilNumPoints*(z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*cpuHaloWidth[BOTTOM]+
                                                                         y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x) + st]
                                = cpuStencil[stencilNumPoints*(z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                               (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                                               (y+localDim[Y])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x) + st];

        MPI_Start(&cpuToCpuStencilSendRequest[TOP]);

    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    for(int st = 0; st < stencilNumPoints; ++st)

                        cpuToCpuStencilSendBuffer[BOTTOM][stencilNumPoints*(z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*cpuHaloWidth[TOP]+
                                                                            y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x) + st]
                                = cpuStencil[stencilNumPoints*(z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                               (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                                               (y+cpuHaloWidth[BOTTOM])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x) + st];

        MPI_Start(&cpuToCpuStencilSendRequest[BOTTOM]);

    } else if(edge == FRONT && haveBoundary[FRONT]) {

        for(int z = 0; z < cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    for(int st = 0; st < stencilNumPoints; ++st)

                        cpuToCpuStencilSendBuffer[FRONT][stencilNumPoints*(z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                                                           (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                                                                           y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x) + st]
                                = cpuStencil[stencilNumPoints*((z+cpuHaloWidth[FRONT])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                               (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                                               y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x) + st];

        MPI_Start(&cpuToCpuStencilSendRequest[FRONT]);

    } else if(edge == BACK && haveBoundary[BACK]) {

        for(int z = 0; z < cpuHaloWidth[FRONT]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    for(int st = 0; st < stencilNumPoints; ++st)

                        cpuToCpuStencilSendBuffer[BACK][stencilNumPoints*(z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                                                          (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                                                                          y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x) + st]
                                = cpuStencil[stencilNumPoints*((z+localDim[Z])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                               (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                                               y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x) + st];

        MPI_Start(&cpuToCpuStencilSendRequest[BACK]);

    }
}

// Complete CPU-CPU exchange to the left
void Tausch3D::completeCpuDataEdge(Edge edge) {

    if(edge != LEFT && edge != RIGHT && edge != TOP && edge != BOTTOM && edge != FRONT && edge != BACK) {
        std::cerr << "completeCpuDataEdge(): ERROR: Invalid edge specified: " << edge << std::endl;
        exit(1);
    }

    if(!cpuStarted[edge]) {
        std::cerr << "ERROR: No edge #" << edge << " CPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    if(edge == LEFT && haveBoundary[LEFT]) {

        MPI_Wait(&cpuToCpuRecvRequest[LEFT], MPI_STATUS_IGNORE);

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < cpuHaloWidth[LEFT]; ++x)

                    cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                            y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x]
                            = cpuToCpuRecvBuffer[LEFT][z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*cpuHaloWidth[LEFT] +
                                                       y*cpuHaloWidth[LEFT] + x];

        MPI_Wait(&cpuToCpuSendRequest[LEFT], MPI_STATUS_IGNORE);

    } else if(edge == RIGHT && haveBoundary[RIGHT]) {

        MPI_Wait(&cpuToCpuRecvRequest[RIGHT], MPI_STATUS_IGNORE);

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < cpuHaloWidth[RIGHT]; ++x)

                    cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                            y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + localDim[X]+cpuHaloWidth[LEFT]+x]
                            = cpuToCpuRecvBuffer[RIGHT][z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*cpuHaloWidth[RIGHT] +
                                                        y*cpuHaloWidth[RIGHT] + x];

        MPI_Wait(&cpuToCpuSendRequest[RIGHT], MPI_STATUS_IGNORE);

    } else if(edge == TOP && haveBoundary[TOP]) {

        MPI_Wait(&cpuToCpuRecvRequest[TOP], MPI_STATUS_IGNORE);

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                           (y+localDim[Y]+cpuHaloWidth[BOTTOM])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x]
                            = cpuToCpuRecvBuffer[TOP][z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*cpuHaloWidth[TOP] +
                                                      y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x];

        MPI_Wait(&cpuToCpuSendRequest[TOP], MPI_STATUS_IGNORE);

    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {

        MPI_Wait(&cpuToCpuRecvRequest[BOTTOM], MPI_STATUS_IGNORE);

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < cpuHaloWidth[BOTTOM]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                            y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x]
                            = cpuToCpuRecvBuffer[BOTTOM][z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*cpuHaloWidth[BOTTOM] +
                                                         y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x];

        MPI_Wait(&cpuToCpuSendRequest[BOTTOM], MPI_STATUS_IGNORE);

    } else if(edge == FRONT && haveBoundary[FRONT]) {

        MPI_Wait(&cpuToCpuRecvRequest[FRONT], MPI_STATUS_IGNORE);

        for(int z = 0; z < cpuHaloWidth[FRONT]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                            y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x]
                            = cpuToCpuRecvBuffer[FRONT][z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                                          (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                                        y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x];

        MPI_Wait(&cpuToCpuSendRequest[FRONT], MPI_STATUS_IGNORE);

    } else if(edge == BACK && haveBoundary[BACK]) {

        MPI_Wait(&cpuToCpuRecvRequest[BACK], MPI_STATUS_IGNORE);

        for(int z = 0; z < cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    cpuData[(z+cpuHaloWidth[FRONT]+localDim[Z])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                                (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                             y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x]
                            = cpuToCpuRecvBuffer[BACK][z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                                         (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                                                       y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x];

        MPI_Wait(&cpuToCpuSendRequest[BACK], MPI_STATUS_IGNORE);

    }
}

// Complete CPU-CPU exchange to the left
void Tausch3D::completeCpuStencilEdge(Edge edge) {

    if(edge != LEFT && edge != RIGHT && edge != TOP && edge != BOTTOM && edge != FRONT && edge != BACK) {
        std::cerr << "completeCpuStencilEdge(): ERROR: Invalid edge specified: " << edge << std::endl;
        exit(1);
    }

    if(!cpuStencilStarted[edge]) {
        std::cerr << "ERROR: No edge #" << edge << " CPU stencil exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    if(edge == LEFT && haveBoundary[LEFT]) {

        MPI_Wait(&cpuToCpuStencilRecvRequest[LEFT], MPI_STATUS_IGNORE);

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < cpuHaloWidth[LEFT]; ++x)

                    for(int st = 0; st < stencilNumPoints; ++st)

                        cpuStencil[stencilNumPoints*(z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                     (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                                     y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x) + st]
                                = cpuToCpuStencilRecvBuffer[LEFT][stencilNumPoints*(z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                                                                    cpuHaloWidth[LEFT] + y*cpuHaloWidth[LEFT] + x) + st];

        MPI_Wait(&cpuToCpuStencilSendRequest[LEFT], MPI_STATUS_IGNORE);

    } else if(edge == RIGHT && haveBoundary[RIGHT]) {

        MPI_Wait(&cpuToCpuStencilRecvRequest[RIGHT], MPI_STATUS_IGNORE);

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < cpuHaloWidth[RIGHT]; ++x)

                    for(int st = 0; st < stencilNumPoints; ++st)

                        cpuStencil[stencilNumPoints*(z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                     (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                                     y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                                                     localDim[X]+cpuHaloWidth[LEFT]+x) + st]
                                = cpuToCpuStencilRecvBuffer[RIGHT][stencilNumPoints*(z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                                                                     cpuHaloWidth[RIGHT] + y*cpuHaloWidth[RIGHT] + x) + st];

        MPI_Wait(&cpuToCpuStencilSendRequest[RIGHT], MPI_STATUS_IGNORE);

    } else if(edge == TOP && haveBoundary[TOP]) {

        MPI_Wait(&cpuToCpuStencilRecvRequest[TOP], MPI_STATUS_IGNORE);

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    for(int st = 0; st < stencilNumPoints; ++st)

                        cpuStencil[stencilNumPoints*(z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                     (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                                     (y+localDim[Y]+cpuHaloWidth[BOTTOM])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x) +st]
                                = cpuToCpuStencilRecvBuffer[TOP][stencilNumPoints*(z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                                                   cpuHaloWidth[TOP] +
                                                                                   y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x) + st];

        MPI_Wait(&cpuToCpuStencilSendRequest[TOP], MPI_STATUS_IGNORE);

    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {

        MPI_Wait(&cpuToCpuStencilRecvRequest[BOTTOM], MPI_STATUS_IGNORE);

        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < cpuHaloWidth[BOTTOM]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    for(int st = 0; st < stencilNumPoints; ++st)

                        cpuStencil[stencilNumPoints*(z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                     (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                                     y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x) + st]
                                = cpuToCpuStencilRecvBuffer[BOTTOM][stencilNumPoints*(z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                                                      cpuHaloWidth[BOTTOM] +
                                                                                      y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +x) +st];

        MPI_Wait(&cpuToCpuStencilSendRequest[BOTTOM], MPI_STATUS_IGNORE);

    } else if(edge == FRONT && haveBoundary[FRONT]) {

        MPI_Wait(&cpuToCpuStencilRecvRequest[FRONT], MPI_STATUS_IGNORE);

        for(int z = 0; z < cpuHaloWidth[FRONT]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    for(int st = 0; st < stencilNumPoints; ++st)

                        cpuStencil[stencilNumPoints*(z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                     (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                                     y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x) + st]
                                = cpuToCpuStencilRecvBuffer[FRONT][stencilNumPoints*(z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                                                                     (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                                                                     y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x) +st];

        MPI_Wait(&cpuToCpuStencilSendRequest[FRONT], MPI_STATUS_IGNORE);

    } else if(edge == BACK && haveBoundary[BACK]) {

        MPI_Wait(&cpuToCpuStencilRecvRequest[BACK], MPI_STATUS_IGNORE);

        for(int z = 0; z < cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)

                    for(int st = 0; st < stencilNumPoints; ++st)

                        cpuStencil[stencilNumPoints*((z+cpuHaloWidth[FRONT]+localDim[Z])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                     (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                                     y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x) + st]
                                = cpuToCpuStencilRecvBuffer[BACK][stencilNumPoints*(z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                                                                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                                                                                    y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x) + st];

        MPI_Wait(&cpuToCpuStencilSendRequest[BACK], MPI_STATUS_IGNORE);

    }

}

#ifdef TAUSCH_OPENCL

void Tausch3D::enableOpenCL(int *gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {

    // gpu disabled by default, only enabled if flag is set
    gpuEnabled = true;
    // local workgroup size
    cl_kernelLocalSize = clLocalWorkgroupSize;
    this->blockingSyncCpuGpu = blockingSyncCpuGpu;
    // The CPU/GPU halo width
    for(int i = 0; i < 6; ++i)
        this->gpuHaloWidth[i] = gpuHaloWidth[i];
    // Tausch creates its own OpenCL environment
    this->setupOpenCL(giveOpenCLDeviceName);

    try {
        cl_gpuHaloWidth = cl::Buffer(cl_context, &gpuHaloWidth[0], (&gpuHaloWidth[5])+1, true);
    } catch(cl::Error error) {
        std::cout << "Tausch3D :: [setup gpuHaloWidth buffer] Error: "
                  << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

void Tausch3D::enableOpenCL(int gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {
    int useHaloWidth[6] = {gpuHaloWidth, gpuHaloWidth, gpuHaloWidth, gpuHaloWidth, gpuHaloWidth, gpuHaloWidth};
    enableOpenCL(useHaloWidth, blockingSyncCpuGpu, clLocalWorkgroupSize, giveOpenCLDeviceName);
}

// If Tausch didn't set up OpenCL, the user needs to pass some OpenCL variables
void Tausch3D::enableOpenCL(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue,
                            int *gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize) {

    this->cl_defaultDevice = cl_defaultDevice;
    this->cl_context = cl_context;
    this->cl_queue = cl_queue;
    this->blockingSyncCpuGpu = blockingSyncCpuGpu;
    this->cl_kernelLocalSize = clLocalWorkgroupSize;
    for(int i = 0; i < 6; ++i)
        this->gpuHaloWidth[i] = gpuHaloWidth[i];

    try {
        cl_gpuHaloWidth = cl::Buffer(cl_context, &gpuHaloWidth[0], (&gpuHaloWidth[5])+1, true);
    } catch(cl::Error error) {
        std::cout << "Tausch3D :: [setup gpuHaloWidth buffer] Error: "
                  << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    gpuEnabled = true;

    compileKernels();

}

void Tausch3D::enableOpenCL(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue,
                            int gpuHaloWidth, bool blockingSyncCpuGpu, int clLocalWorkgroupSize) {
    int useHaloWidth[6] = {gpuHaloWidth, gpuHaloWidth, gpuHaloWidth, gpuHaloWidth, gpuHaloWidth, gpuHaloWidth};
    enableOpenCL(cl_defaultDevice, cl_context, cl_queue, useHaloWidth, blockingSyncCpuGpu, clLocalWorkgroupSize);
}

// get a pointer to the GPU buffer and its dimensions
void Tausch3D::setGpuData(cl::Buffer &dat, int *gpuDim) {

    // check whether OpenCL has been set up
    if(!gpuEnabled) {
        std::cerr << "ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    gpuInfoGiven = true;

    // store parameters
    gpuData = dat;
    this->gpuDim[X] = gpuDim[X];
    this->gpuDim[Y] = gpuDim[Y];
    this->gpuDim[Z] = gpuDim[Z];

    // store buffer to store the GPU and the CPU part of the halo.
    // We do not need two buffers each, as each thread has direct access to both arrays, no communication necessary
    cTgData = gpuHaloWidth[LEFT]*(gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*(gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
              gpuHaloWidth[RIGHT]*(gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*(gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
              gpuHaloWidth[TOP]*gpuDim[X]*(gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
              gpuHaloWidth[BOTTOM]*gpuDim[X]*(gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
              gpuHaloWidth[FRONT]*gpuDim[X]*gpuDim[Y] +
              gpuHaloWidth[BACK]*gpuDim[X]*gpuDim[Y];
    gTcData = gpuHaloWidth[LEFT]*gpuDim[Y]*gpuDim[Z] +
              gpuHaloWidth[RIGHT]*gpuDim[Y]*gpuDim[Z] +
              gpuHaloWidth[TOP]*(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuDim[Z] +
              gpuHaloWidth[BOTTOM]*(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuDim[Z] +
              gpuHaloWidth[FRONT]*(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(gpuDim[Y]-gpuHaloWidth[BOTTOM]-gpuHaloWidth[TOP]) +
              gpuHaloWidth[BACK]*(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(gpuDim[Y]-gpuHaloWidth[BOTTOM]-gpuHaloWidth[TOP]);

    cpuToGpuDataBuffer = new std::atomic<real_t>[cTgData]{};
    gpuToCpuDataBuffer = new std::atomic<real_t>[gTcData]{};

    // set up buffers on device
    try {
        cl_cpuToGpuDataBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, cTgData*sizeof(real_t));
        cl_gpuToCpuDataBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, gTcData*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_cpuToGpuDataBuffer, 0, 0, cTgData*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_gpuToCpuDataBuffer, 0, 0, gTcData*sizeof(real_t));
        cl_gpuDim[X] = cl::Buffer(cl_context, &gpuDim[X], (&gpuDim[X])+1, true);
        cl_gpuDim[Y] = cl::Buffer(cl_context, &gpuDim[Y], (&gpuDim[Y])+1, true);
        cl_gpuDim[Z] = cl::Buffer(cl_context, &gpuDim[Z], (&gpuDim[Z])+1, true);
    } catch(cl::Error error) {
        std::cout << "[setup send/recv buffer] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

void Tausch3D::setGpuStencil(cl::Buffer &stencil, int stencilNumPoints, int *stencilDim) {

    // check whether OpenCL has been set up
    if(!gpuEnabled) {
        std::cerr << "ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    gpuStencilInfoGiven = true;

    // store parameters
    gpuStencil = stencil;
    this->stencilNumPoints = stencilNumPoints;
    this->stencilDim[X] = ((stencilDim==nullptr || stencilDim[X] == 0) ? gpuDim[X] : stencilDim[X]);
    this->stencilDim[Y] = ((stencilDim==nullptr || stencilDim[Y] == 0) ? gpuDim[Y] : stencilDim[Y]);
    this->stencilDim[Z] = ((stencilDim==nullptr || stencilDim[Z] == 0) ? gpuDim[Z] : stencilDim[Z]);
    stencilDim[X] = this->stencilDim[X];
    stencilDim[Y] = this->stencilDim[Y];
    stencilDim[Z] = this->stencilDim[Z];

    // store buffer to store the GPU and the CPU part of the halo.
    // We do not need two buffers each, as each thread has direct access to both arrays, no communication necessary
    cTgStencil = stencilNumPoints*(gpuHaloWidth[LEFT]*(stencilDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*
                                                      (stencilDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
                                   gpuHaloWidth[RIGHT]*(stencilDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*
                                                       (stencilDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
                                   gpuHaloWidth[TOP]*stencilDim[X]*(stencilDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
                                   gpuHaloWidth[BOTTOM]*stencilDim[X]*(stencilDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
                                   gpuHaloWidth[FRONT]*stencilDim[X]*stencilDim[Y] +
                                   gpuHaloWidth[BACK]*stencilDim[X]*stencilDim[Y]);
    gTcStencil = stencilNumPoints*(gpuHaloWidth[LEFT]*stencilDim[Y]*stencilDim[Z] +
                                   gpuHaloWidth[RIGHT]*stencilDim[Y]*stencilDim[Z] +
                                   gpuHaloWidth[TOP]*(stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*stencilDim[Z] +
                                   gpuHaloWidth[BOTTOM]*(stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*stencilDim[Z] +
                                   gpuHaloWidth[FRONT]*(stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*
                                                       (stencilDim[Y]-gpuHaloWidth[BOTTOM]-gpuHaloWidth[TOP]) +
                                   gpuHaloWidth[BACK]*(stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*
                                                      (stencilDim[Y]-gpuHaloWidth[BOTTOM]-gpuHaloWidth[TOP]));

    cpuToGpuStencilBuffer = new std::atomic<real_t>[cTgStencil]{};
    gpuToCpuStencilBuffer = new std::atomic<real_t>[gTcStencil]{};

    // set up buffers on device
    try {
        cl_cpuToGpuStencilBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, cTgStencil*sizeof(real_t));
        cl_gpuToCpuStencilBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, gTcStencil*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_cpuToGpuStencilBuffer, 0, 0, cTgStencil*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_gpuToCpuStencilBuffer, 0, 0, gTcStencil*sizeof(real_t));
        cl_stencilNumPoints = cl::Buffer(cl_context, &stencilNumPoints, (&stencilNumPoints)+1, true);
        cl_stencilDim[X] = cl::Buffer(cl_context, &stencilDim[X], (&stencilDim[X])+1, true);
        cl_stencilDim[Y] = cl::Buffer(cl_context, &stencilDim[Y], (&stencilDim[Y])+1, true);
        cl_stencilDim[Z] = cl::Buffer(cl_context, &stencilDim[Z], (&stencilDim[Z])+1, true);
    } catch(cl::Error error) {
        std::cout << "[setup send/recv buffer] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

// collect cpu side of cpu/gpu halo and store in buffer
void Tausch3D::startCpuToGpuData() {

    // check whether GPU is enabled
    if(!gpuEnabled) {
        std::cerr << "ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    cpuToGpuDataStarted.store(true);

   // left
    for(int z = 0; z < (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]); ++z) {
        for(int y = 0; y < (gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP]); ++y) {
            for(int x = 0; x < gpuHaloWidth[LEFT]; ++x) {

                int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-gpuDim[Z])/2-gpuHaloWidth[FRONT])*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                            (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-gpuDim[Y])/2-gpuHaloWidth[BOTTOM])*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                            x+cpuHaloWidth[LEFT]+(localDim[X]-gpuDim[X])/2 -gpuHaloWidth[LEFT];

                cpuToGpuDataBuffer[z*(gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*gpuHaloWidth[LEFT] +
                                   y*gpuHaloWidth[LEFT] + x].store(cpuData[index]);

            }
        }
    }
    int offset = (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK])*(gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*gpuHaloWidth[LEFT];
    // right
    for(int z = 0; z < (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]); ++z) {
        for(int y = 0; y < (gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP]); ++y) {
            for(int x = 0; x < gpuHaloWidth[RIGHT]; ++x) {

                int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-gpuDim[Z])/2-gpuHaloWidth[FRONT])*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                            (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-gpuDim[Y])/2-gpuHaloWidth[BOTTOM])*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                            x+cpuHaloWidth[LEFT]+(localDim[X]-gpuDim[X])/2+gpuDim[X];

                cpuToGpuDataBuffer[offset + z*(gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*gpuHaloWidth[RIGHT] +
                                            y*gpuHaloWidth[RIGHT] + x].store(cpuData[index]);

            }
        }
    }
    offset += (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK])*(gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*gpuHaloWidth[RIGHT];
    // top
    for(int z = 0; z < (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]); ++z) {
        for(int y = 0; y < gpuHaloWidth[TOP]; ++y) {
            for(int x = 0; x < gpuDim[X]; ++x) {

                int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-gpuDim[Z])/2-gpuHaloWidth[FRONT])*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                            (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-gpuDim[Y])/2+gpuDim[Y])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                            x+cpuHaloWidth[LEFT]+(localDim[X]-gpuDim[X])/2;

                cpuToGpuDataBuffer[offset + z*gpuHaloWidth[TOP]*gpuDim[X] + y*gpuDim[X] + x].store(cpuData[index]);

            }
        }
    }
    offset += (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK])*gpuHaloWidth[TOP]*gpuDim[X];
    // bottom
    for(int z = 0; z < (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]); ++z) {
        for(int y = 0; y < gpuHaloWidth[BOTTOM]; ++y) {
            for(int x = 0; x < gpuDim[X]; ++x) {

                int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-gpuDim[Z])/2-gpuHaloWidth[FRONT])*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                            (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-gpuDim[Y])/2-gpuHaloWidth[BOTTOM])*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                            x+cpuHaloWidth[LEFT]+(localDim[X]-gpuDim[X])/2;

                cpuToGpuDataBuffer[offset + z*gpuHaloWidth[BOTTOM]*gpuDim[X] + y*gpuDim[X] + x].store(cpuData[index]);

            }
        }
    }
    offset += (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK])*gpuHaloWidth[BOTTOM]*gpuDim[X];
    // front
    for(int z = 0; z < gpuHaloWidth[FRONT]; ++z) {
        for(int y = 0; y < gpuDim[Y]; ++y) {
            for(int x = 0; x < gpuDim[X]; ++x) {

                int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-gpuDim[Z])/2-gpuHaloWidth[FRONT])*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                            (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-gpuDim[Y])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                            x+cpuHaloWidth[LEFT]+(localDim[X]-gpuDim[X])/2;

                cpuToGpuDataBuffer[offset + z*gpuDim[Y]*gpuDim[X] + y*gpuDim[X] + x].store(cpuData[index]);

            }
        }
    }
    offset += gpuHaloWidth[FRONT]*gpuDim[Y]*gpuDim[X];
    // back
    for(int z = 0; z < gpuHaloWidth[BACK]; ++z) {
        for(int y = 0; y < gpuDim[Y]; ++y) {
            for(int x = 0; x < gpuDim[X]; ++x) {
                int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-gpuDim[Z])/2+gpuDim[Z])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                            (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                            (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-gpuDim[Y])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                            x+cpuHaloWidth[LEFT]+(localDim[X]-gpuDim[X])/2;

                cpuToGpuDataBuffer[offset + z*gpuDim[Y]*gpuDim[X] + y*gpuDim[X] + x].store(cpuData[index]);

            }
        }
    }

}

// collect cpu side of cpu/gpu halo and store in buffer
void Tausch3D::startCpuToGpuStencil() {

    // check whether GPU is enabled
    if(!gpuEnabled) {
        std::cerr << "ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }

    cpuToGpuStencilStarted.store(true);

   // left
    for(int z = 0; z < (stencilDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]); ++z) {
        for(int y = 0; y < (stencilDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP]); ++y) {
            for(int x = 0; x < gpuHaloWidth[LEFT]; ++x) {

                for(int st = 0; st < stencilNumPoints; ++st) {

                    int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-stencilDim[Z])/2-gpuHaloWidth[FRONT])*
                                (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                                (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-stencilDim[Y])/2-gpuHaloWidth[BOTTOM])*
                                (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                                x+cpuHaloWidth[LEFT]+(localDim[X]-stencilDim[X])/2 - gpuHaloWidth[LEFT];

                    cpuToGpuStencilBuffer[stencilNumPoints*(z*(stencilDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*gpuHaloWidth[LEFT] +
                                                            y*gpuHaloWidth[LEFT] + x) + st].store(cpuStencil[stencilNumPoints*index + st]);

                }
            }
        }
    }
    int offset = (stencilDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK])*
                        (stencilDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*
                         gpuHaloWidth[LEFT];
    // right
    for(int z = 0; z < (stencilDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]); ++z) {
        for(int y = 0; y < (stencilDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP]); ++y) {
            for(int x = 0; x < gpuHaloWidth[RIGHT]; ++x) {

                for(int st = 0; st < stencilNumPoints; ++st) {

                    int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-stencilDim[Z])/2-gpuHaloWidth[FRONT])*
                                (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                                (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-stencilDim[Y])/2-gpuHaloWidth[BOTTOM])*
                                (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                                x+cpuHaloWidth[LEFT]+(localDim[X]-stencilDim[X])/2+stencilDim[X];

                    cpuToGpuStencilBuffer[stencilNumPoints*(offset + z*(stencilDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*gpuHaloWidth[RIGHT] +
                                                                     y*gpuHaloWidth[RIGHT] + x) + st].store(cpuStencil[stencilNumPoints*index + st]);

                }
            }
        }
    }
    offset += (stencilDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK])*
                    (stencilDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*
                     gpuHaloWidth[RIGHT];
    // top
    for(int z = 0; z < (stencilDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]); ++z) {
        for(int y = 0; y < gpuHaloWidth[TOP]; ++y) {
            for(int x = 0; x < stencilDim[X]; ++x) {

                for(int st = 0; st < stencilNumPoints; ++st) {

                    int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-stencilDim[Z])/2-gpuHaloWidth[FRONT])*
                                (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                                (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-stencilDim[Y])/2+stencilDim[Y])*
                                (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                                x+cpuHaloWidth[LEFT]+(localDim[X]-stencilDim[X])/2;

                    cpuToGpuStencilBuffer[stencilNumPoints*(offset + z*gpuHaloWidth[TOP]*stencilDim[X] + y*stencilDim[X] + x) + st]
                                    .store(cpuStencil[stencilNumPoints*index + st]);

                }
            }
        }
    }
    offset += (stencilDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK])*gpuHaloWidth[TOP]*stencilDim[X];
    // bottom
    for(int z = 0; z < (stencilDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]); ++z) {
        for(int y = 0; y < gpuHaloWidth[BOTTOM]; ++y) {
            for(int x = 0; x < stencilDim[X]; ++x) {

                for(int st = 0; st < stencilNumPoints; ++st) {

                    int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-stencilDim[Z])/2-gpuHaloWidth[FRONT])*
                                (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                                (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-stencilDim[Y])/2-gpuHaloWidth[BOTTOM])*
                                (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                                x+cpuHaloWidth[LEFT]+(localDim[X]-stencilDim[X])/2;

                    cpuToGpuStencilBuffer[stencilNumPoints*(offset + z*gpuHaloWidth[BOTTOM]*stencilDim[X] + y*stencilDim[X] + x) + st]
                                    .store(cpuStencil[stencilNumPoints*index + st]);

                }
            }
        }
    }
    offset += (stencilDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK])*gpuHaloWidth[BOTTOM]*stencilDim[X];
    // front
    for(int z = 0; z < gpuHaloWidth[FRONT]; ++z) {
        for(int y = 0; y < stencilDim[Y]; ++y) {
            for(int x = 0; x < stencilDim[X]; ++x) {

                for(int st = 0; st < stencilNumPoints; ++st) {

                    int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-stencilDim[Z])/2-gpuHaloWidth[FRONT])*
                                (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                                (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-stencilDim[Y])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                                x+cpuHaloWidth[LEFT]+(localDim[X]-stencilDim[X])/2;

                    cpuToGpuStencilBuffer[stencilNumPoints*(offset + z*stencilDim[Y]*stencilDim[X] + y*stencilDim[X] + x) + st]
                                    .store(cpuStencil[stencilNumPoints*index + st]);

                }
            }
        }
    }
    offset += gpuHaloWidth[FRONT]*stencilDim[Y]*stencilDim[X];
    // back
    for(int z = 0; z < gpuHaloWidth[BACK]; ++z) {
        for(int y = 0; y < stencilDim[Y]; ++y) {
            for(int x = 0; x < stencilDim[X]; ++x) {

                for(int st = 0; st < stencilNumPoints; ++st) {

                    int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-stencilDim[Z])/2+stencilDim[Z])*
                                (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                                (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-stencilDim[Y])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                                x+cpuHaloWidth[LEFT]+(localDim[X]-stencilDim[X])/2;

                    cpuToGpuStencilBuffer[stencilNumPoints*(offset + z*stencilDim[Y]*stencilDim[X] + y*stencilDim[X] + x) + st]
                                    .store(cpuStencil[stencilNumPoints*index + st]);

                }
            }
        }
    }

}

// collect gpu side of cpu/gpu halo and download into buffer
void Tausch3D::startGpuToCpuData() {

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

    gpuToCpuDataStarted.store(true);

    try {

        auto kernel_collectHalo = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&,
                                                  cl::Buffer&>(cl_programs, "collectHaloData");

        int globalSize = (gTcData/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_collectHalo(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                           cl_gpuDim[X], cl_gpuDim[Y], cl_gpuDim[Z], cl_gpuHaloWidth, gpuData, cl_gpuToCpuDataBuffer);

        double *dat = new double[gTcData];
        cl::copy(cl_queue, cl_gpuToCpuDataBuffer, &dat[0], (&dat[gTcData-1])+1);

        for(int i = 0; i < gTcData; ++i)
            gpuToCpuDataBuffer[i].store(dat[i]);

        delete[] dat;

    } catch(cl::Error error) {
        std::cout << "[kernel collectHalo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

// collect gpu side of cpu/gpu halo and download into buffer
void Tausch3D::startGpuToCpuStencil() {

    // check whether GPU is enabled
    if(!gpuEnabled) {
        std::cerr << "ERROR: GPU flag not passed on when creating Tausch object! Abort..." << std::endl;
        exit(1);
    }
    // check whether GPU info was given
    if(!gpuStencilInfoGiven) {
        std::cerr << "ERROR: GPU info not available! Did you call setOpenCLInfo()? Abort..." << std::endl;
        exit(1);
    }

    gpuToCpuStencilStarted.store(true);

    try {

        auto kernel_collectHalo = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&,
                                                  cl::Buffer&>(cl_programs, "collectHaloStencil");

        int globalSize = (gTcStencil/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_collectHalo(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                           cl_stencilDim[X], cl_stencilDim[Y], cl_stencilDim[Z], cl_gpuHaloWidth,
                           gpuStencil, cl_stencilNumPoints, cl_gpuToCpuStencilBuffer);

        double *dat = new double[gTcStencil];
        cl::copy(cl_queue, cl_gpuToCpuStencilBuffer, &dat[0], (&dat[gTcStencil-1])+1);

        for(int i = 0; i < gTcStencil; ++i)
            gpuToCpuStencilBuffer[i].store(dat[i]);

        delete[] dat;

    } catch(cl::Error error) {
        std::cout << "[kernel collectHalo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

// Complete CPU side of CPU/GPU halo exchange
void Tausch3D::completeGpuToCpuData() {

    // we need to wait for the GPU thread to arrive here
    if(blockingSyncCpuGpu)
        syncCpuAndGpu(false);

    if(!cpuToGpuDataStarted.load()) {
        std::cerr << "ERROR: No CPU->GPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    // left
    for(int i = 0; i < gpuHaloWidth[LEFT]*gpuDim[Y]*gpuDim[Z]; ++i) {

        int index = ((i/(gpuHaloWidth[LEFT]*gpuDim[Y])) + cpuHaloWidth[FRONT] + (localDim[Z]-gpuDim[Z])/2)*
                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                    ((i%(gpuHaloWidth[LEFT]*gpuDim[Y]))/gpuHaloWidth[LEFT] + cpuHaloWidth[BOTTOM] + (localDim[Y]-gpuDim[Y])/2)*
                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                    (i%(gpuHaloWidth[LEFT]*gpuDim[Y]))%gpuHaloWidth[LEFT] + (localDim[X]-gpuDim[X])/2 + cpuHaloWidth[LEFT];

        cpuData[index] = gpuToCpuDataBuffer[i].load();

    }

    int offset = gpuHaloWidth[LEFT]*gpuDim[Y]*gpuDim[Z];

    // right
    for(int i = 0; i < gpuHaloWidth[RIGHT]*gpuDim[Y]*gpuDim[Z]; ++i) {

        int index = ((i/(gpuHaloWidth[RIGHT]*gpuDim[Y])) + cpuHaloWidth[FRONT] + (localDim[Z]-gpuDim[Z])/2)*
                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                    ((i%(gpuHaloWidth[RIGHT]*gpuDim[Y]))/gpuHaloWidth[RIGHT] + cpuHaloWidth[BOTTOM] + (localDim[Y]-gpuDim[Y])/2)*
                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                    (i%(gpuHaloWidth[RIGHT]*gpuDim[Y]))%gpuHaloWidth[RIGHT] + (localDim[X]-gpuDim[X])/2 +
                    gpuDim[X] + cpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT];

        cpuData[index] = gpuToCpuDataBuffer[offset+i].load();

    }

    offset += gpuHaloWidth[RIGHT]*gpuDim[Y]*gpuDim[Z];

    // top
    for(int i = 0; i < (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[TOP]*gpuDim[Z]; ++i) {

        int index = ((i/((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[TOP])) + cpuHaloWidth[FRONT] + (localDim[Z]-gpuDim[Z])/2)*
                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                    ((i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[TOP]))/(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) +
                     gpuDim[Y] + cpuHaloWidth[BOTTOM]-gpuHaloWidth[TOP] + (localDim[Y]-gpuDim[Y])/2)*
                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                    (i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[TOP]))%(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) +
                    (localDim[X]-gpuDim[X])/2 + gpuHaloWidth[LEFT]+cpuHaloWidth[LEFT];

        cpuData[index] = gpuToCpuDataBuffer[offset+i].load();

    }

    offset += (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[TOP]*gpuDim[Z];

    // bottom
    for(int i = 0; i < (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[BOTTOM]*gpuDim[Z]; ++i) {

        int index = ((i/((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[BOTTOM])) + cpuHaloWidth[FRONT] + (localDim[Z]-gpuDim[Z])/2)*
                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                    ((i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[BOTTOM]))/(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])+
                     cpuHaloWidth[BOTTOM] + (localDim[Y]-gpuDim[Y])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                    (i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[BOTTOM]))%(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) +
                    (localDim[X]-gpuDim[X])/2 + gpuHaloWidth[LEFT]+cpuHaloWidth[LEFT];

        cpuData[index] = gpuToCpuDataBuffer[offset+i].load();

    }

    offset += (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[BOTTOM]*gpuDim[Z];

    // front
    for(int i=0; i < (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])*gpuHaloWidth[FRONT]; ++i) {

        int index = ((i/((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM]))) +
                     cpuHaloWidth[FRONT] + (localDim[Z]-gpuDim[Z])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                      (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                    ((i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])))/
                     (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) + gpuHaloWidth[BOTTOM]+cpuHaloWidth[BOTTOM] + (localDim[Y]-gpuDim[Y])/2)*
                      (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                    (i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])))%
                    (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) + (localDim[X]-gpuDim[X])/2 + gpuHaloWidth[LEFT]+cpuHaloWidth[LEFT];

        cpuData[index] = gpuToCpuDataBuffer[offset+i].load();

    }

    offset += (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])*gpuHaloWidth[FRONT];

    // back
    for(int i = 0; i < (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])*gpuHaloWidth[BACK]; ++i){

        int index = ((i/((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM]))) +
                     gpuDim[Z] + cpuHaloWidth[FRONT]-gpuHaloWidth[BACK] + (localDim[Z]-gpuDim[Z])/2)*
                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                    ((i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])))/
                     (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) + gpuHaloWidth[BOTTOM]+cpuHaloWidth[BOTTOM] + (localDim[Y]-gpuDim[Y])/2)*
                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                    (i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])))%
                    (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) + (localDim[X]-gpuDim[X])/2 + gpuHaloWidth[LEFT]+cpuHaloWidth[LEFT];

        cpuData[index] = gpuToCpuDataBuffer[offset+i].load();

    }

}

// Complete CPU side of CPU/GPU halo exchange
void Tausch3D::completeGpuToCpuStencil() {

    // we need to wait for the GPU thread to arrive here
    if(blockingSyncCpuGpu)
        syncCpuAndGpu(false);

    if(!cpuToGpuStencilStarted.load()) {
        std::cerr << "ERROR: No CPU->GPU stencil exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    // left
    for(int i = 0; i < gpuHaloWidth[LEFT]*stencilDim[Y]*stencilDim[Z]; ++i) {

        for(int st = 0; st < stencilNumPoints; ++st) {

            int index = ((i/(gpuHaloWidth[LEFT]*stencilDim[Y])) + cpuHaloWidth[FRONT] + (localDim[Z]-stencilDim[Z])/2)*
                         (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                        ((i%(gpuHaloWidth[LEFT]*stencilDim[Y]))/gpuHaloWidth[LEFT] + cpuHaloWidth[BOTTOM] +
                        (localDim[Y]-stencilDim[Y])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                        (i%(gpuHaloWidth[LEFT]*stencilDim[Y]))%gpuHaloWidth[LEFT] + (localDim[X]-stencilDim[X])/2 + cpuHaloWidth[LEFT];

            cpuStencil[stencilNumPoints*index + st] = gpuToCpuStencilBuffer[stencilNumPoints*i + st].load();

        }
    }

    int offset = gpuHaloWidth[LEFT]*stencilDim[Y]*stencilDim[Z];

    // right
    for(int i = 0; i < gpuHaloWidth[RIGHT]*stencilDim[Y]*stencilDim[Z]; ++i) {

        for(int st = 0; st < stencilNumPoints; ++st) {

            int index = ((i/(gpuHaloWidth[RIGHT]*stencilDim[Y])) + cpuHaloWidth[FRONT] + (localDim[Z]-stencilDim[Z])/2)*
                         (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                        ((i%(gpuHaloWidth[RIGHT]*stencilDim[Y]))/gpuHaloWidth[RIGHT] + cpuHaloWidth[BOTTOM] +
                        (localDim[Y]-stencilDim[Y])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                        (i%(gpuHaloWidth[RIGHT]*stencilDim[Y]))%gpuHaloWidth[RIGHT] + (localDim[X]-stencilDim[X])/2 +
                        stencilDim[X] + cpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT];

            cpuStencil[stencilNumPoints*index + st] = gpuToCpuStencilBuffer[stencilNumPoints*(offset+i) + st].load();

        }
    }

    offset += gpuHaloWidth[RIGHT]*stencilDim[Y]*stencilDim[Z];

    // top
    for(int i = 0; i < (stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[TOP]*stencilDim[Z]; ++i) {

        for(int st = 0; st < stencilNumPoints; ++st) {

            int index = ((i/((stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[TOP])) + cpuHaloWidth[FRONT] +
                         (localDim[Z]-stencilDim[Z])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                        (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                        ((i%((stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[TOP]))/
                         (stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) + stencilDim[Y] + cpuHaloWidth[BOTTOM]-gpuHaloWidth[TOP] +
                         (localDim[Y]-stencilDim[Y])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                        (i%((stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[TOP]))%
                        (stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])+(localDim[X]-stencilDim[X])/2 + gpuHaloWidth[LEFT]+cpuHaloWidth[LEFT];

            cpuStencil[stencilNumPoints*index + st] = gpuToCpuStencilBuffer[stencilNumPoints*(offset+i) + st].load();

        }
    }

    offset += (stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[TOP]*stencilDim[Z];

    // bottom
    for(int i = 0; i < (stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[BOTTOM]*stencilDim[Z]; ++i) {

        for(int st = 0; st < stencilNumPoints; ++st) {

            int index = ((i/((stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[BOTTOM])) + cpuHaloWidth[FRONT] +
                         (localDim[Z]-stencilDim[Z])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                        (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                        ((i%((stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[BOTTOM]))/
                         (stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) + cpuHaloWidth[BOTTOM] +
                         (localDim[Y]-stencilDim[Y])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                        (i%((stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[BOTTOM]))%
                        (stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) + (localDim[X]-stencilDim[X])/2 +gpuHaloWidth[LEFT]+cpuHaloWidth[LEFT];

            cpuStencil[stencilNumPoints*index + st] = gpuToCpuStencilBuffer[stencilNumPoints*(offset+i) + st].load();

        }
    }

    offset += (stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[BOTTOM]*stencilDim[Z];

    // front
    for(int i = 0; i < (stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(stencilDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])*
                       gpuHaloWidth[FRONT]; ++i) {

        for(int st = 0; st < stencilNumPoints; ++st) {

            int index = ((i/((stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(stencilDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM]))) +
                         cpuHaloWidth[FRONT] + (localDim[Z]-stencilDim[Z])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                         (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                        ((i%((stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(stencilDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])))/
                         (stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) + gpuHaloWidth[BOTTOM]+cpuHaloWidth[BOTTOM] +
                        (localDim[Y]-stencilDim[Y])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +

                        (i%((stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(stencilDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])))%
                         (stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) + (localDim[X]-stencilDim[X])/2 + gpuHaloWidth[LEFT]+cpuHaloWidth[LEFT];

            cpuStencil[stencilNumPoints*index + st] = gpuToCpuStencilBuffer[stencilNumPoints*(offset+i) + st].load();

        }
    }

    offset += (stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(stencilDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])*gpuHaloWidth[FRONT];

    // back
    for(int i = 0; i < (stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(stencilDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])*
                       gpuHaloWidth[BACK]; ++i){

        for(int st = 0; st < stencilNumPoints; ++st) {

            int index = ((i/((stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(stencilDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM]))) +
                         stencilDim[Z] + cpuHaloWidth[FRONT]-gpuHaloWidth[BACK] + (localDim[Z]-stencilDim[Z])/2)*
                          (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +

                        ((i%((stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(stencilDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])))/
                         (stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) +
                        gpuHaloWidth[BOTTOM]+cpuHaloWidth[BOTTOM]+(localDim[Y]-stencilDim[Y])/2)*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+

                        (i%((stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*(stencilDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])))%
                        (stencilDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) + (localDim[X]-stencilDim[X])/2 +gpuHaloWidth[LEFT]+cpuHaloWidth[LEFT];

            cpuStencil[stencilNumPoints*index + st] = gpuToCpuStencilBuffer[stencilNumPoints*(offset+i) + st].load();

        }
    }

}

// Complete GPU side of CPU/GPU halo exchange
void Tausch3D::completeCpuToGpuData() {

    // we need to wait for the CPU thread to arrive here
    if(blockingSyncCpuGpu)
        syncCpuAndGpu(false);

    if(!gpuToCpuDataStarted.load()) {
        std::cerr << "ERROR: No GPU->CPU exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    try {

        double *dat = new double[cTgData];
        for(int i = 0; i < cTgData; ++i)
            dat[i] = cpuToGpuDataBuffer[i].load();

        cl::copy(cl_queue, &dat[0], (&dat[cTgData-1])+1, cl_cpuToGpuDataBuffer);

        delete[] dat;

        auto kernelDistributeHaloData = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&,
                                                        cl::Buffer&, cl::Buffer&>(cl_programs, "distributeHaloData");

        int globalSize = (cTgData/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernelDistributeHaloData(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                                 cl_gpuDim[X], cl_gpuDim[Y], cl_gpuDim[Z], cl_gpuHaloWidth, gpuData, cl_cpuToGpuDataBuffer);

    } catch(cl::Error error) {
        std::cout << "[dist halo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }


}

// Complete GPU side of CPU/GPU halo exchange
void Tausch3D::completeCpuToGpuStencil() {

    // we need to wait for the CPU thread to arrive here
    if(blockingSyncCpuGpu)
        syncCpuAndGpu(false);

    if(!gpuToCpuStencilStarted.load()) {
        std::cerr << "ERROR: No GPU->CPU stencil exchange has been started yet... Abort!" << std::endl;
        exit(1);
    }

    try {

        double *dat = new double[cTgStencil];
        for(int i = 0; i < cTgStencil; ++i)
            dat[i] = cpuToGpuStencilBuffer[i].load();

        cl::copy(cl_queue, &dat[0], (&dat[cTgStencil-1])+1, cl_cpuToGpuStencilBuffer);

        delete[] dat;

        auto kernel_distributeHaloStencil = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&,
                                                           cl::Buffer&, cl::Buffer&>(cl_programs, "distributeHaloStencil");

        int globalSize = (cTgStencil/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_distributeHaloStencil(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                                     cl_stencilDim[X], cl_stencilDim[Y], cl_stencilDim[Z], cl_gpuHaloWidth,
                                     gpuStencil, cl_stencilNumPoints, cl_cpuToGpuStencilBuffer);

    } catch(cl::Error error) {
        std::cout << "[dist halo] Error: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }


}

// both the CPU and GPU have to arrive at this point before either can continue
void Tausch3D::syncCpuAndGpu(bool offsetByTwo) {

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

void Tausch3D::compileKernels() {

    // Tausch requires two kernels: One for collecting the halo data and one for distributing that data
    std::string oclstr = "typedef " + std::string((sizeof(real_t)==sizeof(double)) ? "double" : "float") + " real_t;\n";

    oclstr += R"d(
enum { LEFT, RIGHT, TOP, BOTTOM, FRONT, BACK };

kernel void collectHaloData(global const int * restrict const dimX, global const int * restrict const dimY,
                            global const int * restrict const dimZ, global const int * restrict const haloWidth,
                            global const real_t * restrict const vec, global real_t * sync) {

    int current = get_global_id(0);

    int maxNum = haloWidth[LEFT]*(*dimY)*(*dimZ) +
                 haloWidth[RIGHT]*(*dimY)*(*dimZ) +
                 haloWidth[TOP]*(*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimZ) +
                 haloWidth[BOTTOM]*(*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimZ) +
                 haloWidth[FRONT]*(*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM]) +
                 haloWidth[BACK]*(*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM]);

    if(current >= maxNum)
        return;

    // left
    if(current < haloWidth[LEFT]*(*dimY)*(*dimZ)) {

        int index =
           /* z */  (haloWidth[FRONT] + current/(haloWidth[LEFT]*(*dimY))) * ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*
           /* z */  (*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
           /* y */  (haloWidth[BOTTOM] + (current%(haloWidth[LEFT]*(*dimY)))/haloWidth[LEFT]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
           /* x */  current%haloWidth[LEFT] + haloWidth[LEFT];

        sync[current] = vec[index];
        return;
    }

    int offset = haloWidth[LEFT]*(*dimY)*(*dimZ);

    // right
    if(current < offset+haloWidth[RIGHT]*(*dimY)*(*dimZ)) {

        current -= offset;

        int index =
           /* z */  (haloWidth[FRONT] + current/(haloWidth[RIGHT]*(*dimY))) * ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*
           /* z */  (*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
           /* y */  (haloWidth[BOTTOM] + (current%(haloWidth[RIGHT]*(*dimY)))/haloWidth[RIGHT]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
           /* x */  current%haloWidth[RIGHT] + *dimX - haloWidth[LEFT];

        sync[offset+current] = vec[index];
        return;
    }

    offset += haloWidth[RIGHT]*(*dimY)*(*dimZ);

    // top
    if(current < offset + (*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[TOP]*(*dimZ)) {

        current -= offset;

        int index =
           /* z */  (haloWidth[FRONT] + current/((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[TOP])) *
           /* z */  ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
           /* y */  ((current%((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[TOP]))/
           /* y */  (*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + *dimY + haloWidth[TOP]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
           /* x */  current%(*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[LEFT];

        sync[offset+current] = vec[index];
        return;
    }

    offset += (*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[TOP]*(*dimZ);

    // bottom
    if(current < offset + (*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[BOTTOM]*(*dimZ)) {

        current -= offset;

        int index =
           /* z */  (haloWidth[FRONT] + current/((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[BOTTOM])) *
           /* z */  ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
           /* y */  ((current%((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[BOTTOM]))/
           /* y */  (*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + haloWidth[BOTTOM]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
           /* x */  current%(*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[LEFT];

        sync[offset+current] = vec[index];
        return;
    }

    offset += (*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[BOTTOM]*(*dimZ);

    // front
    if(current < offset + (*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])*haloWidth[FRONT]) {

        current -= offset;

        int index =
           /* z */  (haloWidth[FRONT] + current/((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM]))) *
           /* z */  ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
           /* y */  ((current%((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])))/
           /* y */  (*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[BOTTOM]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
           /* x */  current%(*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[LEFT];

        sync[offset+current] = vec[index];
        return;
    }

    offset += (*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])*haloWidth[FRONT];

    // back
    if(current < offset + (*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])*haloWidth[BACK]) {

        current -= offset;

        int index =
           /* z */  (current/((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])) +
           /* z */  (*dimZ) +haloWidth[FRONT]-haloWidth[BACK]) * ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
           /* y */  ((current%((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])))/
           /* y */  (*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[BOTTOM]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
           /* x */  current%(*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[LEFT];

        sync[offset+current] = vec[index];
        return;
    }

}

kernel void collectHaloStencil(global const int * restrict const dimX, global const int * restrict const dimY,
                               global const int * restrict const dimZ, global const int * restrict const haloWidth,
                               global const real_t * restrict const vec, global const int * restrict const stencilNumPoints,
                               global real_t * sync) {

    int current = get_global_id(0);

    int maxNum = (*stencilNumPoints)*(haloWidth[LEFT]*(*dimY)*(*dimZ) +
                                      haloWidth[RIGHT]*(*dimY)*(*dimZ) +
                                      haloWidth[TOP]*(*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimZ) +
                                      haloWidth[BOTTOM]*(*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimZ) +
                                      haloWidth[FRONT]*(*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM]) +
                                      haloWidth[BACK]*(*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM]));
    if(current >= maxNum)
        return;

    // left
    if(current < (*stencilNumPoints)*haloWidth[LEFT]*(*dimY)*(*dimZ)) {

        int current_stencil = current%(*stencilNumPoints);
        int current_index = current/(*stencilNumPoints);

        int index = (*stencilNumPoints)*
               /* z */  ((haloWidth[FRONT] + current_index/(haloWidth[LEFT]*(*dimY))) * ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) *
               /* z */  (*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
               /* y */  (haloWidth[BOTTOM] + (current_index%(haloWidth[LEFT]*(*dimY)))/haloWidth[LEFT]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
               /* x */  current_index%haloWidth[LEFT] + haloWidth[LEFT]) + current_stencil;

        sync[current] = vec[index];
        return;
    }

    int offset = (*stencilNumPoints)*haloWidth[LEFT]*(*dimY)*(*dimZ);

    // right
    if(current < offset+(*stencilNumPoints)*haloWidth[RIGHT]*(*dimY)*(*dimZ)) {

        current -= offset;
        int current_stencil = current%(*stencilNumPoints);
        int current_index = current/(*stencilNumPoints);

        int index = (*stencilNumPoints)*
               /* z */  ((haloWidth[FRONT] + current_index/(haloWidth[RIGHT]*(*dimY))) * ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*
               /* z */  (*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
               /* y */  (haloWidth[BOTTOM] + (current_index%(haloWidth[RIGHT]*(*dimY)))/haloWidth[RIGHT]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
               /* x */  current_index%haloWidth[RIGHT] + *dimX + haloWidth[LEFT] - haloWidth[RIGHT]) + current_stencil;

        sync[offset+current] = vec[index];
        return;
    }

    offset += (*stencilNumPoints)*haloWidth[RIGHT]*(*dimY)*(*dimZ);

    // top
    if(current < offset + (*stencilNumPoints)*(*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[TOP]*(*dimZ)) {

        current -= offset;
        int current_stencil = current%(*stencilNumPoints);
        int current_index = current/(*stencilNumPoints);

        int index = (*stencilNumPoints)*
               /* z */  ((haloWidth[FRONT] + current_index/((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[TOP])) *
               /* z */  ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
               /* y */  ((current_index%((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[TOP]))/(*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + *dimY +
               /* y */  haloWidth[BOTTOM] - haloWidth[TOP]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
               /* x */  current_index%(*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[LEFT]) + current_stencil;

        sync[offset+current] = vec[index];
        return;
    }

    offset += (*stencilNumPoints)*(*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[TOP]*(*dimZ);

    // bottom
    if(current < offset + (*stencilNumPoints)*(*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[BOTTOM]*(*dimZ)) {

        current -= offset;
        int current_stencil = current%(*stencilNumPoints);
        int current_index = current/(*stencilNumPoints);

        int index = (*stencilNumPoints)*
               /* z */  ((haloWidth[FRONT] + current_index/((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[BOTTOM])) *
               /* z */  ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
               /* y */  ((current_index%((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[BOTTOM]))/
               /* y */  (*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + haloWidth[BOTTOM]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
               /* x */  current_index%(*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[LEFT]) + current_stencil;

        sync[offset+current] = vec[index];
        return;
    }

    offset += (*stencilNumPoints)*(*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[BOTTOM]*(*dimZ);

    // front
    if(current < offset + (*stencilNumPoints)*(*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])*haloWidth[FRONT]) {

        current -= offset;
        int current_stencil = current%(*stencilNumPoints);
        int current_index = current/(*stencilNumPoints);

        int index = (*stencilNumPoints)*
               /* z */  ((haloWidth[FRONT] + current_index/((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM]))) *
               /* z */  ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
               /* y */  ((current_index%((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])))/
               /* y */  (*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[BOTTOM]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
               /* x */  current_index%(*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[LEFT]) + current_stencil;

        sync[offset+current] = vec[index];
        return;
    }

    offset += (*stencilNumPoints)*(*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])*haloWidth[FRONT];

    // back
    current -= offset;
    int current_stencil = current%(*stencilNumPoints);
    int current_index = current/(*stencilNumPoints);

    int index = (*stencilNumPoints)*
           /* z */  ((current_index/((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])) +
           /* z */  (*dimZ) +haloWidth[FRONT]-haloWidth[BACK]) * ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
           /* y */  ((current_index%((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])))/
           /* y */  (*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[BOTTOM]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
           /* x */  current_index%(*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[LEFT]) + current_stencil;

    sync[offset+current] = vec[index];

}

kernel void distributeHaloData(global const int * restrict const dimX, global const int * restrict const dimY,
                               global const int * restrict const dimZ, global const int * restrict const haloWidth,
                               global real_t * vec, global const real_t * restrict const sync) {

    int current = get_global_id(0);

    int maxNum = haloWidth[LEFT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]) +
                 haloWidth[RIGHT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]) +
                 haloWidth[TOP]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]) +
                 haloWidth[BOTTOM]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]) +
                 haloWidth[FRONT]*(*dimX)*(*dimY) +
                 haloWidth[BACK]*(*dimX)*(*dimY);

    if(current >= maxNum)
        return;

    // left
    if(current < haloWidth[LEFT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])*(*dimZ+haloWidth[FRONT]+haloWidth[BACK])) {

        int index =
           /* z */  (current/(haloWidth[LEFT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM]))) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) *
           /* z */  (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
           /* y */  ((current%(haloWidth[LEFT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM]))) / haloWidth[LEFT])*(*dimX+haloWidth[LEFT]+haloWidth[RIGHT])+
           /* x */  current%haloWidth[LEFT];

        vec[index] = sync[current];
        return;
    }

    int offset = haloWidth[LEFT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]);

    // right
    if(current < offset + haloWidth[RIGHT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])*(*dimZ+haloWidth[FRONT]+haloWidth[BACK])) {

        current -= offset;

        int index =
           /* z */  (current/(haloWidth[RIGHT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM]))) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) *
           /* z */  (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
           /* y */  ((current%(haloWidth[RIGHT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM]))) / haloWidth[RIGHT]) *
           /* y */  (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
           /* x */  current%haloWidth[RIGHT] + *dimX+haloWidth[LEFT];

        vec[index] = sync[offset+current];
        return;
    }

    offset += haloWidth[RIGHT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]);

    // top
    if(current < offset + haloWidth[TOP]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK])) {

        current -= offset;

        int index =
           /* z */  (current/(haloWidth[TOP]*(*dimX))) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) * (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
           /* y */  ((current%(haloWidth[TOP]*(*dimX))) / (*dimX) + *dimY+haloWidth[BOTTOM]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
           /* x */  current%(*dimX) + haloWidth[LEFT];

        vec[index] = sync[offset+current];
        return;
    }

    offset += haloWidth[TOP]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]);

    // bottom
    if(current < offset + haloWidth[BOTTOM]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK])) {

        current -= offset;

        int index =
           /* z */  (current/(haloWidth[BOTTOM]*(*dimX))) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) * (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
           /* y */  ((current%(haloWidth[BOTTOM]*(*dimX))) / (*dimX)) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
           /* x */  current%(*dimX) + haloWidth[LEFT];

        vec[index] = sync[offset+current];
        return;
    }

    offset += haloWidth[BOTTOM]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]);

    // front
    if(current < offset + haloWidth[FRONT]*(*dimX)*(*dimY)) {

        current -= offset;

        int index =
           /* z */  (current/((*dimX)*(*dimY))) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) * (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
           /* y */  ((current%((*dimX)*(*dimY))) / (*dimX) + haloWidth[BOTTOM]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
           /* x */  current%(*dimX) + haloWidth[LEFT];

        vec[index] = sync[offset+current];
        return;
    }

    offset += haloWidth[FRONT]*(*dimX)*(*dimY);

    // back
    current -= offset;

    int index =
       /* z */  (current/((*dimX)*(*dimY)) + (*dimZ) + haloWidth[FRONT]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) *
       /* z */  (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
       /* y */  ((current%((*dimX)*(*dimY))) / (*dimX) + haloWidth[BOTTOM]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
       /* x */  current%(*dimX) + haloWidth[LEFT];

    vec[index] = sync[offset+current];

}

kernel void distributeHaloStencil(global const int * restrict const dimX, global const int * restrict const dimY,
                                  global const int * restrict const dimZ, global const int * restrict const haloWidth,
                                  global real_t * vec, global const int * restrict const stencilNumPoints,
                                  global const real_t * restrict const sync) {

    int current = get_global_id(0);

    int maxNum = (*stencilNumPoints)*(haloWidth[LEFT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]) +
                                      haloWidth[RIGHT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]) +
                                      haloWidth[TOP]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]) +
                                      haloWidth[BOTTOM]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]) +
                                      haloWidth[FRONT]*(*dimX)*(*dimY) +
                                      haloWidth[BACK]*(*dimX)*(*dimY));

    if(current >= maxNum)
        return;

    // left
    if(current < (*stencilNumPoints)*haloWidth[LEFT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])*(*dimZ+haloWidth[FRONT]+haloWidth[BACK])) {

        int current_stencil = current%(*stencilNumPoints);
        int current_index = current/(*stencilNumPoints);

        int index = (*stencilNumPoints)*
               /* z */  ((current_index/(haloWidth[LEFT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM]))) *
               /* z */  (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) * (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
               /* y */  ((current_index%(haloWidth[LEFT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM]))) / haloWidth[LEFT]) *
               /* y */  (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
               /* x */  current_index%haloWidth[LEFT]) + current_stencil;

        vec[index] = sync[current];
        return;
    }

    int offset = (*stencilNumPoints)*haloWidth[LEFT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]);

    // right
    if(current < offset + (*stencilNumPoints)*haloWidth[RIGHT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])*(*dimZ+haloWidth[FRONT]+haloWidth[BACK])) {

        current -= offset;
        int current_stencil = current%(*stencilNumPoints);
        int current_index = current/(*stencilNumPoints);

        int index = (*stencilNumPoints)*
               /* z */  ((current_index/(haloWidth[RIGHT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM]))) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) *
               /* z */  (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
               /* y */  ((current_index%(haloWidth[RIGHT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM]))) / haloWidth[RIGHT]) *
               /* y */  (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
               /* z */  current_index%haloWidth[RIGHT] + *dimX+haloWidth[LEFT]) + current_stencil;

        vec[index] = sync[offset+current];
        return;
    }

    offset += (*stencilNumPoints)*haloWidth[RIGHT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]);

    // top
    if(current < offset + (*stencilNumPoints)*haloWidth[TOP]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK])) {

        current -= offset;
        int current_stencil = current%(*stencilNumPoints);
        int current_index = current/(*stencilNumPoints);

        int index = (*stencilNumPoints)*
               /* z */  ((current_index/(haloWidth[TOP]*(*dimX))) *(*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) *(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])+
               /* y */  ((current_index%(haloWidth[TOP]*(*dimX))) / (*dimX) + *dimY+haloWidth[BOTTOM]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
               /* x */  current_index%(*dimX) + haloWidth[LEFT]) + current_stencil;

        vec[index] = sync[offset+current];
        return;
    }

    offset += (*stencilNumPoints)*haloWidth[TOP]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]);

    // bottom
    if(current < offset + (*stencilNumPoints)*haloWidth[BOTTOM]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK])) {

        current -= offset;
        int current_stencil = current%(*stencilNumPoints);
        int current_index = current/(*stencilNumPoints);

        int index = (*stencilNumPoints)*
               /* z */  ((current_index/(haloWidth[BOTTOM]*(*dimX))) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) *
               /* z */  (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
               /* y */  ((current_index%(haloWidth[BOTTOM]*(*dimX))) / (*dimX)) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
               /* x */  current_index%(*dimX) + haloWidth[LEFT]) + current_stencil;

        vec[index] = sync[offset+current];
        return;
    }

    offset += (*stencilNumPoints)*haloWidth[BOTTOM]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]);

    // front
    if(current < offset + (*stencilNumPoints)*haloWidth[FRONT]*(*dimX)*(*dimY)) {

        current -= offset;
        int current_stencil = current%(*stencilNumPoints);
        int current_index = current/(*stencilNumPoints);

        int index = (*stencilNumPoints)*
               /* z */  ((current_index/((*dimX)*(*dimY))) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) * (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
               /* y */  ((current_index%((*dimX)*(*dimY)))/(*dimX) + haloWidth[BOTTOM]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
               /* x */  current_index%(*dimX) + haloWidth[LEFT]) + current_stencil;

        vec[index] = sync[offset+current];
        return;
    }

    offset += (*stencilNumPoints)*haloWidth[FRONT]*(*dimX)*(*dimY);

    // back
    current -= offset;
    int current_stencil = current%(*stencilNumPoints);
    int current_index = current/(*stencilNumPoints);

    int index = (*stencilNumPoints)*
                  /* z */  ((current_index/((*dimX)*(*dimY)) + (*dimZ) + haloWidth[FRONT]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) *
                  /* z */  (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
                  /* y */  ((current_index%((*dimX)*(*dimY))) / (*dimX) + haloWidth[BOTTOM]) * (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
                  /* x */  current_index%(*dimX) + haloWidth[LEFT]) + current_stencil;

    vec[index] = sync[offset+current];

}

              )d";

    try {
        cl_programs = cl::Program(cl_context, oclstr, true);
    } catch(cl::Error error) {
        std::cout << "[kernel compile] OpenCL exception caught: "
                  << error.what() << " (" << error.err() << ")" << std::endl;
        if(error.err() == -11) {
            std::string log = cl_programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_defaultDevice);
            std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl
                      << " ******************** " << std::endl << log << std::endl << std::endl
                      << " ******************** " << std::endl << std::endl;
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
                              << " with device #" << device_num[mpiRank%num] << ": "
                              << cl_defaultDevice.getInfo<CL_DEVICE_NAME>() << std::endl;
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
