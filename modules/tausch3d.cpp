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

    // cpu at beginning
    cpuRecvsPosted = false;

    // communication to neither edge has been started
    cpuStarted[LEFT] = false;
    cpuStarted[RIGHT] = false;
    cpuStarted[TOP] = false;
    cpuStarted[BOTTOM] = false;
    cpuStarted[FRONT] = false;
    cpuStarted[BACK] = false;

    mpiDataType = ((sizeof(real_t) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT);

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
void Tausch3D::setCpuData(real_t *dat) {

    cpuInfoGiven = true;
    cpuData = dat;

    // a send and recv buffer for the CPU-CPU communication
    cpuToCpuSendBuffer = new real_t*[6];
    cpuToCpuSendBuffer[LEFT]   = new real_t[cpuHaloWidth[RIGHT]*
                                           (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                           (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK])]{};
    cpuToCpuSendBuffer[RIGHT]  = new real_t[cpuHaloWidth[LEFT]*
                                           (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                           (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK])]{};
    cpuToCpuSendBuffer[TOP]    = new real_t[cpuHaloWidth[BOTTOM]*
                                           (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                           (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK])]{};
    cpuToCpuSendBuffer[BOTTOM] = new real_t[cpuHaloWidth[TOP]*
                                           (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                           (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK])]{};
    cpuToCpuSendBuffer[FRONT]  = new real_t[cpuHaloWidth[BACK]*
                                           (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                           (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])]{};
    cpuToCpuSendBuffer[BACK]   = new real_t[cpuHaloWidth[FRONT]*
                                           (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                           (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])]{};
    cpuToCpuRecvBuffer = new real_t*[6];
    cpuToCpuRecvBuffer[LEFT]   = new real_t[cpuHaloWidth[LEFT]*
                                           (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                           (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK])]{};
    cpuToCpuRecvBuffer[RIGHT]  = new real_t[cpuHaloWidth[RIGHT]*
                                           (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                           (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK])]{};
    cpuToCpuRecvBuffer[TOP]    = new real_t[cpuHaloWidth[TOP]*
                                           (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                           (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK])]{};
    cpuToCpuRecvBuffer[BOTTOM] = new real_t[cpuHaloWidth[BOTTOM]*
                                           (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                           (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK])]{};
    cpuToCpuRecvBuffer[FRONT]  = new real_t[cpuHaloWidth[FRONT]*
                                           (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                           (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])]{};
    cpuToCpuRecvBuffer[BACK]   = new real_t[cpuHaloWidth[BACK]*
                                           (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                           (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])]{};

    // Initialise the Recv/Send operations
    if(haveBoundary[LEFT]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[LEFT], cpuHaloWidth[LEFT]*
                                               (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                               (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
                      mpiDataType, mpiRank-1, 0, TAUSCH_COMM, &cpuToCpuRecvRequest[LEFT]);
        MPI_Send_init(cpuToCpuSendBuffer[LEFT], cpuHaloWidth[RIGHT]*
                                               (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                               (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
                      mpiDataType, mpiRank-1, 2, TAUSCH_COMM, &cpuToCpuSendRequest[LEFT]);
    }
    if(haveBoundary[RIGHT]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[RIGHT], cpuHaloWidth[RIGHT]*
                                                (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                                (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
                      mpiDataType, mpiRank+1, 2, TAUSCH_COMM, &cpuToCpuRecvRequest[RIGHT]);
        MPI_Send_init(cpuToCpuSendBuffer[RIGHT], cpuHaloWidth[LEFT]*
                                                (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                                (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
                      mpiDataType, mpiRank+1, 0, TAUSCH_COMM, &cpuToCpuSendRequest[RIGHT]);
    }
    if(haveBoundary[TOP]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[TOP], cpuHaloWidth[TOP]*
                                              (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                              (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
                      mpiDataType, mpiRank+mpiNum[X], 1, TAUSCH_COMM, &cpuToCpuRecvRequest[TOP]);
        MPI_Send_init(cpuToCpuSendBuffer[TOP], cpuHaloWidth[BOTTOM]*
                                              (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                              (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
                      mpiDataType, mpiRank+mpiNum[X], 3, TAUSCH_COMM, &cpuToCpuSendRequest[TOP]);
    }
    if(haveBoundary[BOTTOM]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[BOTTOM], cpuHaloWidth[BOTTOM]*
                                                 (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                 (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
                      mpiDataType, mpiRank-mpiNum[X], 3, TAUSCH_COMM, &cpuToCpuRecvRequest[BOTTOM]);
        MPI_Send_init(cpuToCpuSendBuffer[BOTTOM], cpuHaloWidth[TOP]*
                                                 (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                 (localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]),
                      mpiDataType, mpiRank-mpiNum[X], 1, TAUSCH_COMM, &cpuToCpuSendRequest[BOTTOM]);
    }
    if(haveBoundary[FRONT]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[FRONT], cpuHaloWidth[FRONT]*
                                                (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]),
                      mpiDataType, mpiRank-mpiNum[X]*mpiNum[Y], 4, TAUSCH_COMM, &cpuToCpuRecvRequest[FRONT]);
        MPI_Send_init(cpuToCpuSendBuffer[FRONT], cpuHaloWidth[BACK]*
                                                (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]),
                      mpiDataType, mpiRank-mpiNum[X]*mpiNum[Y], 5, TAUSCH_COMM, &cpuToCpuSendRequest[FRONT]);
    }
    if(haveBoundary[BACK]) {
        MPI_Recv_init(cpuToCpuRecvBuffer[BACK], cpuHaloWidth[BACK]*
                                               (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                               (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]),
                      mpiDataType, mpiRank+mpiNum[X]*mpiNum[Y], 5, TAUSCH_COMM, &cpuToCpuRecvRequest[BACK]);
        MPI_Send_init(cpuToCpuSendBuffer[BACK], cpuHaloWidth[FRONT]*
                                               (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                               (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]),
                      mpiDataType, mpiRank+mpiNum[X]*mpiNum[Y], 4, TAUSCH_COMM, &cpuToCpuSendRequest[BACK]);
    }

}

// post the MPI_Irecv's for inter-rank communication
void Tausch3D::postCpuReceives() {

    if(!cpuInfoGiven) {
        std::cerr << "ERROR: You didn't tell me yet where to find the data! Abort..." << std::endl;
        exit(1);
    }

    cpuRecvsPosted = true;

    mpiDataType = ((sizeof(real_t) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT);

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

    mpiDataType = ((sizeof(real_t) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT);

    if(edge == LEFT && haveBoundary[LEFT]) {
        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < cpuHaloWidth[RIGHT]; ++x)
                    cpuToCpuSendBuffer[LEFT][z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*cpuHaloWidth[RIGHT]+
                                             y*cpuHaloWidth[RIGHT] + x]
                            = cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                        (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                      y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                                      x+cpuHaloWidth[LEFT]];
        MPI_Start(&cpuToCpuSendRequest[LEFT]);
    } else if(edge == RIGHT && haveBoundary[RIGHT]) {
        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < cpuHaloWidth[LEFT]; ++x)
                    cpuToCpuSendBuffer[RIGHT][z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*cpuHaloWidth[LEFT]+
                                              y*cpuHaloWidth[LEFT] + x]
                            = cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                        (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                      y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                                      localDim[X]+x];
        MPI_Start(&cpuToCpuSendRequest[RIGHT]);
    } else if(edge == TOP && haveBoundary[TOP]) {
        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < cpuHaloWidth[BOTTOM]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)
                    cpuToCpuSendBuffer[TOP][z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*cpuHaloWidth[BOTTOM]+
                                            y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x]
                            = cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                        (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                     (y+localDim[Y])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x];
        MPI_Start(&cpuToCpuSendRequest[TOP]);
    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {
        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)
                    cpuToCpuSendBuffer[BOTTOM][z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*cpuHaloWidth[TOP]+
                                               y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x]
                            = cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                        (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
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
                                                              (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                                       y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x];
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
        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < cpuHaloWidth[LEFT]; ++x)
                    cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                              (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                            y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x]
                            = cpuToCpuRecvBuffer[LEFT][z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                                          cpuHaloWidth[LEFT] + y*cpuHaloWidth[LEFT] + x];
        MPI_Wait(&cpuToCpuSendRequest[LEFT], MPI_STATUS_IGNORE);
    } else if(edge == RIGHT && haveBoundary[RIGHT]) {
        MPI_Wait(&cpuToCpuRecvRequest[RIGHT], MPI_STATUS_IGNORE);
        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < cpuHaloWidth[RIGHT]; ++x)
                    cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                              (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                            y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                            localDim[X]+cpuHaloWidth[LEFT]+x]
                            = cpuToCpuRecvBuffer[RIGHT][z*(localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP])*
                                                           cpuHaloWidth[RIGHT] + y*cpuHaloWidth[RIGHT] + x];
        MPI_Wait(&cpuToCpuSendRequest[RIGHT], MPI_STATUS_IGNORE);
    } else if(edge == TOP && haveBoundary[TOP]) {
        MPI_Wait(&cpuToCpuRecvRequest[TOP], MPI_STATUS_IGNORE);
        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)
                    cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                              (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                           (y+localDim[Y]+cpuHaloWidth[BOTTOM])*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x]
                            = cpuToCpuRecvBuffer[TOP][z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                         cpuHaloWidth[TOP] +
                                                      y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x];
        MPI_Wait(&cpuToCpuSendRequest[TOP], MPI_STATUS_IGNORE);
    } else if(edge == BOTTOM && haveBoundary[BOTTOM]) {
        MPI_Wait(&cpuToCpuRecvRequest[BOTTOM], MPI_STATUS_IGNORE);
        for(int z = 0; z < localDim[Z]+cpuHaloWidth[FRONT]+cpuHaloWidth[BACK]; ++z)
            for(int y = 0; y < cpuHaloWidth[BOTTOM]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)
                    cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                              (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                            y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])+x]
                            = cpuToCpuRecvBuffer[BOTTOM][z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                                            cpuHaloWidth[BOTTOM] +
                                                         y*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) + x];
        MPI_Wait(&cpuToCpuSendRequest[BOTTOM], MPI_STATUS_IGNORE);
    } else if(edge == FRONT && haveBoundary[FRONT]) {
        MPI_Wait(&cpuToCpuRecvRequest[FRONT], MPI_STATUS_IGNORE);
        for(int z = 0; z < cpuHaloWidth[FRONT]; ++z)
            for(int y = 0; y < localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]; ++y)
                for(int x = 0; x < localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]; ++x)
                    cpuData[z*(localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                              (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
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

#ifdef TAUSCH_OPENCL

void Tausch3D::enableOpenCL(int *gpuHaloWidth, bool blockingSyncCpuGpu,
                            int clLocalWorkgroupSize, bool giveOpenCLDeviceName) {

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

    cl_gpuHaloWidth = cl::Buffer(cl_context, &gpuHaloWidth, (&gpuHaloWidth)+1, true);

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
    int cTg = gpuHaloWidth[LEFT]*(gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*
                                 (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
              gpuHaloWidth[RIGHT]*(gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*
                                  (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
              gpuHaloWidth[TOP]*gpuDim[X]*(gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
              gpuHaloWidth[BOTTOM]*gpuDim[X]*(gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
              gpuHaloWidth[FRONT]*gpuDim[X]*gpuDim[Y] +
              gpuHaloWidth[BACK]*gpuDim[X]*gpuDim[Y];
    int gTc = gpuHaloWidth[LEFT]*gpuDim[Y]*gpuDim[Z] +
              gpuHaloWidth[RIGHT]*gpuDim[Y]*gpuDim[Z] +
              gpuHaloWidth[TOP]*(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuDim[Z] +
              gpuHaloWidth[BOTTOM]*(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuDim[Z] +
              gpuHaloWidth[FRONT]*(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*
                                  (gpuDim[Y]-gpuHaloWidth[BOTTOM]-gpuHaloWidth[TOP]) +
              gpuHaloWidth[BACK]*(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*
                                 (gpuDim[Y]-gpuHaloWidth[BOTTOM]-gpuHaloWidth[TOP]);

    cpuToGpuBuffer = new std::atomic<real_t>[cTg]{};
    gpuToCpuBuffer = new std::atomic<real_t>[gTc]{};

    // set up buffers on device
    try {
        cl_cpuToGpuBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, cTg*sizeof(real_t));
        cl_gpuToCpuBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, gTc*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_cpuToGpuBuffer, 0, 0, cTg*sizeof(real_t));
        cl_queue.enqueueFillBuffer(cl_gpuToCpuBuffer, 0, 0, gTc*sizeof(real_t));
        cl_gpuDim[X] = cl::Buffer(cl_context, &gpuDim[X], (&gpuDim[X])+1, true);
        cl_gpuDim[Y] = cl::Buffer(cl_context, &gpuDim[Y], (&gpuDim[Y])+1, true);
        cl_gpuDim[Z] = cl::Buffer(cl_context, &gpuDim[Z], (&gpuDim[Z])+1, true);
        cl_gpuHaloWidth = cl::Buffer(cl_context, &gpuHaloWidth[0], (&gpuHaloWidth[5])+1, true);
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
    for(int z = 0; z < (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]); ++z) {
        for(int y = 0; y < (gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP]); ++y) {
            for(int x = 0; x < gpuHaloWidth[LEFT]; ++x) {
                int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-gpuDim[Z])/2-gpuHaloWidth[FRONT])*
                                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                    (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                            (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-gpuDim[Y])/2-gpuHaloWidth[BOTTOM])*
                                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                            x+cpuHaloWidth[LEFT]+(localDim[X]-gpuDim[X])/2 - gpuHaloWidth[LEFT];
                cpuToGpuBuffer[z*(gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*gpuHaloWidth[LEFT] +
                               y*gpuHaloWidth[LEFT] + x].store(cpuData[index]);
            }
        }
    }
    int offset = (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK])*
                        (gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*
                         gpuHaloWidth[LEFT];
    // right
    for(int z = 0; z < (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]); ++z) {
        for(int y = 0; y < (gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP]); ++y) {
            for(int x = 0; x < gpuHaloWidth[RIGHT]; ++x) {
                int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-gpuDim[Z])/2-gpuHaloWidth[FRONT])*
                                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                    (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                            (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-gpuDim[Y])/2-gpuHaloWidth[BOTTOM])*
                                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                            x+cpuHaloWidth[LEFT]+(localDim[X]-gpuDim[X])/2+gpuDim[X];
                cpuToGpuBuffer[offset + z*(gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*gpuHaloWidth[RIGHT] +
                               y*gpuHaloWidth[RIGHT] + x].store(cpuData[index]);
            }
        }
    }
    offset += (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK])*
                    (gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*
                     gpuHaloWidth[RIGHT];
    // top
    for(int z = 0; z < (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]); ++z) {
        for(int y = 0; y < gpuHaloWidth[TOP]; ++y) {
            for(int x = 0; x < gpuDim[X]; ++x) {
                int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-gpuDim[Z])/2-gpuHaloWidth[FRONT])*
                                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                    (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                            (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-gpuDim[Y])/2+gpuDim[Y])*
                                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                            x+cpuHaloWidth[LEFT]+(localDim[X]-gpuDim[X])/2;
                cpuToGpuBuffer[offset + z*gpuHaloWidth[TOP]*gpuDim[X] + y*gpuDim[X] + x].store(cpuData[index]);
            }
        }
    }
    offset += (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK])*gpuHaloWidth[TOP]*gpuDim[X];
    // bottom
    for(int z = 0; z < (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]); ++z) {
        for(int y = 0; y < gpuHaloWidth[BOTTOM]; ++y) {
            for(int x = 0; x < gpuDim[X]; ++x) {
                int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-gpuDim[Z])/2-gpuHaloWidth[FRONT])*
                                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                    (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                            (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-gpuDim[Y])/2-gpuHaloWidth[BOTTOM])*
                                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                            x+cpuHaloWidth[LEFT]+(localDim[X]-gpuDim[X])/2;
                cpuToGpuBuffer[offset + z*gpuHaloWidth[BOTTOM]*gpuDim[X] + y*gpuDim[X] + x].store(cpuData[index]);
            }
        }
    }
    offset += (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK])*gpuHaloWidth[BOTTOM]*gpuDim[X];
    // front
    for(int z = 0; z < gpuHaloWidth[FRONT]; ++z) {
        for(int y = 0; y < gpuDim[Y]; ++y) {
            for(int x = 0; x < gpuDim[X]; ++x) {
                int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-gpuDim[Z])/2-gpuHaloWidth[FRONT])*
                                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                    (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                            (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-gpuDim[Y])/2)*
                                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                            x+cpuHaloWidth[LEFT]+(localDim[X]-gpuDim[X])/2;
                cpuToGpuBuffer[offset + z*gpuDim[Y]*gpuDim[X] + y*gpuDim[X] + x].store(cpuData[index]);
            }
        }
    }
    offset += gpuHaloWidth[FRONT]*gpuDim[Y]*gpuDim[X];
    // back
    for(int z = 0; z < gpuHaloWidth[BACK]; ++z) {
        for(int y = 0; y < gpuDim[Y]; ++y) {
            for(int x = 0; x < gpuDim[X]; ++x) {
                int index = (z+cpuHaloWidth[FRONT]+(localDim[Z]-gpuDim[Z])/2+gpuDim[Z])*
                                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                                    (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                            (y+cpuHaloWidth[BOTTOM]+(localDim[Y]-gpuDim[Y])/2)*
                                    (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                            x+cpuHaloWidth[LEFT]+(localDim[X]-gpuDim[X])/2;
                cpuToGpuBuffer[offset + z*gpuDim[Y]*gpuDim[X] + y*gpuDim[X] + x].store(cpuData[index]);
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

        int gTc = gpuHaloWidth[LEFT]*gpuDim[Y]*gpuDim[Z] +
                  gpuHaloWidth[RIGHT]*gpuDim[Y]*gpuDim[Z] +
                  gpuHaloWidth[TOP]*(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuDim[Z] +
                  gpuHaloWidth[BOTTOM]*(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuDim[Z] +
                  gpuHaloWidth[FRONT]*(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*
                                      (gpuDim[Y]-gpuHaloWidth[BOTTOM]-gpuHaloWidth[TOP]) +
                  gpuHaloWidth[BACK]*(gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*
                                     (gpuDim[Y]-gpuHaloWidth[BOTTOM]-gpuHaloWidth[TOP]);

        auto kernel_collectHalo = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&,
                                                  cl::Buffer&>(cl_programs, "collectHaloData");

        int globalSize = (gTc/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_collectHalo(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                           cl_gpuDim[X], cl_gpuDim[Y], cl_gpuDim[Z], cl_gpuHaloWidth, gpuData, cl_gpuToCpuBuffer);

        double *dat = new double[gTc];
        cl::copy(cl_queue, cl_gpuToCpuBuffer, &dat[0], (&dat[gTc-1])+1);

        for(int i = 0; i < gTc; ++i)
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
    for(int i = 0; i < gpuHaloWidth[LEFT]*gpuDim[Y]*gpuDim[Z]; ++i) {
        int index = ((i/(gpuHaloWidth[LEFT]*gpuDim[Y])) + cpuHaloWidth[FRONT] + (localDim[Z]-gpuDim[Z])/2)*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                            (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                    ((i%(gpuHaloWidth[LEFT]*gpuDim[Y]))/gpuHaloWidth[LEFT] + cpuHaloWidth[BOTTOM] +
                     (localDim[Y]-gpuDim[Y])/2)*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    (i%(gpuHaloWidth[LEFT]*gpuDim[Y]))%gpuHaloWidth[LEFT] +
                    (localDim[X]-gpuDim[X])/2 + cpuHaloWidth[LEFT];
        cpuData[index] = gpuToCpuBuffer[i].load();
    }
    int offset = gpuHaloWidth[LEFT]*gpuDim[Y]*gpuDim[Z];
    // right
    for(int i = 0; i < gpuHaloWidth[RIGHT]*gpuDim[Y]*gpuDim[Z]; ++i) {
        int index = ((i/(gpuHaloWidth[RIGHT]*gpuDim[Y])) + cpuHaloWidth[FRONT] + (localDim[Z]-gpuDim[Z])/2)*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                            (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                    ((i%(gpuHaloWidth[RIGHT]*gpuDim[Y]))/gpuHaloWidth[RIGHT] + cpuHaloWidth[BOTTOM] +
                     (localDim[Y]-gpuDim[Y])/2)*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    (i%(gpuHaloWidth[RIGHT]*gpuDim[Y]))%gpuHaloWidth[RIGHT] + (localDim[X]-gpuDim[X])/2 +
                     gpuDim[X] + cpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT];
        cpuData[index] = gpuToCpuBuffer[offset+i].load();
    }
    offset += gpuHaloWidth[RIGHT]*gpuDim[Y]*gpuDim[Z];
    // top
    for(int i = 0; i < (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[TOP]*gpuDim[Z]; ++i) {
        int index = ((i/((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[TOP])) +
                      cpuHaloWidth[FRONT] + (localDim[Z]-gpuDim[Z])/2)*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                            (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                    ((i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[TOP]))/
                     (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) + gpuDim[Y] +
                      cpuHaloWidth[BOTTOM]-gpuHaloWidth[TOP] + (localDim[Y]-gpuDim[Y])/2)*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    (i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[TOP]))%
                    (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) +
                    (localDim[X]-gpuDim[X])/2 + gpuHaloWidth[LEFT]+cpuHaloWidth[LEFT];
        cpuData[index] = gpuToCpuBuffer[offset+i].load();
    }
    offset += (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[TOP]*gpuDim[Z];
    // bottom
    for(int i = 0; i < (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[BOTTOM]*gpuDim[Z]; ++i) {
        int index = ((i/((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[BOTTOM])) +
                      cpuHaloWidth[FRONT] + (localDim[Z]-gpuDim[Z])/2)*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                            (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                    ((i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[BOTTOM]))/
                     (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) + cpuHaloWidth[BOTTOM] +
                     (localDim[Y]-gpuDim[Y])/2)*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    (i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[BOTTOM]))%
                    (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) +
                    (localDim[X]-gpuDim[X])/2 + gpuHaloWidth[LEFT]+cpuHaloWidth[LEFT];
        cpuData[index] = gpuToCpuBuffer[offset+i].load();
    }
    offset += (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*gpuHaloWidth[BOTTOM]*gpuDim[Z];
    // front
    for(int i = 0; i < (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*
                       (gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])*
                        gpuHaloWidth[FRONT]; ++i) {
        int index = ((i/((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*
                         (gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM]))) +
                     cpuHaloWidth[FRONT] + (localDim[Z]-gpuDim[Z])/2)*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                            (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                    ((i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*
                         (gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])))/
                     (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) + gpuHaloWidth[BOTTOM]+cpuHaloWidth[BOTTOM] +
                     (localDim[Y]-gpuDim[Y])/2)*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    (i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*
                        (gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])))%
                    (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) +
                    (localDim[X]-gpuDim[X])/2 + gpuHaloWidth[LEFT]+cpuHaloWidth[LEFT];
        cpuData[index] = gpuToCpuBuffer[offset+i].load();
    }
    offset += (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*
              (gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])*
               gpuHaloWidth[FRONT];
    // back
    for(int i = 0; i < (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*
                       (gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])*
                        gpuHaloWidth[BACK]; ++i) {
        int index = ((i/((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*
                         (gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM]))) +
                     gpuDim[Z] + cpuHaloWidth[FRONT]-gpuHaloWidth[BACK] + (localDim[Z]-gpuDim[Z])/2)*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT])*
                            (localDim[Y]+cpuHaloWidth[BOTTOM]+cpuHaloWidth[TOP]) +
                    ((i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*
                         (gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])))/
                     (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) +
                     gpuHaloWidth[BOTTOM]+cpuHaloWidth[BOTTOM] + (localDim[Y]-gpuDim[Y])/2)*
                            (localDim[X]+cpuHaloWidth[LEFT]+cpuHaloWidth[RIGHT]) +
                    (i%((gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT])*
                        (gpuDim[Y]-gpuHaloWidth[TOP]-gpuHaloWidth[BOTTOM])))%
                    (gpuDim[X]-gpuHaloWidth[LEFT]-gpuHaloWidth[RIGHT]) +
                    (localDim[X]-gpuDim[X])/2 + gpuHaloWidth[LEFT]+cpuHaloWidth[LEFT];
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

        int cTg = gpuHaloWidth[LEFT]*(gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*
                                     (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
                  gpuHaloWidth[RIGHT]*(gpuDim[Y]+gpuHaloWidth[BOTTOM]+gpuHaloWidth[TOP])*
                                      (gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
                  gpuHaloWidth[TOP]*gpuDim[X]*(gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
                  gpuHaloWidth[BOTTOM]*gpuDim[X]*(gpuDim[Z]+gpuHaloWidth[FRONT]+gpuHaloWidth[BACK]) +
                  gpuHaloWidth[FRONT]*gpuDim[X]*gpuDim[Y] +
                  gpuHaloWidth[BACK]*gpuDim[X]*gpuDim[Y];

        double *dat = new double[cTg];
        for(int i = 0; i < cTg; ++i)
            dat[i] = cpuToGpuBuffer[i].load();

        cl::copy(cl_queue, &dat[0], (&dat[cTg-1])+1, cl_cpuToGpuBuffer);

        delete[] dat;

        auto kernelDistributeHaloData = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&,
                                                        cl::Buffer&, cl::Buffer&>(cl_programs, "distributeHaloData");

        int globalSize = (cTg/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernelDistributeHaloData(cl::EnqueueArgs(cl_queue, cl::NDRange(globalSize), cl::NDRange(cl_kernelLocalSize)),
                                 cl_gpuDim[X], cl_gpuDim[Y], cl_gpuDim[Z], cl_gpuHaloWidth, gpuData, cl_cpuToGpuBuffer);

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
        int index = (haloWidth[FRONT] + current/(haloWidth[LEFT]*(*dimY))) *
                            ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
                    (haloWidth[BOTTOM] + (current%(haloWidth[LEFT]*(*dimY)))/haloWidth[LEFT]) *
                            (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
                    current%haloWidth[LEFT] + haloWidth[LEFT];
        sync[current] = vec[index];
        return;
    }
    int offset = haloWidth[LEFT]*(*dimY)*(*dimZ);
    // right
    if(current < offset+haloWidth[RIGHT]*(*dimY)*(*dimZ)) {
        current -= offset;
        int index = (haloWidth[FRONT] + current/(haloWidth[RIGHT]*(*dimY))) *
                            ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
                    (haloWidth[BOTTOM] + (current%(haloWidth[RIGHT]*(*dimY)))/haloWidth[RIGHT]) *
                            (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
                    current%haloWidth[RIGHT] + *dimX - haloWidth[LEFT];
        sync[offset+current] = vec[index];
        return;
    }
    // top
    offset += haloWidth[RIGHT]*(*dimY)*(*dimZ);
    if(current < offset + (*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[TOP]*(*dimZ)) {
        current -= offset;
        int index = (haloWidth[FRONT] + current/((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[TOP])) *
                            ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
                    ((current%((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[TOP]))/
                     (*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + *dimY + haloWidth[TOP]) *
                            (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
                    current%(*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[LEFT];
        sync[offset+current] = vec[index];
        return;
    }
    // bottom
    offset += (*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[TOP]*(*dimZ);
    if(current < offset + (*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[BOTTOM]*(*dimZ)) {
        current -= offset;
        int index = (haloWidth[FRONT] + current/((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[BOTTOM])) *
                            ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
                    ((current%((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[BOTTOM]))/
                     (*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + haloWidth[BOTTOM]) *
                            (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
                    current%(*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[LEFT];
        sync[offset+current] = vec[index];
        return;
    }
    // front
    offset += (*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*haloWidth[BOTTOM]*(*dimZ);
    if(current < offset + (*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*
                          (*dimY-haloWidth[TOP]-haloWidth[BOTTOM])*haloWidth[FRONT]) {
        current -= offset;
        int index = (haloWidth[FRONT] + current/((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*
                                                 (*dimY-haloWidth[TOP]-haloWidth[BOTTOM]))) *
                            ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
                    ((current%((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])))/
                     (*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[BOTTOM]) *
                            (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
                    current%(*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[LEFT];
        sync[offset+current] = vec[index];
        return;
    }
    // back
    offset += (*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])*haloWidth[FRONT];
    current -= offset;
    int index = (current/((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])) +
                (*dimZ) +haloWidth[FRONT]-haloWidth[BACK]) *
                        ((*dimX+haloWidth[LEFT]+haloWidth[RIGHT])*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])) +
                ((current%((*dimX-haloWidth[LEFT]-haloWidth[RIGHT])*(*dimY-haloWidth[TOP]-haloWidth[BOTTOM])))/
                 (*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[BOTTOM]) *
                        (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
                current%(*dimX-haloWidth[LEFT]-haloWidth[RIGHT]) + 2*haloWidth[LEFT];
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
        int index = (current/(haloWidth[LEFT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM]))) *
                            (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) *
                    (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
                    ((current%(haloWidth[LEFT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM]))) / haloWidth[LEFT]) *
                            (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
                    current%haloWidth[LEFT];
        vec[index] = sync[current];
        return;
    }
    int offset = haloWidth[LEFT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]);
    // right
    if(current < offset + haloWidth[RIGHT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])*
                                           (*dimZ+haloWidth[FRONT]+haloWidth[BACK])) {
        current -= offset;
        int index = (current/(haloWidth[RIGHT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM]))) *
                            (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) * (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
                    ((current%(haloWidth[RIGHT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM]))) / haloWidth[RIGHT]) *
                            (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
                    current%haloWidth[RIGHT] + *dimX+haloWidth[LEFT];
        vec[index] = sync[offset+current];
        return;
    }
    // top
    offset += haloWidth[RIGHT]*(*dimY+haloWidth[TOP]+haloWidth[BOTTOM])*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]);
    if(current < offset + haloWidth[TOP]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK])) {
        current -= offset;
        int index = (current/(haloWidth[TOP]*(*dimX))) *
                            (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) * (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
                    ((current%(haloWidth[TOP]*(*dimX))) / (*dimX) + *dimY+haloWidth[BOTTOM]) *
                            (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
                    current%(*dimX) + haloWidth[LEFT];
        vec[index] = sync[offset+current];
        return;
    }
    // bottom
    offset += haloWidth[TOP]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]);
    if(current < offset + haloWidth[BOTTOM]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK])) {
        current -= offset;
        int index = (current/(haloWidth[BOTTOM]*(*dimX))) *
                            (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) * (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
                    ((current%(haloWidth[BOTTOM]*(*dimX))) / (*dimX)) *
                            (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
                    current%(*dimX) + haloWidth[LEFT];
        vec[index] = sync[offset+current];
        return;
    }
    // front
    offset += haloWidth[BOTTOM]*(*dimX)*(*dimZ+haloWidth[FRONT]+haloWidth[BACK]);
    if(current < offset + haloWidth[FRONT]*(*dimX)*(*dimY)) {
        current -= offset;
        int index = (current/((*dimX)*(*dimY))) *
                            (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) * (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
                    ((current%((*dimX)*(*dimY))) / (*dimX) + haloWidth[BOTTOM]) *
                            (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
                    current%(*dimX) + haloWidth[LEFT];
        vec[index] = sync[offset+current];
        return;
    }
    // back
    offset += haloWidth[FRONT]*(*dimX)*(*dimY);
    current -= offset;
    int index = (current/((*dimX)*(*dimY)) + (*dimZ) + haloWidth[FRONT]) *
                        (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) * (*dimY+haloWidth[TOP]+haloWidth[BOTTOM]) +
                ((current%((*dimX)*(*dimY))) / (*dimX) + haloWidth[BOTTOM]) *
                        (*dimX+haloWidth[LEFT]+haloWidth[RIGHT]) +
                current%(*dimX) + haloWidth[LEFT];
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
