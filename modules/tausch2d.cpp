#include "tausch2d.h"

template <class real_t> Tausch2D<real_t>::Tausch2D(int *localDim, int *haloWidth, int numBuffers, int valuesPerPoint, MPI_Comm comm) {

    MPI_Comm_dup(comm, &TAUSCH_COMM);

    // get MPI info
    MPI_Comm_rank(TAUSCH_COMM, &mpiRank);
    MPI_Comm_size(TAUSCH_COMM, &mpiSize);

    this->localDim[0] = localDim[0];
    this->localDim[1] = localDim[1];

    for(int i = 0; i < 4; ++i)
        this->haloWidth[i] = haloWidth[i];

    this->numBuffers = numBuffers;
    this->valuesPerPoint = valuesPerPoint;

    mpiDatatype = (std::is_same<real_t, float>::value ? MPI_FLOAT
                        : (std::is_same<real_t, int>::value ? MPI_INT
                        : (std::is_same<real_t, unsigned int>::value ? MPI_UNSIGNED
                        : (std::is_same<real_t, long>::value ? MPI_LONG
                        : (std::is_same<real_t, long long>::value ? MPI_LONG_LONG
                        : (std::is_same<real_t, long double>::value ? MPI_LONG_DOUBLE : MPI_DOUBLE))))));

}

template <class real_t> Tausch2D<real_t>::~Tausch2D() {
    for(int i = 0; i < localHaloNumParts; ++i) {
        delete[] localHaloSpecs[i];
        delete[] mpiSendBuffer[i];
    }
    for(int i = 0; i < remoteHaloNumParts; ++i) {
        delete[] remoteHaloSpecs[i];
        delete[] mpiRecvBuffer[i];
    }
    delete[] localHaloSpecs;
    delete[] mpiSendBuffer;
    delete[] remoteHaloSpecs;
    delete[] mpiRecvBuffer;

    delete[] mpiSendRequests;
    delete[] numBuffersPacked;
    delete[] mpiRecvRequests;
    delete[] numBuffersUnpacked;
}

template <class real_t> void Tausch2D<real_t>::setCpuLocalHaloInfo(int numHaloParts, int **haloSpecs) {

    localHaloNumParts = numHaloParts;
    localHaloSpecs = new int*[numHaloParts];
    mpiSendBuffer = new real_t*[numHaloParts];
    mpiSendRequests = new MPI_Request[numHaloParts];
    numBuffersPacked =  new int[numHaloParts]{};

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecs[i] = new int[6];
        for(int j = 0; j < 6; ++j)
            localHaloSpecs[i][j] = haloSpecs[i][j];

        mpiSendBuffer[i] = new real_t[numBuffers*valuesPerPoint*haloSpecs[i][2]*haloSpecs[i][3]]{};

        int size = localHaloSpecs[i][2] * localHaloSpecs[i][3];
        MPI_Send_init(&mpiSendBuffer[i][0], numBuffers*valuesPerPoint*size, mpiDatatype, localHaloSpecs[i][4], haloSpecs[i][5], TAUSCH_COMM, &mpiSendRequests[i]);

    }

}

template <class real_t> void Tausch2D<real_t>::setCpuRemoteHaloInfo(int numHaloParts, int **haloSpecs) {

    remoteHaloNumParts = numHaloParts;
    remoteHaloSpecs = new int*[numHaloParts];
    mpiRecvBuffer = new real_t*[numHaloParts];
    mpiRecvRequests = new MPI_Request[numHaloParts];
    numBuffersUnpacked =  new int[numHaloParts]{};

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecs[i] = new int[6];
        for(int j = 0; j < 6; ++j)
            remoteHaloSpecs[i][j] = haloSpecs[i][j];

        mpiRecvBuffer[i] = new real_t[numBuffers*valuesPerPoint*haloSpecs[i][2]*haloSpecs[i][3]]{};

        int size = remoteHaloSpecs[i][2] * remoteHaloSpecs[i][3];
        int sender = remoteHaloSpecs[i][4];
        MPI_Recv_init(&mpiRecvBuffer[i][0], numBuffers*valuesPerPoint*size, mpiDatatype, sender, haloSpecs[i][5], TAUSCH_COMM, &mpiRecvRequests[i]);

    }

}

template <class real_t> void Tausch2D<real_t>::postMpiReceives() {

    for(int rem = 0; rem < remoteHaloNumParts; ++rem)
        MPI_Start(&mpiRecvRequests[rem]);

}

template <class real_t> void Tausch2D<real_t>::packNextSendBuffer(int id, real_t *buf) {

    if(numBuffersPacked[id] == numBuffers)
        numBuffersPacked[id] = 0;

    int size = localHaloSpecs[id][2] * localHaloSpecs[id][3];
    for(int s = 0; s < size; ++s) {
        int index = (s/localHaloSpecs[id][2] + localHaloSpecs[id][1])*(localDim[TAUSCH_X] + haloWidth[TAUSCH_LEFT] + haloWidth[TAUSCH_RIGHT]) +
                    s%localHaloSpecs[id][2] + localHaloSpecs[id][0];
        for(int val = 0; val < valuesPerPoint; ++val)
            mpiSendBuffer[id][numBuffersPacked[id]*valuesPerPoint*size + valuesPerPoint*s + val] = buf[valuesPerPoint*index + val];
    }
    ++numBuffersPacked[id];

}

template <class real_t> void Tausch2D<real_t>::send(int id) {

    if(numBuffersPacked[id] != numBuffers) {
        std::cerr << "[Tausch2D] ERROR: halo part " << id << " has " << numBuffersPacked[id] << " out of "
                  << numBuffers << " send buffers packed... Abort!" << std::endl;
        exit(1);
    }

    MPI_Start(&mpiSendRequests[id]);

}

template <class real_t> void Tausch2D<real_t>::recv(int id) {
    numBuffersUnpacked[id] = 0;
    MPI_Wait(&mpiRecvRequests[id], MPI_STATUS_IGNORE);
}

template <class real_t> void Tausch2D<real_t>::unpackNextRecvBuffer(int id, real_t *buf) {

    int size = remoteHaloSpecs[id][2] * remoteHaloSpecs[id][3];
    for(int s = 0; s < size; ++s) {
        int index = (s/remoteHaloSpecs[id][2] + remoteHaloSpecs[id][1])*(localDim[TAUSCH_X] + haloWidth[TAUSCH_LEFT] + haloWidth[TAUSCH_RIGHT]) +
                    s%remoteHaloSpecs[id][2] + remoteHaloSpecs[id][0];
        for(int val = 0; val < valuesPerPoint; ++val)
            buf[valuesPerPoint*index + val] = mpiRecvBuffer[id][numBuffersUnpacked[id]*valuesPerPoint*size + valuesPerPoint*s + val];
    }
    ++numBuffersUnpacked[id];

}

template <class real_t> void Tausch2D<real_t>::packAndSend(int id, real_t *buf) {
    packNextSendBuffer(id, buf);
    send(id);
}

template <class real_t> void Tausch2D<real_t>::recvAndUnpack(int id, real_t *buf) {
    recv(id);
    unpackNextRecvBuffer(id, buf);
}

template class Tausch2D<double>;
template class Tausch2D<float>;
template class Tausch2D<int>;
template class Tausch2D<unsigned int>;
template class Tausch2D<long>;
template class Tausch2D<long long>;
template class Tausch2D<long double>;
