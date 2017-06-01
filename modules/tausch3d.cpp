#include "../tausch.h"
#include "tausch3d.h"

template <class real_t> Tausch3D<real_t>::Tausch3D(int *localDim, int *haloWidth, MPI_Datatype mpiDataType, int numBuffers, int valuesPerPoint, MPI_Comm comm) {

    MPI_Comm_dup(comm, &TAUSCH_COMM);

    // get MPI info
    MPI_Comm_rank(TAUSCH_COMM, &mpiRank);
    MPI_Comm_size(TAUSCH_COMM, &mpiSize);

    for(int i = 0; i < 3; ++i)
        this->localDim[i] = localDim[i];

    for(int i = 0; i < 6; ++i)
        this->haloWidth[i] = haloWidth[i];

    this->numBuffers = numBuffers;
    this->valuesPerPoint = valuesPerPoint;

    this->mpiDataType = mpiDataType;

}

template <class real_t> Tausch3D<real_t>::~Tausch3D() {
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

template <class real_t> void Tausch3D<real_t>::setLocalHaloInfoCpu(int numHaloParts, int **haloSpecs) {

    localHaloNumParts = numHaloParts;
    localHaloSpecs = new int*[numHaloParts];
    mpiSendBuffer = new real_t*[numHaloParts];
    mpiSendRequests = new MPI_Request[numHaloParts];
    numBuffersPacked =  new int[numHaloParts]{};

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecs[i] = new int[8];
        for(int j = 0; j < 8; ++j)
            localHaloSpecs[i][j] = haloSpecs[i][j];

        int size = numBuffers*valuesPerPoint*haloSpecs[i][3]*haloSpecs[i][4]*haloSpecs[i][5];

        mpiSendBuffer[i] = new real_t[size]{};
        MPI_Send_init(&mpiSendBuffer[i][0], size, mpiDataType, haloSpecs[i][6], haloSpecs[i][7], TAUSCH_COMM, &mpiSendRequests[i]);

    }

}

template <class real_t> void Tausch3D<real_t>::setRemoteHaloInfoCpu(int numHaloParts, int **haloSpecs) {

    remoteHaloNumParts = numHaloParts;
    remoteHaloSpecs = new int*[numHaloParts];
    mpiRecvBuffer = new real_t*[numHaloParts];
    mpiRecvRequests = new MPI_Request[numHaloParts];
    numBuffersUnpacked =  new int[numHaloParts]{};

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecs[i] = new int[8];
        for(int j = 0; j < 8; ++j)
            remoteHaloSpecs[i][j] = haloSpecs[i][j];

        int size = numBuffers*valuesPerPoint*haloSpecs[i][3]*haloSpecs[i][4]*haloSpecs[i][5];

        mpiRecvBuffer[i] = new real_t[size]{};
        MPI_Recv_init(&mpiRecvBuffer[i][0], size, mpiDataType, haloSpecs[i][6], haloSpecs[i][7], TAUSCH_COMM, &mpiRecvRequests[i]);

    }

}

template <class real_t> void Tausch3D<real_t>::postReceivesCpu() {

    for(int rem = 0; rem < remoteHaloNumParts; ++rem)
        MPI_Start(&mpiRecvRequests[rem]);

}

template <class real_t> void Tausch3D<real_t>::packNextSendBufferCpu(int id, real_t *buf) {

    if(numBuffersPacked[id] == numBuffers)
        numBuffersPacked[id] = 0;

    int size = localHaloSpecs[id][3] * localHaloSpecs[id][4] * localHaloSpecs[id][5];
    for(int s = 0; s < size; ++s) {
        int index = (s/(localHaloSpecs[id][3]*localHaloSpecs[id][4]) + localHaloSpecs[id][2]) *
                        (haloWidth[TAUSCH_LEFT]+localDim[TAUSCH_X]+haloWidth[TAUSCH_RIGHT])*
                        (haloWidth[TAUSCH_BOTTOM]+localDim[TAUSCH_Y]+haloWidth[TAUSCH_TOP]) +
                    ((s%(localHaloSpecs[id][3]*localHaloSpecs[id][4]))/localHaloSpecs[id][3] + localHaloSpecs[id][1]) *
                        (haloWidth[TAUSCH_LEFT]+localDim[TAUSCH_X]+haloWidth[TAUSCH_RIGHT]) +
                    s%localHaloSpecs[id][3] + localHaloSpecs[id][0];
        for(int val = 0; val < valuesPerPoint; ++val)
            mpiSendBuffer[id][numBuffersPacked[id]*valuesPerPoint*size + valuesPerPoint*s + val] = buf[valuesPerPoint*index + val];
    }
    ++numBuffersPacked[id];

}

template <class real_t> void Tausch3D<real_t>::sendCpu(int id) {

    if(numBuffersPacked[id] != numBuffers) {
        std::cerr << "[Tausch3D] ERROR: halo part " << id << " has " << numBuffersPacked[id] << " out of "
                  << numBuffers << " send buffers packed... Abort!" << std::endl;
        exit(1);
    }

    MPI_Start(&mpiSendRequests[id]);

}

template <class real_t> void Tausch3D<real_t>::recvCpu(int id) {
    numBuffersUnpacked[id] = 0;
    MPI_Wait(&mpiRecvRequests[id], MPI_STATUS_IGNORE);
}

template <class real_t> void Tausch3D<real_t>::unpackNextRecvBufferCpu(int id, real_t *buf) {

    int size = remoteHaloSpecs[id][3] * remoteHaloSpecs[id][4] * remoteHaloSpecs[id][5];
    for(int s = 0; s < size; ++s) {
        int index = (s/(remoteHaloSpecs[id][3]*remoteHaloSpecs[id][4]) + remoteHaloSpecs[id][2]) *
                        (haloWidth[TAUSCH_LEFT]+localDim[TAUSCH_X]+haloWidth[TAUSCH_RIGHT])*
                        (haloWidth[TAUSCH_BOTTOM]+localDim[TAUSCH_Y]+haloWidth[TAUSCH_TOP]) +
                    ((s%(remoteHaloSpecs[id][3]*remoteHaloSpecs[id][4]))/remoteHaloSpecs[id][3] + remoteHaloSpecs[id][1]) *
                        (haloWidth[TAUSCH_LEFT]+localDim[TAUSCH_X]+haloWidth[TAUSCH_RIGHT]) +
                    s%remoteHaloSpecs[id][3] + remoteHaloSpecs[id][0];
        for(int val = 0; val < valuesPerPoint; ++val)
            buf[valuesPerPoint*index + val] = mpiRecvBuffer[id][numBuffersUnpacked[id]*valuesPerPoint*size + valuesPerPoint*s + val];
    }
    ++numBuffersUnpacked[id];

}

template <class real_t> void Tausch3D<real_t>::packAndSendCpu(int id, real_t *buf) {
    packNextSendBufferCpu(id, buf);
    sendCpu(id);
}

template <class real_t> void Tausch3D<real_t>::recvAndUnpackCpu(int id, real_t *buf) {
    recvCpu(id);
    unpackNextRecvBufferCpu(id, buf);
}

template class Tausch3D<char>;
template class Tausch3D<char16_t>;
template class Tausch3D<char32_t>;
template class Tausch3D<wchar_t>;
template class Tausch3D<signed char>;
template class Tausch3D<short int>;
template class Tausch3D<int>;
template class Tausch3D<long>;
template class Tausch3D<long long>;
template class Tausch3D<unsigned char>;
template class Tausch3D<unsigned short int>;
template class Tausch3D<unsigned int>;
template class Tausch3D<unsigned long>;
template class Tausch3D<unsigned long long>;
template class Tausch3D<float>;
template class Tausch3D<double>;
template class Tausch3D<long double>;
template class Tausch3D<bool>;
