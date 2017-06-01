#include "../tausch.h"
#include "tausch1d.h"

template <class real_t> Tausch1D<real_t>::Tausch1D(int *localDim, int *haloWidth, MPI_Datatype mpiDataType,
                                                   int numBuffers, int valuesPerPoint, MPI_Comm comm) {

    MPI_Comm_dup(comm, &TAUSCH_COMM);

    // get MPI info
    MPI_Comm_rank(TAUSCH_COMM, &mpiRank);
    MPI_Comm_size(TAUSCH_COMM, &mpiSize);

    this->localDim = localDim[0];

    for(int i = 0; i < 2; ++i)
        this->haloWidth[i] = haloWidth[i];

    this->numBuffers = numBuffers;
    this->valuesPerPoint = valuesPerPoint;
    this->mpiDataType = mpiDataType;

}

template <class real_t> Tausch1D<real_t>::~Tausch1D() {
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

template <class real_t> void Tausch1D<real_t>::setLocalHaloInfoCpu(int numHaloParts, int **haloSpecs) {

    localHaloNumParts = numHaloParts;
    localHaloSpecs = new int*[numHaloParts];
    mpiSendBuffer = new real_t*[numHaloParts];
    mpiSendRequests = new MPI_Request[numHaloParts];
    numBuffersPacked =  new int[numHaloParts]{};

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecs[i] = new int[4];
        for(int j = 0; j < 4; ++j)
            localHaloSpecs[i][j] = haloSpecs[i][j];

        mpiSendBuffer[i] = new real_t[numBuffers*valuesPerPoint*haloSpecs[i][1]]{};

    }

}

template <class real_t> void Tausch1D<real_t>::setRemoteHaloInfoCpu(int numHaloParts, int **haloSpecs) {

    remoteHaloNumParts = numHaloParts;
    remoteHaloSpecs = new int*[numHaloParts];
    mpiRecvBuffer = new real_t*[numHaloParts];
    mpiRecvRequests = new MPI_Request[numHaloParts];
    numBuffersUnpacked =  new int[numHaloParts]{};

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecs[i] = new int[3];
        for(int j = 0; j < 3; ++j)
            remoteHaloSpecs[i][j] = haloSpecs[i][j];

        mpiRecvBuffer[i] = new real_t[numBuffers*valuesPerPoint*remoteHaloSpecs[i][1]]{};

    }

}

template <class real_t> void Tausch1D<real_t>::postReceiveCpu(int id, int mpitag) {

    MPI_Irecv(&mpiRecvBuffer[id][0], numBuffers*valuesPerPoint*remoteHaloSpecs[id][1], mpiDataType, remoteHaloSpecs[id][2],
              mpitag, TAUSCH_COMM, &mpiRecvRequests[id]);

}

template <class real_t> void Tausch2D<real_t>::postAllReceivesCpu(int *mpitag) {

    for(int id = 0; id < remoteHaloNumParts; ++id)
        postReceiveCpu(id,mpitag[id]);

}

template <class real_t> void Tausch1D<real_t>::packNextSendBufferCpu(int id, real_t *buf) {

    if(numBuffersPacked[id] == numBuffers)
        numBuffersPacked[id] = 0;

    int size = localHaloSpecs[id][1];
    for(int s = 0; s < size; ++s) {
        int index = localHaloSpecs[id][0] + s;
        for(int val = 0; val < valuesPerPoint; ++val)
            mpiSendBuffer[id][numBuffersPacked[id]*valuesPerPoint*size + valuesPerPoint*s + val] = buf[valuesPerPoint*index + val];
    }
    ++numBuffersPacked[id];

}

template <class real_t> void Tausch1D<real_t>::sendCpu(int id, int mpitag) {

    if(numBuffersPacked[id] != numBuffers) {
        std::cerr << "[Tausch1D] ERROR: halo part " << id << " has " << numBuffersPacked[id] << " out of "
                  << numBuffers << " send buffers packed... Abort!" << std::endl;
        exit(1);
    }

    MPI_Isend(&mpiSendBuffer[id][0], numBuffers*valuesPerPoint*localHaloSpecs[id][1], mpiDataType, localHaloSpecs[id][2],
              mpitag, TAUSCH_COMM, &mpiSendRequests[id]);

}

template <class real_t> void Tausch1D<real_t>::recvCpu(int id) {
    numBuffersUnpacked[id] = 0;
    MPI_Wait(&mpiRecvRequests[id], MPI_STATUS_IGNORE);
}

template <class real_t> void Tausch1D<real_t>::unpackNextRecvBufferCpu(int id, real_t *buf) {

    int size = remoteHaloSpecs[id][1];
    for(int s = 0; s < size; ++s) {
        int index = remoteHaloSpecs[id][0] + s;
        for(int val = 0; val < valuesPerPoint; ++val)
            buf[valuesPerPoint*index + val] = mpiRecvBuffer[id][numBuffersUnpacked[id]*valuesPerPoint*size + valuesPerPoint*s + val];
    }
    ++numBuffersUnpacked[id];

}

template <class real_t> void Tausch1D<real_t>::packAndSendCpu(int id, int mpitag, real_t *buf) {
    packNextSendBufferCpu(id, buf);
    sendCpu(id, mpitag);
}

template <class real_t> void Tausch1D<real_t>::recvAndUnpackCpu(int id, real_t *buf) {
    recvCpu(id);
    unpackNextRecvBufferCpu(id, buf);
}

template class Tausch1D<char>;
template class Tausch1D<char16_t>;
template class Tausch1D<char32_t>;
template class Tausch1D<wchar_t>;
template class Tausch1D<signed char>;
template class Tausch1D<short int>;
template class Tausch1D<int>;
template class Tausch1D<long>;
template class Tausch1D<long long>;
template class Tausch1D<unsigned char>;
template class Tausch1D<unsigned short int>;
template class Tausch1D<unsigned int>;
template class Tausch1D<unsigned long>;
template class Tausch1D<unsigned long long>;
template class Tausch1D<float>;
template class Tausch1D<double>;
template class Tausch1D<long double>;
template class Tausch1D<bool>;
