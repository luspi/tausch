#include "tausch1d.h"

template <class buf_t> Tausch1D<buf_t>::Tausch1D(MPI_Datatype mpiDataType,
                                                   size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm) {

    MPI_Comm_dup(comm, &TAUSCH_COMM);

    // get MPI info
    MPI_Comm_rank(TAUSCH_COMM, &mpiRank);
    MPI_Comm_size(TAUSCH_COMM, &mpiSize);

    this->numBuffers = numBuffers;

    this->valuesPerPointPerBuffer = new size_t[numBuffers];
    if(valuesPerPointPerBuffer == nullptr)
        for(int i = 0; i < numBuffers; ++i)
            this->valuesPerPointPerBuffer[i] = 1;
    else
        for(int i = 0; i < numBuffers; ++i)
            this->valuesPerPointPerBuffer[i] = valuesPerPointPerBuffer[i];

    this->mpiDataType = mpiDataType;

}

template <class buf_t> Tausch1D<buf_t>::~Tausch1D() {
    for(int i = 0; i < localHaloNumParts; ++i)
        delete[] mpiSendBuffer[i];
    for(int i = 0; i < remoteHaloNumParts; ++i)
        delete[] mpiRecvBuffer[i];
    delete[] localHaloSpecs;
    delete[] mpiSendBuffer;
    delete[] remoteHaloSpecs;
    delete[] mpiRecvBuffer;

    delete[] mpiSendRequests;
    delete[] mpiRecvRequests;

    delete[] setupMpiRecv;
    delete[] setupMpiSend;

    delete[] valuesPerPointPerBuffer;
}

template <class buf_t> void Tausch1D<buf_t>::setLocalHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumParts = numHaloParts;
    localHaloSpecs = new TauschHaloSpec[numHaloParts];
    mpiSendBuffer = new buf_t*[numHaloParts];
    mpiSendRequests = new MPI_Request[numHaloParts];
    setupMpiSend = new bool[numHaloParts];
    setupMpiRecv = new bool[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecs[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecs[i].haloX = haloSpecs[i].haloX;
        localHaloSpecs[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecs[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth;
        mpiSendBuffer[i] = new buf_t[bufsize]{};

        setupMpiSend[i] = false;

    }

}

template <class buf_t> void Tausch1D<buf_t>::setRemoteHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumParts = numHaloParts;
    remoteHaloSpecs = new TauschHaloSpec[numHaloParts];
    mpiRecvBuffer = new buf_t*[numHaloParts];
    mpiRecvRequests = new MPI_Request[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecs[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecs[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecs[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecs[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth;
        mpiRecvBuffer[i] = new buf_t[bufsize]{};

        setupMpiRecv[i] = false;

    }

}

template <class buf_t> void Tausch1D<buf_t>::postReceiveCpu(size_t id, int mpitag) {

    if(!setupMpiRecv[id]) {

        if(mpitag == -1) {
            std::cerr << "[Tausch1D] ERROR: MPI_Recv for halo region #" << id << " hasn't been posted before, missing mpitag... Abort!" << std::endl;
            exit(1);
        }

        setupMpiRecv[id] = true;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecs[id].haloWidth;

        MPI_Recv_init(&mpiRecvBuffer[id][0], bufsize, mpiDataType, remoteHaloSpecs[id].remoteMpiRank,
                      mpitag, TAUSCH_COMM, &mpiRecvRequests[id]);

    }

    MPI_Start(&mpiRecvRequests[id]);

}

template <class buf_t> void Tausch1D<buf_t>::postAllReceivesCpu(int *mpitag) {

    if(mpitag == nullptr) {
        mpitag = new int[remoteHaloNumParts];
        for(int id = 0; id < remoteHaloNumParts; ++id)
            mpitag[id] = -1;
    }

    for(int id = 0; id < remoteHaloNumParts; ++id)
        postReceiveCpu(id,mpitag[id]);

}

template <class buf_t> void Tausch1D<buf_t>::packSendBufferCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    int size = region.width;
    for(int s = 0; s < size; ++s) {
        int index = localHaloSpecs[haloId].haloX + s;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b]*size;
            mpiSendBuffer[haloId][offset + valuesPerPointPerBuffer[bufferId]*s + val] =
                    buf[valuesPerPointPerBuffer[bufferId]*index + val];
        }
    }

}

template <class buf_t> void Tausch1D<buf_t>::sendCpu(size_t id, int mpitag) {

    if(!setupMpiSend[id]) {

        if(mpitag == -1) {
            std::cerr << "[Tausch1D] ERROR: MPI_Send for halo region #" << id << " hasn't been posted before, missing mpitag... Abort!" << std::endl;
            exit(1);
        }

        setupMpiSend[id] = true;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecs[id].haloWidth;

        MPI_Send_init(&mpiSendBuffer[id][0], bufsize, mpiDataType, localHaloSpecs[id].remoteMpiRank,
                      mpitag, TAUSCH_COMM, &mpiSendRequests[id]);

    }

    MPI_Start(&mpiSendRequests[id]);

}

template <class buf_t> void Tausch1D<buf_t>::recvCpu(size_t id) {
    MPI_Wait(&mpiRecvRequests[id], MPI_STATUS_IGNORE);
}

template <class buf_t> void Tausch1D<buf_t>::unpackRecvBufferCpu(size_t haloId, size_t bufferId, buf_t *buf) {

    int size = remoteHaloSpecs[haloId].haloWidth;
    for(int s = 0; s < size; ++s) {
        int index = remoteHaloSpecs[haloId].haloX + s;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b]*size;
            buf[valuesPerPointPerBuffer[bufferId]*index + val] =
                    mpiRecvBuffer[haloId][offset + valuesPerPointPerBuffer[bufferId]*s + val];
        }
    }

}

template <class buf_t> void Tausch1D<buf_t>::packAndSendCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region, int mpitag) {
    packSendBufferCpu(haloId, bufferId, buf, region);
    sendCpu(haloId, mpitag);
}

template <class buf_t> void Tausch1D<buf_t>::recvAndUnpackCpu(size_t haloId, size_t bufferId, buf_t *buf) {
    recvCpu(haloId);
    unpackRecvBufferCpu(haloId, bufferId, buf);
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
