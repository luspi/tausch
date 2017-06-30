#include "tausch3d.h"

template <class buf_t> Tausch3D<buf_t>::Tausch3D(MPI_Datatype mpiDataType,
                                                   int numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm) {

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

template <class buf_t> Tausch3D<buf_t>::~Tausch3D() {
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

template <class buf_t> void Tausch3D<buf_t>::setLocalHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumParts = numHaloParts;
    localHaloSpecs = new TauschHaloSpec[numHaloParts];
    mpiSendBuffer = new buf_t*[numHaloParts];
    mpiSendRequests = new MPI_Request[numHaloParts];
    setupMpiSend = new bool[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecs[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecs[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecs[i].bufferDepth = haloSpecs[i].bufferDepth;
        localHaloSpecs[i].haloX = haloSpecs[i].haloX;
        localHaloSpecs[i].haloY = haloSpecs[i].haloY;
        localHaloSpecs[i].haloZ = haloSpecs[i].haloZ;
        localHaloSpecs[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecs[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecs[i].haloDepth = haloSpecs[i].haloDepth;
        localHaloSpecs[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight*haloSpecs[i].haloDepth;

        mpiSendBuffer[i] = new buf_t[bufsize]{};

        setupMpiSend[i] = false;

    }

}

template <class buf_t> void Tausch3D<buf_t>::setRemoteHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumParts = numHaloParts;
    remoteHaloSpecs = new TauschHaloSpec[numHaloParts];
    mpiRecvBuffer = new buf_t*[numHaloParts];
    mpiRecvRequests = new MPI_Request[numHaloParts];
    setupMpiRecv = new bool[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecs[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecs[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecs[i].bufferDepth = haloSpecs[i].bufferDepth;
        remoteHaloSpecs[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecs[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecs[i].haloZ = haloSpecs[i].haloZ;
        remoteHaloSpecs[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecs[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecs[i].haloDepth = haloSpecs[i].haloDepth;
        remoteHaloSpecs[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight*haloSpecs[i].haloDepth;

        mpiRecvBuffer[i] = new buf_t[bufsize]{};

        setupMpiRecv[i] = false;

    }

}

template <class buf_t> void Tausch3D<buf_t>::postReceiveCpu(size_t id, int mpitag) {

    if(!setupMpiRecv[id]) {

        if(mpitag == -1) {
            std::cerr << "[Tausch3D] ERROR: MPI_Recv for halo region #" << id << " hasn't been posted before, missing mpitag... Abort!" << std::endl;
            exit(1);
        }

        setupMpiRecv[id] = true;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecs[id].haloWidth*remoteHaloSpecs[id].haloHeight*remoteHaloSpecs[id].haloDepth;

        MPI_Recv_init(&mpiRecvBuffer[id][0], bufsize,
                      mpiDataType, remoteHaloSpecs[id].remoteMpiRank, mpitag, TAUSCH_COMM, &mpiRecvRequests[id]);

    }

    MPI_Start(&mpiRecvRequests[id]);

}

template <class buf_t> void Tausch3D<buf_t>::postAllReceivesCpu(int *mpitag) {

    if(mpitag == nullptr) {
        mpitag = new int[remoteHaloNumParts];
        for(int id = 0; id < remoteHaloNumParts; ++id)
            mpitag[id] = -1;
    }

    for(int id = 0; id < remoteHaloNumParts; ++id)
        postReceiveCpu(id,mpitag[id]);

}

template <class buf_t> void Tausch3D<buf_t>::packSendBufferCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    int size = region.width * region.height * region.depth;
    for(int s = 0; s < size; ++s) {
        int index = (s/(region.width*region.height) + localHaloSpecs[haloId].haloZ) * localHaloSpecs[haloId].bufferWidth * localHaloSpecs[haloId].bufferHeight +
                    ((s%(region.width*region.height))/localHaloSpecs[haloId].haloWidth + localHaloSpecs[haloId].haloY) * localHaloSpecs[haloId].bufferWidth +
                    s%region.width + localHaloSpecs[haloId].haloX;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b]*size;
            mpiSendBuffer[haloId][offset + valuesPerPointPerBuffer[bufferId]*s + val] =
                    buf[valuesPerPointPerBuffer[bufferId]*index + val];
        }
    }

}

template <class buf_t> void Tausch3D<buf_t>::sendCpu(size_t id, int mpitag) {


    if(!setupMpiSend[id]) {

        if(mpitag == -1) {
            std::cerr << "[Tausch3D] ERROR: MPI_Send for halo region #" << id << " hasn't been posted before, missing mpitag... Abort!" << std::endl;
            exit(1);
        }

        setupMpiSend[id] = true;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecs[id].haloWidth*localHaloSpecs[id].haloHeight*localHaloSpecs[id].haloDepth;

        MPI_Send_init(&mpiSendBuffer[id][0], bufsize,
                  mpiDataType, localHaloSpecs[id].remoteMpiRank, mpitag, TAUSCH_COMM, &mpiSendRequests[id]);

    }

    MPI_Start(&mpiSendRequests[id]);

}

template <class buf_t> void Tausch3D<buf_t>::recvCpu(size_t id) {
    MPI_Wait(&mpiRecvRequests[id], MPI_STATUS_IGNORE);
}

template <class buf_t> void Tausch3D<buf_t>::unpackRecvBufferCpu(size_t haloId, size_t bufferId, buf_t *buf) {

    int size = remoteHaloSpecs[haloId].haloWidth * remoteHaloSpecs[haloId].haloHeight * remoteHaloSpecs[haloId].haloDepth;
    for(int s = 0; s < size; ++s) {
        int index = (s/(remoteHaloSpecs[haloId].haloWidth*remoteHaloSpecs[haloId].haloHeight) + remoteHaloSpecs[haloId].haloZ) * remoteHaloSpecs[haloId].bufferWidth * remoteHaloSpecs[haloId].bufferHeight +
                    ((s%(remoteHaloSpecs[haloId].haloWidth*remoteHaloSpecs[haloId].haloHeight))/remoteHaloSpecs[haloId].haloWidth + remoteHaloSpecs[haloId].haloY) * remoteHaloSpecs[haloId].bufferWidth +
                    s%remoteHaloSpecs[haloId].haloWidth + remoteHaloSpecs[haloId].haloX;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b]*size;
            buf[valuesPerPointPerBuffer[bufferId]*index + val] =
                    mpiRecvBuffer[haloId][offset + valuesPerPointPerBuffer[bufferId]*s + val];
        }
    }

}

template <class buf_t> void Tausch3D<buf_t>::packAndSendCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region, int mpitag) {
    packSendBufferCpu(haloId, bufferId, buf, region);
    sendCpu(haloId, mpitag);
}

template <class buf_t> void Tausch3D<buf_t>::recvAndUnpackCpu(size_t haloId, size_t bufferId, buf_t *buf) {
    recvCpu(haloId);
    unpackRecvBufferCpu(haloId, bufferId, buf);
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
