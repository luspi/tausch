#include "tausch3d.h"

template <class buf_t> Tausch3D<buf_t>::Tausch3D(size_t *localDim, MPI_Datatype mpiDataType,
                                                   int numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm) {

    MPI_Comm_dup(comm, &TAUSCH_COMM);

    // get MPI info
    MPI_Comm_rank(TAUSCH_COMM, &mpiRank);
    MPI_Comm_size(TAUSCH_COMM, &mpiSize);

    for(int i = 0; i < 3; ++i)
        this->localDim[i] = localDim[i];

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
    delete[] numBuffersPacked;
    delete[] mpiRecvRequests;
    delete[] numBuffersUnpacked;

    delete[] setupMpiRecv;
    delete[] setupMpiSend;

    delete[] valuesPerPointPerBuffer;
}

template <class buf_t> void Tausch3D<buf_t>::setLocalHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumParts = numHaloParts;
    localHaloSpecs = new TauschHaloSpec[numHaloParts];
    mpiSendBuffer = new buf_t*[numHaloParts];
    mpiSendRequests = new MPI_Request[numHaloParts];
    numBuffersPacked =  new size_t[numHaloParts]{};
    setupMpiSend = new bool[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecs[i].x = haloSpecs[i].x;
        localHaloSpecs[i].y = haloSpecs[i].y;
        localHaloSpecs[i].z = haloSpecs[i].z;
        localHaloSpecs[i].width = haloSpecs[i].width;
        localHaloSpecs[i].height = haloSpecs[i].height;
        localHaloSpecs[i].depth = haloSpecs[i].depth;
        localHaloSpecs[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].width*haloSpecs[i].height*haloSpecs[i].depth;

        mpiSendBuffer[i] = new buf_t[bufsize]{};

        setupMpiSend[i] = false;

    }

}

template <class buf_t> void Tausch3D<buf_t>::setRemoteHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumParts = numHaloParts;
    remoteHaloSpecs = new TauschHaloSpec[numHaloParts];
    mpiRecvBuffer = new buf_t*[numHaloParts];
    mpiRecvRequests = new MPI_Request[numHaloParts];
    numBuffersUnpacked =  new size_t[numHaloParts]{};
    setupMpiRecv = new bool[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecs[i].x = haloSpecs[i].x;
        remoteHaloSpecs[i].y = haloSpecs[i].y;
        remoteHaloSpecs[i].z = haloSpecs[i].z;
        remoteHaloSpecs[i].width = haloSpecs[i].width;
        remoteHaloSpecs[i].height = haloSpecs[i].height;
        remoteHaloSpecs[i].depth = haloSpecs[i].depth;
        remoteHaloSpecs[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].width*haloSpecs[i].height*haloSpecs[i].depth;

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
            bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecs[id].width*remoteHaloSpecs[id].height*remoteHaloSpecs[id].depth;

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

template <class buf_t> void Tausch3D<buf_t>::packNextSendBufferCpu(size_t id, buf_t *buf) {

    if(numBuffersPacked[id] == numBuffers)
        numBuffersPacked[id] = 0;

    int size = localHaloSpecs[id].width * localHaloSpecs[id].height * localHaloSpecs[id].depth;
    for(int s = 0; s < size; ++s) {
        int index = (s/(localHaloSpecs[id].width*localHaloSpecs[id].height) + localHaloSpecs[id].z) * localDim[TAUSCH_X] * localDim[TAUSCH_Y] +
                    ((s%(localHaloSpecs[id].width*localHaloSpecs[id].height))/localHaloSpecs[id].width + localHaloSpecs[id].y) * localDim[TAUSCH_X] +
                    s%localHaloSpecs[id].width + localHaloSpecs[id].x;
        for(int val = 0; val < valuesPerPointPerBuffer[numBuffersPacked[id]]; ++val) {
            int offset = 0;
            for(int b = 0; b < numBuffersPacked[id]; ++b)
                offset += valuesPerPointPerBuffer[b]*size;
            mpiSendBuffer[id][offset + valuesPerPointPerBuffer[numBuffersPacked[id]]*s + val] =
                    buf[valuesPerPointPerBuffer[numBuffersPacked[id]]*index + val];
        }
    }
    ++numBuffersPacked[id];

}

template <class buf_t> void Tausch3D<buf_t>::sendCpu(size_t id, int mpitag) {

    if(numBuffersPacked[id] != numBuffers) {
        std::cerr << "[Tausch3D] ERROR: halo part " << id << " has " << numBuffersPacked[id] << " out of "
                  << numBuffers << " send buffers packed... Abort!" << std::endl;
        exit(1);
    }

    if(!setupMpiSend[id]) {

        if(mpitag == -1) {
            std::cerr << "[Tausch3D] ERROR: MPI_Send for halo region #" << id << " hasn't been posted before, missing mpitag... Abort!" << std::endl;
            exit(1);
        }

        setupMpiSend[id] = true;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecs[id].width*localHaloSpecs[id].height*localHaloSpecs[id].depth;

        MPI_Send_init(&mpiSendBuffer[id][0], bufsize,
                  mpiDataType, localHaloSpecs[id].remoteMpiRank, mpitag, TAUSCH_COMM, &mpiSendRequests[id]);

    }

    MPI_Start(&mpiSendRequests[id]);

}

template <class buf_t> void Tausch3D<buf_t>::recvCpu(size_t id) {
    numBuffersUnpacked[id] = 0;
    MPI_Wait(&mpiRecvRequests[id], MPI_STATUS_IGNORE);
}

template <class buf_t> void Tausch3D<buf_t>::unpackNextRecvBufferCpu(size_t id, buf_t *buf) {

    int size = remoteHaloSpecs[id].width * remoteHaloSpecs[id].height * remoteHaloSpecs[id].depth;
    for(int s = 0; s < size; ++s) {
        int index = (s/(remoteHaloSpecs[id].width*remoteHaloSpecs[id].height) + remoteHaloSpecs[id].z) * localDim[TAUSCH_X] * localDim[TAUSCH_Y] +
                    ((s%(remoteHaloSpecs[id].width*remoteHaloSpecs[id].height))/remoteHaloSpecs[id].width + remoteHaloSpecs[id].y) * localDim[TAUSCH_X] +
                    s%remoteHaloSpecs[id].width + remoteHaloSpecs[id].x;
        for(int val = 0; val < valuesPerPointPerBuffer[numBuffersUnpacked[id]]; ++val) {
            int offset = 0;
            for(int b = 0; b < numBuffersUnpacked[id]; ++b)
                offset += valuesPerPointPerBuffer[b]*size;
            buf[valuesPerPointPerBuffer[numBuffersUnpacked[id]]*index + val] =
                    mpiRecvBuffer[id][offset + valuesPerPointPerBuffer[numBuffersUnpacked[id]]*s + val];
        }
    }
    ++numBuffersUnpacked[id];

}

template <class buf_t> void Tausch3D<buf_t>::packAndSendCpu(size_t id, buf_t *buf, int mpitag) {
    packNextSendBufferCpu(id, buf);
    sendCpu(id, mpitag);
}

template <class buf_t> void Tausch3D<buf_t>::recvAndUnpackCpu(size_t id, buf_t *buf) {
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
