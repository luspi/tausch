#include "tausch2d.h"

template <class buf_t> Tausch2D<buf_t>::Tausch2D(size_t *localDim, MPI_Datatype mpiDataType,
                                                 size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm) {

    MPI_Comm_dup(comm, &TAUSCH_COMM);

    // get MPI info
    MPI_Comm_rank(TAUSCH_COMM, &mpiRank);
    MPI_Comm_size(TAUSCH_COMM, &mpiSize);

    this->localDim[0] = localDim[0];
    this->localDim[1] = localDim[1];

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

template <class buf_t> Tausch2D<buf_t>::~Tausch2D() {
    for(int i = 0; i < localHaloNumPartsCpu; ++i)
        delete[] mpiSendBuffer[i];
    for(int i = 0; i < remoteHaloNumPartsCpu; ++i)
        delete[] mpiRecvBuffer[i];
    delete[] localHaloSpecsCpu;
    delete[] mpiSendBuffer;
    delete[] remoteHaloSpecsCpu;
    delete[] mpiRecvBuffer;

    delete[] mpiSendRequests;
    delete[] numBuffersPackedCpu;
    delete[] mpiRecvRequests;
    delete[] numBuffersUnpackedCpu;

    delete[] setupMpiRecv;
    delete[] setupMpiSend;

    delete[] valuesPerPointPerBuffer;
}

template <class buf_t> void Tausch2D<buf_t>::setLocalHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumPartsCpu = numHaloParts;
    localHaloSpecsCpu = new TauschHaloSpec[numHaloParts];
    mpiSendBuffer = new buf_t*[numHaloParts];
    mpiSendRequests = new MPI_Request[numHaloParts];
    numBuffersPackedCpu =  new size_t[numHaloParts]{};
    setupMpiSend = new bool[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecsCpu[i].x = haloSpecs[i].x;
        localHaloSpecsCpu[i].y = haloSpecs[i].y;
        localHaloSpecsCpu[i].width = haloSpecs[i].width;
        localHaloSpecsCpu[i].height = haloSpecs[i].height;
        localHaloSpecsCpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].width*haloSpecs[i].height;
        mpiSendBuffer[i] = new buf_t[bufsize]{};

        setupMpiSend[i] = false;

    }

}

template <class buf_t> void Tausch2D<buf_t>::setRemoteHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsCpu = numHaloParts;
    remoteHaloSpecsCpu = new TauschHaloSpec[numHaloParts];
    mpiRecvBuffer = new buf_t*[numHaloParts];
    mpiRecvRequests = new MPI_Request[numHaloParts];
    numBuffersUnpackedCpu =  new size_t[numHaloParts]{};
    setupMpiRecv = new bool[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsCpu[i].x = haloSpecs[i].x;
        remoteHaloSpecsCpu[i].y = haloSpecs[i].y;
        remoteHaloSpecsCpu[i].width = haloSpecs[i].width;
        remoteHaloSpecsCpu[i].height = haloSpecs[i].height;
        remoteHaloSpecsCpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].width*haloSpecs[i].height;
        mpiRecvBuffer[i] = new buf_t[bufsize]{};

        setupMpiRecv[i] = false;

    }

}

template <class buf_t> void Tausch2D<buf_t>::postReceiveCpu(size_t id, int mpitag) {

    if(!setupMpiRecv[id]) {

        if(mpitag == -1) {
            std::cerr << "[Tausch2D] ERROR: MPI_Recv for halo region #" << id << " hasn't been posted before, missing mpitag... Abort!" << std::endl;
            exit(1);
        }

        setupMpiRecv[id] = true;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsCpu[id].width*remoteHaloSpecsCpu[id].height;
        MPI_Recv_init(&mpiRecvBuffer[id][0], bufsize, mpiDataType,
                      remoteHaloSpecsCpu[id].remoteMpiRank, mpitag, TAUSCH_COMM, &mpiRecvRequests[id]);

    }

    MPI_Start(&mpiRecvRequests[id]);

}

template <class buf_t> void Tausch2D<buf_t>::postAllReceivesCpu(int *mpitag) {

    if(mpitag == nullptr) {
        mpitag = new int[remoteHaloNumPartsCpu];
        for(int id = 0; id < remoteHaloNumPartsCpu; ++id)
            mpitag[id] = -1;
    }

    for(int id = 0; id < remoteHaloNumPartsCpu; ++id)
        postReceiveCpu(id,mpitag[id]);

}

template <class buf_t> void Tausch2D<buf_t>::packNextSendBufferCpu(size_t id, buf_t *buf) {

    if(numBuffersPackedCpu[id] == numBuffers)
        numBuffersPackedCpu[id] = 0;

    int size = localHaloSpecsCpu[id].width * localHaloSpecsCpu[id].height;
    for(int s = 0; s < size; ++s) {
        int index = (s/localHaloSpecsCpu[id].width + localHaloSpecsCpu[id].y)*localDim[TAUSCH_X] +
                    s%localHaloSpecsCpu[id].width + localHaloSpecsCpu[id].x;
        for(int val = 0; val < valuesPerPointPerBuffer[numBuffersPackedCpu[id]]; ++val) {
            int offset = 0;
            for(int b = 0; b < numBuffersPackedCpu[id]; ++b)
                offset += valuesPerPointPerBuffer[b]*size;
            mpiSendBuffer[id][offset + valuesPerPointPerBuffer[numBuffersPackedCpu[id]]*s + val] =
                    buf[valuesPerPointPerBuffer[numBuffersPackedCpu[id]]*index + val];
        }
    }
    ++numBuffersPackedCpu[id];

}

template <class buf_t> void Tausch2D<buf_t>::sendCpu(size_t id, int mpitag) {

    if(numBuffersPackedCpu[id] != numBuffers) {
        std::cerr << "[Tausch2D] ERROR: halo part " << id << " has " << numBuffersPackedCpu[id] << " out of "
                  << numBuffers << " send buffers packed... Abort!" << std::endl;
        exit(1);
    }

    if(!setupMpiSend[id]) {

        if(mpitag == -1) {
            std::cerr << "[Tausch2D] ERROR: MPI_Send for halo region #" << id << " hasn't been posted before, missing mpitag... Abort!" << std::endl;
            exit(1);
        }

        setupMpiSend[id] = true;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsCpu[id].width*localHaloSpecsCpu[id].height;
        MPI_Send_init(&mpiSendBuffer[id][0], bufsize, mpiDataType, localHaloSpecsCpu[id].remoteMpiRank,
                  mpitag, TAUSCH_COMM, &mpiSendRequests[id]);

    }

    MPI_Start(&mpiSendRequests[id]);

}

template <class buf_t> void Tausch2D<buf_t>::recvCpu(size_t id) {
    numBuffersUnpackedCpu[id] = 0;
    MPI_Wait(&mpiRecvRequests[id], MPI_STATUS_IGNORE);
}

template <class buf_t> void Tausch2D<buf_t>::unpackNextRecvBufferCpu(size_t id, buf_t *buf) {

    int size = remoteHaloSpecsCpu[id].width * remoteHaloSpecsCpu[id].height;
    for(int s = 0; s < size; ++s) {
        int index = (s/remoteHaloSpecsCpu[id].width + remoteHaloSpecsCpu[id].y)*localDim[TAUSCH_X] +
                    s%remoteHaloSpecsCpu[id].width + remoteHaloSpecsCpu[id].x;
        for(int val = 0; val < valuesPerPointPerBuffer[numBuffersUnpackedCpu[id]]; ++val) {
            int offset = 0;
            for(int b = 0; b < numBuffersUnpackedCpu[id]; ++b)
                offset += valuesPerPointPerBuffer[b]*size;
            buf[valuesPerPointPerBuffer[numBuffersUnpackedCpu[id]]*index + val] =
                    mpiRecvBuffer[id][offset + valuesPerPointPerBuffer[numBuffersUnpackedCpu[id]]*s + val];
        }
    }
    ++numBuffersUnpackedCpu[id];

}

template <class buf_t> void Tausch2D<buf_t>::packAndSendCpu(size_t id, buf_t *buf, int mpitag) {
    packNextSendBufferCpu(id, buf);
    sendCpu(id, mpitag);
}

template <class buf_t> void Tausch2D<buf_t>::recvAndUnpackCpu(size_t id, buf_t *buf) {
    recvCpu(id);
    unpackNextRecvBufferCpu(id, buf);
}

template class Tausch2D<char>;
template class Tausch2D<char16_t>;
template class Tausch2D<char32_t>;
template class Tausch2D<wchar_t>;
template class Tausch2D<signed char>;
template class Tausch2D<short int>;
template class Tausch2D<int>;
template class Tausch2D<long>;
template class Tausch2D<long long>;
template class Tausch2D<unsigned char>;
template class Tausch2D<unsigned short int>;
template class Tausch2D<unsigned int>;
template class Tausch2D<unsigned long>;
template class Tausch2D<unsigned long long>;
template class Tausch2D<float>;
template class Tausch2D<double>;
template class Tausch2D<long double>;
template class Tausch2D<bool>;
