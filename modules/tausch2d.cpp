#include "tausch2d.h"
#include <thread>

template <class buf_t> Tausch2D<buf_t>::Tausch2D(MPI_Datatype mpiDataType,
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

    setupCpuWithCpu = false;
#ifdef TAUSCH_OPENCL
    setupCpuWithGpu = false;
    setupGpuWithCpu = false;
    setupGpuWithGpu = false;
#endif

}

template <class buf_t> Tausch2D<buf_t>::~Tausch2D() {

    if(setupCpuWithCpu) {

        for(int i = 0; i < localHaloNumPartsCpuWithCpu; ++i)
            delete[] mpiSendBufferCpuWithCpu[i];
        delete[] localHaloSpecsCpuWithCpu;
        delete[] mpiSendBufferCpuWithCpu;
        delete[] mpiSendRequestsCpuWithCpu;
        delete[] setupMpiSendCpuWithCpu;

        for(int i = 0; i < remoteHaloNumPartsCpuWithCpu; ++i)
            delete[] mpiRecvBufferCpuWithCpu[i];
        delete[] remoteHaloSpecsCpuWithCpu;
        delete[] mpiRecvBufferCpuWithCpu;
        delete[] mpiRecvRequestsCpuWithCpu;
        delete[] setupMpiRecvCpuWithCpu;

    }

#ifdef TAUSCH_OPENCL

    if(setupCpuWithGpu) {

        for(int i = 0; i < localHaloNumPartsCpuWithGpu; ++i)
            delete[] sendBufferCpuWithGpu[i];
        delete[] localHaloSpecsCpuWithGpu;
        delete[] sendBufferCpuWithGpu;
        delete[] msgtagsCpuToGpu;

        for(int i = 0; i < remoteHaloNumPartsCpuWithGpu; ++i)
            delete[] recvBufferCpuWithGpu[i];
        delete[] remoteHaloSpecsCpuWithGpu;
        delete[] recvBufferCpuWithGpu;

    }

    if(setupGpuWithCpu) {

        for(int i = 0; i < localHaloNumPartsGpuWithCpu; ++i)
            delete[] sendBufferGpuWithCpu[i];
        delete[] localHaloSpecsGpuWithCpu;
        delete[] sendBufferGpuWithCpu;
        delete[] msgtagsGpuToCpu;
        delete[] cl_sendBufferGpuWithCpu;
        delete[] cl_localHaloSpecsGpuWithCpu;

        for(int i = 0; i < remoteHaloNumPartsGpuWithCpu; ++i)
            delete[] recvBufferGpuWithCpu[i];
        delete[] remoteHaloSpecsGpuWithCpu;
        delete[] recvBufferGpuWithCpu;
        delete[] cl_recvBufferCpuWithGpu;
        delete[] cl_remoteHaloSpecsGpuWithCpu;

    }

    if(setupGpuWithGpu) {

        for(int i = 0; i < localHaloNumPartsGpuWithGpu; ++i)
            delete[] mpiSendBufferGpuWithGpu[i];
        delete[] localHaloSpecsGpuWithGpu;
        delete[] mpiSendBufferGpuWithGpu;
        delete[] mpiSendRequestsGpuWithGpu;
        delete[] setupMpiSendGpuWithGpu;
        delete[] cl_sendBufferGpuWithGpu;
        delete[] cl_localHaloSpecsGpuWithGpu;

        for(int i = 0; i < remoteHaloNumPartsGpuWithGpu; ++i)
            delete[] mpiRecvBufferGpuWithGpu[i];
        delete[] remoteHaloSpecsGpuWithGpu;
        delete[] mpiRecvBufferGpuWithGpu;
        delete[] mpiRecvRequestsGpuWithGpu;
        delete[] setupMpiRecvGpuWithGpu;
        delete[] cl_recvBufferGpuWithGpu;
        delete[] cl_remoteHaloSpecsGpuWithGpu;

    }

#endif

    delete[] valuesPerPointPerBuffer;
}

/////////////////////////////////////////////
/// PUBLIC API FUNCTION
/////////////////////////////////////////////

template <class buf_t> void Tausch2D<buf_t>::setLocalHaloInfo(TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return setLocalHaloInfoCpu(numHaloParts, haloSpecs);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return setLocalHaloInfoCpuForGpu(numHaloParts, haloSpecs);
    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return setLocalHaloInfoGpu(numHaloParts, haloSpecs);
    if(flags == TAUSCH_GPU+TAUSCH_WITHGPU)
        return setLocalHaloInfoGpuWithGpu(numHaloParts, haloSpecs);
#endif

    std::cerr << "Tausch2D::setLocalHaloInfo :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch2D<buf_t>::setRemoteHaloInfo(TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return setRemoteHaloInfoCpu(numHaloParts, haloSpecs);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return setRemoteHaloInfoCpuForGpu(numHaloParts, haloSpecs);
    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return setRemoteHaloInfoGpu(numHaloParts, haloSpecs);
    if(flags == TAUSCH_GPU+TAUSCH_WITHGPU)
        return setRemoteHaloInfoGpuWithGpu(numHaloParts, haloSpecs);
#endif

    std::cerr << "Tausch2D::setRemoteHaloInfo :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch2D<buf_t>::postReceive(TauschDeviceDirection flags, size_t haloId, int msgtag) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return postReceiveCpu(haloId, msgtag);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return postReceiveCpuForGpu(haloId, msgtag);
    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return postReceiveGpu(haloId, msgtag);
    if(flags == TAUSCH_GPU+TAUSCH_WITHGPU)
        return postReceiveGpuWithGpu(haloId, msgtag);
#endif

    std::cerr << "Tausch2D::postReceive :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch2D<buf_t>::postAllReceives(TauschDeviceDirection flags, int *msgtag) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return postAllReceivesCpu(msgtag);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return postAllReceivesCpuForGpu(msgtag);
    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return postAllReceivesGpu(msgtag);
    if(flags == TAUSCH_GPU+TAUSCH_WITHGPU)
        return postAllReceivesGpuWithGpu(msgtag);
#endif

    std::cerr << "Tausch2D::postAllReceives :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch2D<buf_t>::packSendBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf,
                                                            TauschPackRegion region) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return packSendBufferCpu(haloId, bufferId, buf, region);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return packSendBufferCpuToGpu(haloId, bufferId, buf, region);
#endif

    std::cerr << "Tausch2D::postReceive(buf_t) :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch2D<buf_t>::packSendBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.width = localHaloSpecsCpuWithCpu[haloId].haloWidth;
        region.height = localHaloSpecsCpuWithCpu[haloId].haloHeight;
        return packSendBufferCpu(haloId, bufferId, buf, region);
    }
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.width = localHaloSpecsCpuWithGpu[haloId].haloWidth;
        region.height = localHaloSpecsCpuWithGpu[haloId].haloHeight;
        return packSendBufferCpuToGpu(haloId, bufferId, buf, region);
    }
#endif

    std::cerr << "Tausch2D::postReceive(buf_t) :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::packSendBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, cl::Buffer buf) {

    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return packSendBufferGpuToCpu(haloId, bufferId, buf);
    if(flags == TAUSCH_GPU+TAUSCH_WITHGPU)
        return packSendBufferGpuWithGpu(haloId, bufferId, buf);

    std::cerr << "Tausch2D::postReceive(cl::Buffer) :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}
#endif

template <class buf_t> void Tausch2D<buf_t>::send(TauschDeviceDirection flags, size_t haloId, int msgtag) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return sendCpu(haloId, msgtag);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return sendCpuToGpu(haloId, msgtag);
    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return sendGpuToCpu(haloId, msgtag);
    if(flags == TAUSCH_GPU+TAUSCH_WITHGPU)
        return sendGpuWithGpu(haloId, msgtag);
#endif

    std::cerr << "Tausch2D::send :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch2D<buf_t>::recv(TauschDeviceDirection flags, size_t haloId) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return recvCpu(haloId);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return recvGpuToCpu(haloId);
    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return recvCpuToGpu(haloId);
    if(flags == TAUSCH_GPU+TAUSCH_WITHGPU)
        return recvGpuWithGpu(haloId);
#endif

    std::cerr << "Tausch2D::recv :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch2D<buf_t>::unpackRecvBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf,
                                                              TauschPackRegion region) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return unpackRecvBufferCpu(haloId, bufferId, buf, region);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return unpackRecvBufferGpuToCpu(haloId, bufferId, buf, region);
#endif

    std::cerr << "Tausch2D::recv(buf_t) :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch2D<buf_t>::unpackRecvBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.width = remoteHaloSpecsCpuWithCpu[haloId].haloWidth;
        region.height = remoteHaloSpecsCpuWithCpu[haloId].haloHeight;
        return unpackRecvBufferCpu(haloId, bufferId, buf, region);
    }
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.width = remoteHaloSpecsCpuWithGpu[haloId].haloWidth;
        region.height = remoteHaloSpecsCpuWithGpu[haloId].haloHeight;
        return unpackRecvBufferGpuToCpu(haloId, bufferId, buf, region);
    }
#endif

    std::cerr << "Tausch2D::recv(buf_t) :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}
#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::unpackRecvBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, cl::Buffer buf) {

    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return unpackRecvBufferCpuToGpu(haloId, bufferId, buf);
    if(flags == TAUSCH_GPU+TAUSCH_WITHGPU)
        return unpackRecvBufferGpuWithGpu(haloId, bufferId, buf);

    std::cerr << "Tausch2D::recv(cl::Buffer) :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}
#endif

template <class buf_t> void Tausch2D<buf_t>::packAndSend(TauschDeviceDirection flags, size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag){

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return packAndSendCpu(haloId, buf, region, msgtag);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return packAndSendCpuForGpu(haloId, buf, region, msgtag);
#endif

}

template <class buf_t> void Tausch2D<buf_t>::packAndSend(TauschDeviceDirection flags, size_t haloId, buf_t *buf, int msgtag) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.width = localHaloSpecsCpuWithCpu[haloId].haloWidth;
        region.height = localHaloSpecsCpuWithCpu[haloId].haloHeight;
        return packAndSendCpu(haloId, buf, region, msgtag);
    }
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.width = localHaloSpecsCpuWithGpu[haloId].haloWidth;
        region.height = localHaloSpecsCpuWithGpu[haloId].haloHeight;
        return packAndSendCpuForGpu(haloId, buf, region, msgtag);
    }
#endif

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::packAndSend(TauschDeviceDirection flags, size_t haloId, cl::Buffer buf, int msgtag) {

    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return packAndSendGpu(haloId, buf, msgtag);
    if(flags == TAUSCH_GPU+TAUSCH_WITHGPU)
        return packAndSendGpuWithGpu(haloId, buf, msgtag);

}
#endif

template <class buf_t> void Tausch2D<buf_t>::recvAndUnpack(TauschDeviceDirection flags, size_t haloId, buf_t *buf, TauschPackRegion region) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return recvAndUnpackCpu(haloId, buf, region);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return recvAndUnpackCpuForGpu(haloId, buf, region);
#endif

}

template <class buf_t> void Tausch2D<buf_t>::recvAndUnpack(TauschDeviceDirection flags, size_t haloId, buf_t *buf) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.width = remoteHaloSpecsCpuWithCpu[haloId].haloWidth;
        region.height = remoteHaloSpecsCpuWithCpu[haloId].haloHeight;
        return recvAndUnpackCpu(haloId, buf, region);
    }
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.width = remoteHaloSpecsCpuWithGpu[haloId].haloWidth;
        region.height = remoteHaloSpecsCpuWithGpu[haloId].haloHeight;
        return recvAndUnpackCpuForGpu(haloId, buf, region);
    }
#endif

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::recvAndUnpack(TauschDeviceDirection flags, size_t haloId, cl::Buffer buf) {

    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return recvAndUnpackGpu(haloId, buf);
    if(flags == TAUSCH_GPU+TAUSCH_WITHGPU)
        return recvAndUnpackGpuWithGpu(haloId, buf);

}
#endif


template <class buf_t> TauschPackRegion Tausch2D<buf_t>::createFilledPackRegion(size_t x, size_t width) {
    TauschPackRegion region;
    region.x = x;
    region.width = width;
    return region;
}

template <class buf_t> TauschPackRegion Tausch2D<buf_t>::createFilledPackRegion(size_t x, size_t y, size_t width, size_t height) {
    TauschPackRegion region;
    region.x = x;
    region.width = width;
    return region;
}

template <class buf_t> TauschPackRegion Tausch2D<buf_t>::createFilledPackRegion(size_t x, size_t y, size_t z,
                                                                                size_t width, size_t height, size_t depth) {
    TauschPackRegion region;
    region.x = x;
    region.y = y;
    region.z = z;
    region.width = width;
    region.height = height;
    region.depth = depth;
    return region;
}

template <class buf_t> TauschHaloSpec Tausch2D<buf_t>::createFilledHaloSpec(size_t bufferWidth, size_t haloX, size_t haloWidth, int remoteMpiRank) {
    TauschHaloSpec halo;
    halo.bufferWidth = bufferWidth;
    halo.haloX = haloX;
    halo.haloWidth = haloWidth;
    halo.remoteMpiRank = remoteMpiRank;
    return halo;
}

template <class buf_t> TauschHaloSpec Tausch2D<buf_t>::createFilledHaloSpec(size_t bufferWidth, size_t bufferHeight, size_t haloX, size_t haloY,
                                                                            size_t haloWidth, size_t haloHeight, int remoteMpiRank) {
    TauschHaloSpec halo;
    halo.bufferWidth = bufferWidth;
    halo.bufferHeight = bufferHeight;
    halo.haloX = haloX;
    halo.haloY = haloY;
    halo.haloWidth = haloWidth;
    halo.haloHeight = haloHeight;
    halo.remoteMpiRank = remoteMpiRank;
    return halo;
}

template <class buf_t> TauschHaloSpec Tausch2D<buf_t>::createFilledHaloSpec(size_t bufferWidth, size_t bufferHeight, size_t bufferDepth,
                                                                            size_t haloX, size_t haloY, size_t haloZ,
                                                                            size_t haloWidth, size_t haloHeight, size_t haloDepth, int remoteMpiRank){
    TauschHaloSpec halo;
    halo.bufferWidth = bufferWidth;
    halo.bufferHeight = bufferHeight;
    halo.bufferDepth = bufferDepth;
    halo.haloX = haloX;
    halo.haloY = haloY;
    halo.haloZ = haloZ;
    halo.haloWidth = haloWidth;
    halo.haloHeight = haloHeight;
    halo.haloDepth = haloDepth;
    halo.remoteMpiRank = remoteMpiRank;
    return halo;
}



//////////////////////////////////////
/// PRIVATE MEMBER FUNCTIONS
//////////////////////////////////////



////////////////////////
/// Set local halo info

template <class buf_t> void Tausch2D<buf_t>::setLocalHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumPartsCpuWithCpu = numHaloParts;
    localHaloSpecsCpuWithCpu = new TauschHaloSpec[numHaloParts];
    mpiSendBufferCpuWithCpu = new buf_t*[numHaloParts];
    mpiSendRequestsCpuWithCpu = new MPI_Request[numHaloParts];
    setupMpiSendCpuWithCpu = new bool[numHaloParts];

    setupCpuWithCpu = true;

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecsCpuWithCpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsCpuWithCpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecsCpuWithCpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsCpuWithCpu[i].haloY = haloSpecs[i].haloY;
        localHaloSpecsCpuWithCpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsCpuWithCpu[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecsCpuWithCpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        mpiSendBufferCpuWithCpu[i] = new buf_t[bufsize]{};

        setupMpiSendCpuWithCpu[i] = false;

    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::setLocalHaloInfoCpuForGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumPartsCpuWithGpu = numHaloParts;
    localHaloSpecsCpuWithGpu = new TauschHaloSpec[numHaloParts];
    sendBufferCpuWithGpu = new std::atomic<buf_t>*[numHaloParts];
    msgtagsCpuToGpu = new std::atomic<int>[numHaloParts]{};

    setupCpuWithGpu = true;

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecsCpuWithGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsCpuWithGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecsCpuWithGpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsCpuWithGpu[i].haloY = haloSpecs[i].haloY;
        localHaloSpecsCpuWithGpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsCpuWithGpu[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecsCpuWithGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        sendBufferCpuWithGpu[i] = new std::atomic<buf_t>[bufsize]{};

    }

}

template <class buf_t> void Tausch2D<buf_t>::setLocalHaloInfoGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumPartsGpuWithCpu = numHaloParts;
    localHaloSpecsGpuWithCpu = new TauschHaloSpec[numHaloParts];
    sendBufferGpuWithCpu = new std::atomic<buf_t>*[numHaloParts];
    msgtagsGpuToCpu = new std::atomic<int>[numHaloParts]{};

    setupGpuWithCpu = true;

    try {
        cl_sendBufferGpuWithCpu = new cl::Buffer[numHaloParts];
        cl_localHaloSpecsGpuWithCpu = new cl::Buffer[numHaloParts];
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: setLocalHaloInfoGpu() (1) :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecsGpuWithCpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsGpuWithCpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecsGpuWithCpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsGpuWithCpu[i].haloY = haloSpecs[i].haloY;
        localHaloSpecsGpuWithCpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsGpuWithCpu[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecsGpuWithCpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        sendBufferGpuWithCpu[i] = new std::atomic<buf_t>[bufsize]{};

        size_t tmpHaloSpecs[6] = {haloSpecs[i].haloX, haloSpecs[i].haloY, haloSpecs[i].haloWidth, haloSpecs[i].haloHeight,
                                  haloSpecs[i].bufferWidth, haloSpecs[i].bufferHeight};

        try {
            cl_sendBufferGpuWithCpu[i] = cl::Buffer(cl_context, CL_MEM_READ_WRITE, bufsize*sizeof(buf_t));
            cl_localHaloSpecsGpuWithCpu[i] = cl::Buffer(cl_context, &tmpHaloSpecs[0], &tmpHaloSpecs[6], true);
        } catch(cl::Error error) {
            std::cerr << "Tausch2D :: setLocalHaloInfoGpu() (2) :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")"
                      << std::endl;
            exit(1);
        }

    }

}

template <class buf_t> void Tausch2D<buf_t>::setLocalHaloInfoGpuWithGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumPartsGpuWithGpu = numHaloParts;
    localHaloSpecsGpuWithGpu = new TauschHaloSpec[numHaloParts];
    mpiSendBufferGpuWithGpu = new buf_t*[numHaloParts];
    mpiSendRequestsGpuWithGpu = new MPI_Request[numHaloParts];
    setupMpiSendGpuWithGpu = new bool[numHaloParts];

    setupGpuWithGpu = true;

    try {
        cl_sendBufferGpuWithGpu = new cl::Buffer[numHaloParts];
        cl_localHaloSpecsGpuWithGpu = new cl::Buffer[numHaloParts];
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: setLocalHaloInfoGpuWithGpu() (1) :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecsGpuWithGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsGpuWithGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecsGpuWithGpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsGpuWithGpu[i].haloY = haloSpecs[i].haloY;
        localHaloSpecsGpuWithGpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsGpuWithGpu[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecsGpuWithGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        mpiSendBufferGpuWithGpu[i] = new buf_t[bufsize]{};

        setupMpiSendGpuWithGpu[i] = false;

        size_t tmpHaloSpecs[6] = {haloSpecs[i].haloX, haloSpecs[i].haloY, haloSpecs[i].haloWidth, haloSpecs[i].haloHeight,
                                  haloSpecs[i].bufferWidth, haloSpecs[i].bufferHeight};

        try {
            cl_sendBufferGpuWithGpu[i] = cl::Buffer(cl_context, CL_MEM_READ_WRITE, bufsize*sizeof(buf_t));
            cl_localHaloSpecsGpuWithGpu[i] = cl::Buffer(cl_context, &tmpHaloSpecs[0], &tmpHaloSpecs[6], true);
        } catch(cl::Error error) {
            std::cerr << "Tausch2D :: setLocalHaloInfoGpuWithGpu() (2) :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")"
                      << std::endl;
            exit(1);
        }

    }

}
#endif


////////////////////////
/// Set remote halo info

template <class buf_t> void Tausch2D<buf_t>::setRemoteHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsCpuWithCpu = numHaloParts;
    remoteHaloSpecsCpuWithCpu = new TauschHaloSpec[numHaloParts];
    mpiRecvBufferCpuWithCpu = new buf_t*[numHaloParts];
    mpiRecvRequestsCpuWithCpu = new MPI_Request[numHaloParts];
    setupMpiRecvCpuWithCpu = new bool[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsCpuWithCpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsCpuWithCpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecsCpuWithCpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsCpuWithCpu[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecsCpuWithCpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsCpuWithCpu[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecsCpuWithCpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        mpiRecvBufferCpuWithCpu[i] = new buf_t[bufsize]{};

        setupMpiRecvCpuWithCpu[i] = false;

    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::setRemoteHaloInfoCpuForGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsCpuWithGpu = numHaloParts;
    remoteHaloSpecsCpuWithGpu = new TauschHaloSpec[numHaloParts];
    recvBufferCpuWithGpu = new buf_t*[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsCpuWithGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsCpuWithGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecsCpuWithGpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsCpuWithGpu[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecsCpuWithGpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsCpuWithGpu[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecsCpuWithGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        recvBufferCpuWithGpu[i] = new buf_t[bufsize];

    }

}

template <class buf_t> void Tausch2D<buf_t>::setRemoteHaloInfoGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsGpuWithCpu = numHaloParts;
    remoteHaloSpecsGpuWithCpu = new TauschHaloSpec[numHaloParts];
    recvBufferGpuWithCpu = new buf_t*[numHaloParts];

    try {
        cl_recvBufferCpuWithGpu = new cl::Buffer[remoteHaloNumPartsGpuWithCpu];
        cl_remoteHaloSpecsGpuWithCpu = new cl::Buffer[numHaloParts];
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: enableOpenCL() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsGpuWithCpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsGpuWithCpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecsGpuWithCpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsGpuWithCpu[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecsGpuWithCpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsGpuWithCpu[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecsGpuWithCpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        recvBufferGpuWithCpu[i] = new buf_t[bufsize];

        size_t tmpHaloSpecs[6] = {haloSpecs[i].haloX, haloSpecs[i].haloY, haloSpecs[i].haloWidth, haloSpecs[i].haloHeight,
                                  haloSpecs[i].bufferWidth, haloSpecs[i].bufferHeight, };

        try {
            cl_recvBufferCpuWithGpu[i] = cl::Buffer(cl_context, CL_MEM_READ_WRITE, bufsize*sizeof(double));
            cl_remoteHaloSpecsGpuWithCpu[i] = cl::Buffer(cl_context, &tmpHaloSpecs[0], &tmpHaloSpecs[6], true);
        } catch(cl::Error error) {
            std::cerr << "Tausch2D :: setRemoteHaloInfo() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

    }

}

template <class buf_t> void Tausch2D<buf_t>::setRemoteHaloInfoGpuWithGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsGpuWithGpu = numHaloParts;
    remoteHaloSpecsGpuWithGpu = new TauschHaloSpec[numHaloParts];
    mpiRecvBufferGpuWithGpu = new buf_t*[numHaloParts];
    mpiRecvRequestsGpuWithGpu = new MPI_Request[numHaloParts];
    setupMpiRecvGpuWithGpu = new bool[numHaloParts];

    try {
        cl_recvBufferGpuWithGpu = new cl::Buffer[remoteHaloNumPartsGpuWithGpu];
        cl_remoteHaloSpecsGpuWithGpu = new cl::Buffer[numHaloParts];
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: setRemoteHaloInfoGpuWithGpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsGpuWithGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsGpuWithGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecsGpuWithGpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsGpuWithGpu[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecsGpuWithGpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsGpuWithGpu[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecsGpuWithGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        mpiRecvBufferGpuWithGpu[i] = new buf_t[bufsize]{};

        setupMpiRecvGpuWithGpu[i] = false;

        size_t tmpHaloSpecs[6] = {haloSpecs[i].haloX, haloSpecs[i].haloY, haloSpecs[i].haloWidth, haloSpecs[i].haloHeight,
                                  haloSpecs[i].bufferWidth, haloSpecs[i].bufferHeight, };

        try {
            cl_recvBufferGpuWithGpu[i] = cl::Buffer(cl_context, CL_MEM_READ_WRITE, bufsize*sizeof(double));
            cl_remoteHaloSpecsGpuWithGpu[i] = cl::Buffer(cl_context, &tmpHaloSpecs[0], &tmpHaloSpecs[6], true);
        } catch(cl::Error error) {
            std::cerr << "Tausch2D :: setRemoteHaloInfoGpuWithGpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

    }

}
#endif


////////////////////////
/// Post Receives

template <class buf_t> void Tausch2D<buf_t>::postReceiveCpu(size_t haloId, int mpitag) {

    if(!setupMpiRecvCpuWithCpu[haloId]) {

        if(mpitag == -1) {
            std::cerr << "[Tausch2D] ERROR: MPI_Recv for halo region #" << haloId << " hasn't been posted before, missing mpitag... Abort!"
                      << std::endl;
            exit(1);
        }

        setupMpiRecvCpuWithCpu[haloId] = true;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsCpuWithCpu[haloId].haloWidth*remoteHaloSpecsCpuWithCpu[haloId].haloHeight;

        MPI_Recv_init(&mpiRecvBufferCpuWithCpu[haloId][0], bufsize, mpiDataType,
                      remoteHaloSpecsCpuWithCpu[haloId].remoteMpiRank, mpitag, TAUSCH_COMM, &mpiRecvRequestsCpuWithCpu[haloId]);

    }

    MPI_Start(&mpiRecvRequestsCpuWithCpu[haloId]);

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::postReceiveCpuForGpu(size_t haloId, int msgtag) {
    msgtagsCpuToGpu[haloId].store(msgtag);
}

template <class buf_t> void Tausch2D<buf_t>::postReceiveGpu(size_t haloId, int msgtag) {
    msgtagsGpuToCpu[haloId].store(msgtag);
}

template <class buf_t> void Tausch2D<buf_t>::postReceiveGpuWithGpu(size_t haloId, int msgtag) {

    if(!setupMpiRecvGpuWithGpu[haloId]) {

        if(msgtag == -1) {
            std::cerr << "[Tausch2D] ERROR: MPI_Recv for halo region #" << haloId << " hasn't been posted before, missing mpitag... Abort!"
                      << std::endl;
            exit(1);
        }

        setupMpiRecvGpuWithGpu[haloId] = true;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsGpuWithGpu[haloId].haloWidth*remoteHaloSpecsGpuWithGpu[haloId].haloHeight;

        MPI_Recv_init(&mpiRecvBufferGpuWithGpu[haloId][0], bufsize, mpiDataType,
                      remoteHaloSpecsGpuWithGpu[haloId].remoteMpiRank, msgtag, TAUSCH_COMM, &mpiRecvRequestsGpuWithGpu[haloId]);

    }

    MPI_Start(&mpiRecvRequestsGpuWithGpu[haloId]);
}
#endif


////////////////////////
/// Post ALL Receives

template <class buf_t> void Tausch2D<buf_t>::postAllReceivesCpu(int *mpitag) {

    if(mpitag == nullptr) {
        mpitag = new int[remoteHaloNumPartsCpuWithCpu];
        for(int id = 0; id < remoteHaloNumPartsCpuWithCpu; ++id)
            mpitag[id] = -1;
    }

    for(int id = 0; id < remoteHaloNumPartsCpuWithCpu; ++id)
        postReceiveCpu(id, mpitag[id]);

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::postAllReceivesCpuForGpu(int *msgtag) {

    if(msgtag == nullptr) {
        std::cerr << "Tausch2D::postAllReceives :: ERROR :: msgtag cannot be nullptr for device " << TAUSCH_CPU+TAUSCH_WITHGPU << std::endl;
        return;
    }

    for(int id = 0; id < remoteHaloNumPartsCpuWithGpu; ++id)
        postReceiveCpuForGpu(id, msgtag[id]);

}

template <class buf_t> void Tausch2D<buf_t>::postAllReceivesGpu(int *msgtag) {

    if(msgtag == nullptr) {
        std::cerr << "Tausch2D::postAllReceives :: ERROR :: msgtag cannot be nullptr for device " << TAUSCH_GPU << std::endl;
        return;
    }

    for(int id = 0; id < remoteHaloNumPartsGpuWithCpu; ++id)
        postReceiveGpu(id, msgtag[id]);

}

template <class buf_t> void Tausch2D<buf_t>::postAllReceivesGpuWithGpu(int *msgtag) {

    if(msgtag == nullptr) {
        msgtag = new int[remoteHaloNumPartsCpuWithCpu];
        for(int id = 0; id < remoteHaloNumPartsCpuWithCpu; ++id)
            msgtag[id] = -1;
    }

    for(int id = 0; id < remoteHaloNumPartsGpuWithGpu; ++id)
        postReceiveGpuWithGpu(id, msgtag[id]);

}
#endif


////////////////////////
/// Pack send buffer

template <class buf_t> void Tausch2D<buf_t>::packSendBufferCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    for(int s = 0; s < region.width*region.height; ++s) {
        int bufIndex = (region.y + s/region.width + localHaloSpecsCpuWithCpu[haloId].haloY)*localHaloSpecsCpuWithCpu[haloId].bufferWidth +
                    s%region.width + localHaloSpecsCpuWithCpu[haloId].haloX + region.x;
        int mpiIndex = (s/region.width + region.y)*localHaloSpecsCpuWithCpu[haloId].haloWidth + s%region.width + region.x;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b] * localHaloSpecsCpuWithCpu[haloId].haloWidth * localHaloSpecsCpuWithCpu[haloId].haloHeight;
            mpiSendBufferCpuWithCpu[haloId][offset + valuesPerPointPerBuffer[bufferId]*mpiIndex + val] =
                    buf[valuesPerPointPerBuffer[bufferId]*bufIndex + val];
        }
    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::packSendBufferCpuToGpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    for(int s = 0; s < region.width * region.height; ++s) {
        int bufIndex = (s/region.width + localHaloSpecsCpuWithGpu[haloId].haloY + region.y)*localHaloSpecsCpuWithGpu[haloId].bufferWidth+
                    s%region.width +localHaloSpecsCpuWithGpu[haloId].haloX + region.x;
        int mpiIndex = (s/region.width + region.y)*localHaloSpecsCpuWithGpu[haloId].haloWidth + s%region.width + region.x;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b] * localHaloSpecsCpuWithGpu[haloId].haloWidth * localHaloSpecsCpuWithGpu[haloId].haloHeight;
            sendBufferCpuWithGpu[haloId][offset + valuesPerPointPerBuffer[bufferId]*mpiIndex + val]
                    .store(buf[valuesPerPointPerBuffer[bufferId]*bufIndex + val]);
        }
    }

}

template <class buf_t> void Tausch2D<buf_t>::packSendBufferGpuToCpu(size_t haloId, size_t bufferId, cl::Buffer buf) {

    try {

        auto kernel_packNextSendBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                    (cl_programs, "packSendBuffer");

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsGpuWithCpu[haloId].haloWidth*localHaloSpecsGpuWithCpu[haloId].haloHeight;

        int globalsize = (bufsize/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        cl::Buffer cl_bufferId(cl_context, &bufferId, (&bufferId)+1, true);

        kernel_packNextSendBuffer(cl::EnqueueArgs(cl_queue, cl::NDRange(globalsize), cl::NDRange(cl_kernelLocalSize)),
                                  cl_localHaloSpecsGpuWithCpu[haloId], cl_valuesPerPointPerBuffer,
                                  cl_bufferId, cl_sendBufferGpuWithCpu[haloId], buf);

    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch2D<buf_t>::packSendBufferGpuWithGpu(size_t haloId, size_t bufferId, cl::Buffer buf) {

    try {

        auto kernel_packSendBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                    (cl_programs, "packSendBuffer");

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsGpuWithGpu[haloId].haloWidth*localHaloSpecsGpuWithGpu[haloId].haloHeight;

        int globalsize = (bufsize/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        cl::Buffer cl_bufferId(cl_context, &bufferId, (&bufferId)+1, true);

        kernel_packSendBuffer(cl::EnqueueArgs(cl_queue, cl::NDRange(globalsize), cl::NDRange(cl_kernelLocalSize)),
                                  cl_localHaloSpecsGpuWithGpu[haloId], cl_valuesPerPointPerBuffer,
                                  cl_bufferId, cl_sendBufferGpuWithGpu[haloId], buf);

    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: packSendBufferGpuWithGpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}
#endif


////////////////////////
/// Send data off

template <class buf_t> void Tausch2D<buf_t>::sendCpu(size_t haloId, int mpitag) {

    if(!setupMpiSendCpuWithCpu[haloId]) {

        if(mpitag == -1) {
            std::cerr << "[Tausch2D] ERROR: MPI_Send for halo region #" << haloId << " hasn't been posted before, missing mpitag... Abort!"
                      << std::endl;
            exit(1);
        }

        setupMpiSendCpuWithCpu[haloId] = true;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsCpuWithCpu[haloId].haloWidth*localHaloSpecsCpuWithCpu[haloId].haloHeight;

        MPI_Send_init(&mpiSendBufferCpuWithCpu[haloId][0], bufsize, mpiDataType, localHaloSpecsCpuWithCpu[haloId].remoteMpiRank,
                  mpitag, TAUSCH_COMM, &mpiSendRequestsCpuWithCpu[haloId]);

    }

    MPI_Start(&mpiSendRequestsCpuWithCpu[haloId]);

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::sendCpuToGpu(size_t haloId, int msgtag) {
    msgtagsCpuToGpu[haloId].store(msgtag);
    syncTwoThreads();
}

template <class buf_t> void Tausch2D<buf_t>::sendGpuToCpu(size_t haloId, int msgtag) {

    msgtagsGpuToCpu[haloId].store(msgtag);

    try {

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsGpuWithCpu[haloId].haloWidth*localHaloSpecsGpuWithCpu[haloId].haloHeight;

        buf_t *tmp = new buf_t[bufsize];
        cl::copy(cl_queue, cl_sendBufferGpuWithCpu[haloId], &tmp[0], &tmp[bufsize]);
        for(int i = 0; i < bufsize; ++i)
            sendBufferGpuWithCpu[haloId][i].store(tmp[i]);

    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: sendGpuToCpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    syncTwoThreads();

}

template <class buf_t> void Tausch2D<buf_t>::sendGpuWithGpu(size_t haloId, int msgtag) {

    size_t bufsize = 0;
    for(int n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsGpuWithGpu[haloId].haloWidth*localHaloSpecsGpuWithGpu[haloId].haloHeight;

    if(!setupMpiSendGpuWithGpu[haloId]) {

        if(msgtag == -1) {
            std::cerr << "[Tausch2D] ERROR: MPI_Send for halo region #" << haloId << " hasn't been posted before, missing msgtag... Abort!"
                      << std::endl;
            exit(1);
        }

        setupMpiSendGpuWithGpu[haloId] = true;

        MPI_Send_init(&mpiSendBufferGpuWithGpu[haloId][0], bufsize, mpiDataType, localHaloSpecsGpuWithGpu[haloId].remoteMpiRank,
                  msgtag, TAUSCH_COMM, &mpiSendRequestsGpuWithGpu[haloId]);

    }

    try {
        cl::copy(cl_queue, cl_sendBufferGpuWithGpu[haloId], &mpiSendBufferGpuWithGpu[haloId][0], &mpiSendBufferGpuWithGpu[haloId][bufsize]);
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: sendGpuWithGpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    MPI_Start(&mpiSendRequestsGpuWithGpu[haloId]);

}
#endif


////////////////////////
/// Receive data

template <class buf_t> void Tausch2D<buf_t>::recvCpu(size_t haloId) {
    MPI_Wait(&mpiRecvRequestsCpuWithCpu[haloId], MPI_STATUS_IGNORE);
}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::recvCpuToGpu(size_t haloId) {

    syncTwoThreads();

    int remoteid = obtainRemoteId(msgtagsCpuToGpu[haloId]);

    size_t bufsize = 0;
    for(int n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsGpuWithCpu[haloId].haloWidth*remoteHaloSpecsGpuWithCpu[haloId].haloHeight;

    for(int j = 0; j < bufsize; ++j)
        recvBufferGpuWithCpu[haloId][j] = sendBufferCpuWithGpu[remoteid][j].load();

    try {
        cl_recvBufferCpuWithGpu[haloId] = cl::Buffer(cl_context, &recvBufferGpuWithCpu[haloId][0], &recvBufferGpuWithCpu[haloId][bufsize], false);
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: recvCpuToGpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch2D<buf_t>::recvGpuToCpu(size_t haloId) {

    syncTwoThreads();

    int remoteid = obtainRemoteId(msgtagsGpuToCpu[haloId]);

    size_t bufsize = 0;
    for(int n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsCpuWithGpu[haloId].haloWidth*remoteHaloSpecsCpuWithGpu[haloId].haloHeight;
    for(int i = 0; i < bufsize; ++i)
        recvBufferCpuWithGpu[haloId][i] = sendBufferGpuWithCpu[remoteid][i].load();

}

template <class buf_t> void Tausch2D<buf_t>::recvGpuWithGpu(size_t haloId) {

    MPI_Wait(&mpiRecvRequestsGpuWithGpu[haloId], MPI_STATUS_IGNORE);

    size_t bufsize = 0;
    for(int n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsGpuWithGpu[haloId].haloWidth*remoteHaloSpecsGpuWithGpu[haloId].haloHeight;

    try {
        cl_recvBufferGpuWithGpu[haloId] = cl::Buffer(cl_context, &mpiRecvBufferGpuWithGpu[haloId][0], &mpiRecvBufferGpuWithGpu[haloId][bufsize], false);
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: recvGpuWithGpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}
#endif


////////////////////////
/// Unpack received data

template <class buf_t> void Tausch2D<buf_t>::unpackRecvBufferCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    for(int s = 0; s < region.width * region.height; ++s) {
        int bufIndex = (region.y + s/region.width + remoteHaloSpecsCpuWithCpu[haloId].haloY)*remoteHaloSpecsCpuWithCpu[haloId].bufferWidth +
                    s%region.width + remoteHaloSpecsCpuWithCpu[haloId].haloX + region.x;
        int mpiIndex = (s/region.width + region.y)*remoteHaloSpecsCpuWithCpu[haloId].haloWidth + s%region.width + region.x;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b] * remoteHaloSpecsCpuWithCpu[haloId].haloWidth * remoteHaloSpecsCpuWithCpu[haloId].haloHeight;
            buf[valuesPerPointPerBuffer[bufferId]*bufIndex + val] =
                    mpiRecvBufferCpuWithCpu[haloId][offset + valuesPerPointPerBuffer[bufferId]*mpiIndex + val];
        }
    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::unpackRecvBufferGpuToCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    for(int s = 0; s < region.width * region.height; ++s) {
        int bufIndex = (s/region.width + remoteHaloSpecsCpuWithGpu[haloId].haloY + region.y)*remoteHaloSpecsCpuWithGpu[haloId].bufferWidth +
                    s%region.width +remoteHaloSpecsCpuWithGpu[haloId].haloX + region.x;
        int mpiIndex = (s/region.width + region.y)*remoteHaloSpecsCpuWithGpu[haloId].haloWidth + s%region.width + region.x;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b] * remoteHaloSpecsCpuWithGpu[haloId].haloWidth * remoteHaloSpecsCpuWithGpu[haloId].haloHeight;
            buf[valuesPerPointPerBuffer[bufferId]*bufIndex + val]
                    = recvBufferCpuWithGpu[haloId][offset + valuesPerPointPerBuffer[bufferId]*mpiIndex + val];
        }
    }

}

template <class buf_t> void Tausch2D<buf_t>::unpackRecvBufferCpuToGpu(size_t haloId, size_t bufferId, cl::Buffer buf) {

    try {
        auto kernel_unpackRecvBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                (cl_programs, "unpackRecvBuffer");

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsGpuWithCpu[haloId].haloWidth*remoteHaloSpecsGpuWithCpu[haloId].haloHeight;

        int globalsize = (bufsize/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        cl::Buffer cl_bufferId(cl_context, &bufferId, (&bufferId)+1, true);

        kernel_unpackRecvBuffer(cl::EnqueueArgs(cl_queue, cl::NDRange(globalsize), cl::NDRange(cl_kernelLocalSize)),
                                cl_remoteHaloSpecsGpuWithCpu[haloId], cl_valuesPerPointPerBuffer, cl_bufferId,
                                cl_recvBufferCpuWithGpu[haloId], buf);

    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: unpackRecvBufferCpuToGpu() :: OpenCL exception caught: " << error.what()
                  << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch2D<buf_t>::unpackRecvBufferGpuWithGpu(size_t haloId, size_t bufferId, cl::Buffer buf) {

    try {
        auto kernel_unpackRecvBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                (cl_programs, "unpackRecvBuffer");

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsGpuWithGpu[haloId].haloWidth*remoteHaloSpecsGpuWithGpu[haloId].haloHeight;

        int globalsize = (bufsize/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        cl::Buffer cl_bufferId(cl_context, &bufferId, (&bufferId)+1, true);

        kernel_unpackRecvBuffer(cl::EnqueueArgs(cl_queue, cl::NDRange(globalsize), cl::NDRange(cl_kernelLocalSize)),
                                cl_remoteHaloSpecsGpuWithGpu[haloId], cl_valuesPerPointPerBuffer, cl_bufferId,
                                cl_recvBufferGpuWithGpu[haloId], buf);

    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: unpackRecvBufferGpuWithGpu() :: OpenCL exception caught: " << error.what()
                  << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}
#endif


////////////////////////
/// Pack buffer and send data off

template <class buf_t> void Tausch2D<buf_t>::packAndSendCpu(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag) {
    packSendBufferCpu(haloId, 0, buf, region);
    sendCpu(haloId, msgtag);
}
#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::packAndSendCpuForGpu(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag) {
    packSendBufferCpuToGpu(haloId, 0, buf, region);
    sendCpuToGpu(haloId, msgtag);
}
template <class buf_t> void Tausch2D<buf_t>::packAndSendGpu(size_t haloId, cl::Buffer buf, int msgtag) {
    packSendBufferGpuToCpu(haloId, 0, buf);
    sendGpuToCpu(haloId, msgtag);
}
template <class buf_t> void Tausch2D<buf_t>::packAndSendGpuWithGpu(size_t haloId, cl::Buffer buf, int msgtag) {
    packSendBufferGpuWithGpu(haloId, 0, buf);
    sendGpuWithGpu(haloId, msgtag);
}
#endif


////////////////////////
/// Receive data and unpack

template <class buf_t> void Tausch2D<buf_t>::recvAndUnpackCpu(size_t haloId, buf_t *buf, TauschPackRegion region) {
    recvCpu(haloId);
    unpackRecvBufferCpu(haloId, 0, buf, region);
}
#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::recvAndUnpackCpuForGpu(size_t haloId, buf_t *buf, TauschPackRegion region) {
    recvCpuToGpu(haloId);
    unpackRecvBufferGpuToCpu(haloId, 0, buf, region);
}
template <class buf_t> void Tausch2D<buf_t>::recvAndUnpackGpu(size_t haloId, cl::Buffer buf) {
    recvCpu(haloId);
    unpackRecvBufferCpuToGpu(haloId, 0, buf);
}
template <class buf_t> void Tausch2D<buf_t>::recvAndUnpackGpuWithGpu(size_t haloId, cl::Buffer buf) {
    recvGpuWithGpu(haloId);
    unpackRecvBufferGpuWithGpu(haloId, 0, buf);
}
#endif






///////////////////////////////////////////////////////////////
/// SOME GENERAL PURPOSE OPENCL FUNCTIONS - PART OF PUBLIC API
///////////////////////////////////////////////////////////////

#ifdef TAUSCH_OPENCL

template <class buf_t> void Tausch2D<buf_t>::enableOpenCL(bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName,
                                                          bool showOpenCLBuildLog) {

    this->blockingSyncCpuGpu = blockingSyncCpuGpu;
    cl_kernelLocalSize = clLocalWorkgroupSize;
    this->showOpenCLBuildLog = showOpenCLBuildLog;

    sync_counter[0].store(0); sync_counter[1].store(0);
    sync_lock[0].store(0); sync_lock[1].store(0);

    setupOpenCL(giveOpenCLDeviceName);

    try {
        cl_valuesPerPointPerBuffer = cl::Buffer(cl_context, &valuesPerPointPerBuffer[0], &valuesPerPointPerBuffer[numBuffers], true);
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: enableOpenCL() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch2D<buf_t>::enableOpenCL(cl::Device cl_defaultDevice, cl::Context cl_context, cl::CommandQueue cl_queue,
                                                          bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool showOpenCLBuildLog) {

    this->blockingSyncCpuGpu = blockingSyncCpuGpu;
    this->cl_kernelLocalSize = clLocalWorkgroupSize;
    this->showOpenCLBuildLog = showOpenCLBuildLog;

    sync_counter[0].store(0); sync_counter[1].store(0);
    sync_lock[0].store(0); sync_lock[1].store(0);

    this->cl_defaultDevice = cl_defaultDevice;
    this->cl_context = cl_context;
    this->cl_queue = cl_queue;

    compileKernels();

    try {
        cl_valuesPerPointPerBuffer = cl::Buffer(cl_context, &valuesPerPointPerBuffer[0], &valuesPerPointPerBuffer[numBuffers], true);
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: enableOpenCL() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch2D<buf_t>::syncTwoThreads() {

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

template <class buf_t> int Tausch2D<buf_t>::obtainRemoteId(int msgtag) {
    for(int j = 0; j < remoteHaloNumPartsCpuWithGpu; ++j) {
        if(msgtagsCpuToGpu[j].load() == msgtag)
            return j;
    }
    return 0;
}

template <class buf_t> void Tausch2D<buf_t>::setupOpenCL(bool giveOpenCLDeviceName) {

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

template <class buf_t> void Tausch2D<buf_t>::compileKernels() {

  std::string oclstr = R"d(
kernel void packSendBuffer(global const size_t * restrict const haloSpecs,
                           global const size_t * restrict const valuesPerPointPerBuffer, global int * restrict const bufferId,
                           global double * restrict const haloBuffer, global const double * restrict const buffer) {

    const int current = get_global_id(0);

    int maxSize = haloSpecs[2]*haloSpecs[3];

    if(current >= maxSize) return;

    int index = (current/haloSpecs[2] + haloSpecs[1])*haloSpecs[4] +
                 current%haloSpecs[2] + haloSpecs[0];

    for(int val = 0; val < valuesPerPointPerBuffer[*bufferId]; ++val) {
        int offset = 0;
        for(int b = 0; b < *bufferId; ++b)
            offset += valuesPerPointPerBuffer[b]*maxSize;
        haloBuffer[offset+ valuesPerPointPerBuffer[*bufferId]*current + val] = buffer[valuesPerPointPerBuffer[*bufferId]*index + val];
    }

}

kernel void unpackRecvBuffer(global const size_t * restrict const haloSpecs,
                             global const size_t * restrict const valuesPerPointPerBuffer, global int * restrict const bufferId,
                             global const double * restrict const haloBuffer, global double * restrict const buffer) {

    const int current = get_global_id(0);

    int maxSize = haloSpecs[2]*haloSpecs[3];

    if(current >= maxSize) return;

    int index = (current/haloSpecs[2] + haloSpecs[1])*haloSpecs[4] +
                 current%haloSpecs[2] + haloSpecs[0];

    for(int val = 0; val < valuesPerPointPerBuffer[*bufferId]; ++val) {
        int offset = 0;
        for(int b = 0; b < *bufferId; ++b)
            offset += valuesPerPointPerBuffer[b]*maxSize;
        buffer[valuesPerPointPerBuffer[*bufferId]*index + val] =
                haloBuffer[offset + valuesPerPointPerBuffer[*bufferId]*current + val];
    }

}
      )d";

    try {
        cl_programs = cl::Program(cl_context, oclstr, false);
        cl_programs.build("");
        if(showOpenCLBuildLog) {
            try {
                std::string log = cl_programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_defaultDevice);
                std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl
                          << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** "
                          << std::endl << std::endl;
            } catch(cl::Error err) {
                std::cout << "Tausch2D :: compileKernels() :: getBuildInfo :: OpenCL exception caught: " << err.what() << " (" << err.err() << ")"
                          << std::endl;
            }
        }
    } catch(cl::Error error) {
        std::cout << "Tausch2D :: compileKernels() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        if(error.err() == -11) {
            try {
                std::string log = cl_programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_defaultDevice);
                std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl
                          << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** "
                          << std::endl << std::endl;
            } catch(cl::Error err) {
                std::cout << "Tausch2D :: compileKernels() :: getBuildInfo :: OpenCL exception caught: " << err.what() << " (" << err.err() << ")"
                          << std::endl;
            }
        }
    }

}


#endif

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
