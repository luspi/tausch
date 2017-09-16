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
    delete[] localHaloSpecsCpu;
    delete[] mpiSendBuffer;
    delete[] remoteHaloSpecsCpu;
    delete[] mpiRecvBuffer;

    delete[] mpiSendRequests;
    delete[] mpiRecvRequests;

    delete[] setupMpiRecv;
    delete[] setupMpiSend;

    delete[] valuesPerPointPerBuffer;
}

/////////////////////////////////////////////
/// PUBLIC API FUNCTION
/////////////////////////////////////////////

template <class buf_t> void Tausch3D<buf_t>::setLocalHaloInfo(TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return setLocalHaloInfoCpu(numHaloParts, haloSpecs);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return setLocalHaloInfoCpuForGpu(numHaloParts, haloSpecs);
    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return setLocalHaloInfoGpu(numHaloParts, haloSpecs);
#endif

    std::cerr << "Tausch3D::setLocalHaloInfo :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch3D<buf_t>::setRemoteHaloInfo(TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return setRemoteHaloInfoCpu(numHaloParts, haloSpecs);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return setRemoteHaloInfoCpuForGpu(numHaloParts, haloSpecs);
    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return setRemoteHaloInfoGpu(numHaloParts, haloSpecs);
#endif

    std::cerr << "Tausch3D::setRemoteHaloInfo :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch3D<buf_t>::postReceive(TauschDeviceDirection flags, size_t haloId, int msgtag) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return postReceiveCpu(haloId, msgtag);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return postReceiveCpuForGpu(haloId, msgtag);
    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return postReceiveGpu(haloId, msgtag);
#endif

    std::cerr << "Tausch3D::postReceive :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch3D<buf_t>::postAllReceives(TauschDeviceDirection flags, int *msgtag) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return postAllReceivesCpu(msgtag);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return postAllReceivesCpuForGpu(msgtag);
    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return postAllReceivesGpu(msgtag);
#endif

    std::cerr << "Tausch3D::postAllReceives :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch3D<buf_t>::packSendBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return packSendBufferCpu(haloId, bufferId, buf, region);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return packSendBufferCpuToGpu(haloId, bufferId, buf, region);
#endif

    std::cerr << "Tausch3D::postReceive(buf_t) :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch3D<buf_t>::packSendBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.z = 0;
        region.width = localHaloSpecsCpu[haloId].haloWidth;
        region.height = localHaloSpecsCpu[haloId].haloHeight;
        region.depth = localHaloSpecsCpu[haloId].haloDepth;
        return packSendBufferCpu(haloId, bufferId, buf, region);
    }
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.z = 0;
        region.width = localHaloSpecsCpuForGpu[haloId].haloWidth;
        region.height = localHaloSpecsCpuForGpu[haloId].haloHeight;
        region.depth = localHaloSpecsCpuForGpu[haloId].haloDepth;
        return packSendBufferCpuToGpu(haloId, bufferId, buf, region);
    }
#endif

    std::cerr << "Tausch3D::postReceive(buf_t) :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::packSendBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, cl::Buffer buf) {

    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return packSendBufferGpuToCpu(haloId, bufferId, buf);

    std::cerr << "Tausch3D::postReceive(cl::Buffer) :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}
#endif

template <class buf_t> void Tausch3D<buf_t>::send(TauschDeviceDirection flags, size_t haloId, int msgtag) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return sendCpu(haloId, msgtag);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return sendCpuToGpu(haloId, msgtag);
    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return sendGpuToCpu(haloId, msgtag);
#endif

    std::cerr << "Tausch3D::send :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch3D<buf_t>::recv(TauschDeviceDirection flags, size_t haloId) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return recvCpu(haloId);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return recvGpuToCpu(haloId);
    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return recvCpuToGpu(haloId);
#endif

    std::cerr << "Tausch3D::recv :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch3D<buf_t>::unpackRecvBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return unpackRecvBufferCpu(haloId, bufferId, buf, region);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return unpackRecvBufferGpuToCpu(haloId, bufferId, buf, region);
#endif

    std::cerr << "Tausch3D::recv(buf_t) :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}

template <class buf_t> void Tausch3D<buf_t>::unpackRecvBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, buf_t *buf) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.z = 0;
        region.width = remoteHaloSpecsCpu[haloId].haloWidth;
        region.height = remoteHaloSpecsCpu[haloId].haloHeight;
        region.depth = remoteHaloSpecsCpu[haloId].haloDepth;
        return unpackRecvBufferCpu(haloId, bufferId, buf, region);
    }
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.z = 0;
        region.width = remoteHaloSpecsCpuForGpu[haloId].haloWidth;
        region.height = remoteHaloSpecsCpuForGpu[haloId].haloHeight;
        region.depth = remoteHaloSpecsCpuForGpu[haloId].haloDepth;
        return unpackRecvBufferGpuToCpu(haloId, bufferId, buf, region);
    }
#endif

    std::cerr << "Tausch3D::recv(buf_t) :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}
#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::unpackRecvBuffer(TauschDeviceDirection flags, size_t haloId, size_t bufferId, cl::Buffer buf) {

    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return unpackRecvBufferCpuToGpu(haloId, bufferId, buf);

    std::cerr << "Tausch3D::recv(cl::Buffer) :: ERROR :: Unknown device specification provided: " << flags << std::endl;

}
#endif

template <class buf_t> void Tausch3D<buf_t>::packAndSend(TauschDeviceDirection flags, size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return packAndSendCpu(haloId, buf, region, msgtag);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return packAndSendCpuForGpu(haloId, buf, region, msgtag);
#endif

}

template <class buf_t> void Tausch3D<buf_t>::packAndSend(TauschDeviceDirection flags, size_t haloId, buf_t *buf, int msgtag) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.width = localHaloSpecsCpu[haloId].haloWidth;
        region.height = localHaloSpecsCpu[haloId].haloHeight;
        return packAndSendCpu(haloId, buf, region, msgtag);
    }
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.width = localHaloSpecsCpuForGpu[haloId].haloWidth;
        region.height = localHaloSpecsCpuForGpu[haloId].haloHeight;
        return packAndSendCpuForGpu(haloId, buf, region, msgtag);
    }
#endif

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::packAndSend(TauschDeviceDirection flags, size_t haloId, cl::Buffer buf, int msgtag) {

    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return packAndSendGpu(haloId, buf, msgtag);

}
#endif

template <class buf_t> void Tausch3D<buf_t>::recvAndUnpack(TauschDeviceDirection flags, size_t haloId, buf_t *buf, TauschPackRegion region) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU)
        return recvAndUnpackCpu(haloId, buf, region);
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU)
        return recvAndUnpackCpuForGpu(haloId, buf, region);
#endif

}

template <class buf_t> void Tausch3D<buf_t>::recvAndUnpack(TauschDeviceDirection flags, size_t haloId, buf_t *buf) {

    if(flags == TAUSCH_CPU+TAUSCH_WITHCPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.width = remoteHaloSpecsCpu[haloId].haloWidth;
        region.height = remoteHaloSpecsCpu[haloId].haloHeight;
        return recvAndUnpackCpu(haloId, buf, region);
    }
#ifdef TAUSCH_OPENCL
    if(flags == TAUSCH_CPU+TAUSCH_WITHGPU) {
        TauschPackRegion region;
        region.x = 0;
        region.y = 0;
        region.width = remoteHaloSpecsCpuForGpu[haloId].haloWidth;
        region.height = remoteHaloSpecsCpuForGpu[haloId].haloHeight;
        return recvAndUnpackCpuForGpu(haloId, buf, region);
    }
#endif

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::recvAndUnpack(TauschDeviceDirection flags, size_t haloId, cl::Buffer buf) {

    if(flags == TAUSCH_GPU+TAUSCH_WITHCPU)
        return recvAndUnpackGpu(haloId, buf);

}
#endif



//////////////////////////////////////
/// PRIVATE MEMBER FUNCTIONS
//////////////////////////////////////



////////////////////////
/// Set local halo info

template <class buf_t> void Tausch3D<buf_t>::setLocalHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumParts = numHaloParts;
    localHaloSpecsCpu = new TauschHaloSpec[numHaloParts];
    mpiSendBuffer = new buf_t*[numHaloParts];
    mpiSendRequests = new MPI_Request[numHaloParts];
    setupMpiSend = new bool[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecsCpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsCpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecsCpu[i].bufferDepth = haloSpecs[i].bufferDepth;
        localHaloSpecsCpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsCpu[i].haloY = haloSpecs[i].haloY;
        localHaloSpecsCpu[i].haloZ = haloSpecs[i].haloZ;
        localHaloSpecsCpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsCpu[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecsCpu[i].haloDepth = haloSpecs[i].haloDepth;
        localHaloSpecsCpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight*haloSpecs[i].haloDepth;

        mpiSendBuffer[i] = new buf_t[bufsize]{};

        setupMpiSend[i] = false;

    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::setLocalHaloInfoCpuForGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumPartsCpuForGpu = numHaloParts;
    localHaloSpecsCpuForGpu = new TauschHaloSpec[numHaloParts];
    cpuToGpuSendBuffer = new std::atomic<buf_t>*[numHaloParts];
    msgtagsCpuToGpu = new std::atomic<int>[numHaloParts]{};

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecsCpuForGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsCpuForGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecsCpuForGpu[i].bufferDepth = haloSpecs[i].bufferDepth;
        localHaloSpecsCpuForGpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsCpuForGpu[i].haloY = haloSpecs[i].haloY;
        localHaloSpecsCpuForGpu[i].haloZ = haloSpecs[i].haloZ;
        localHaloSpecsCpuForGpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsCpuForGpu[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecsCpuForGpu[i].haloDepth = haloSpecs[i].haloDepth;
        localHaloSpecsCpuForGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight*haloSpecs[i].haloDepth;
        cpuToGpuSendBuffer[i] = new std::atomic<buf_t>[bufsize]{};

    }

}

template <class buf_t> void Tausch3D<buf_t>::setLocalHaloInfoGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumPartsGpu = numHaloParts;
    localHaloSpecsGpu = new TauschHaloSpec[numHaloParts];
    gpuToCpuSendBuffer = new std::atomic<buf_t>*[numHaloParts];
    msgtagsGpuToCpu = new std::atomic<int>[numHaloParts]{};

    try {
        cl_gpuToCpuSendBuffer = new cl::Buffer[numHaloParts];
        cl_localHaloSpecsGpu = new cl::Buffer[numHaloParts];
    } catch(cl::Error error) {
        std::cerr << "Tausch3D :: setLocalHaloInfoGpu() (1) :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecsGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecsGpu[i].bufferDepth = haloSpecs[i].bufferDepth;
        localHaloSpecsGpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsGpu[i].haloY = haloSpecs[i].haloY;
        localHaloSpecsGpu[i].haloZ = haloSpecs[i].haloZ;
        localHaloSpecsGpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsGpu[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecsGpu[i].haloDepth = haloSpecs[i].haloDepth;
        localHaloSpecsGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight*haloSpecs[i].haloDepth;
        gpuToCpuSendBuffer[i] = new std::atomic<buf_t>[bufsize]{};

        size_t tmpHaloSpecs[9] = {haloSpecs[i].haloX, haloSpecs[i].haloY, haloSpecs[i].haloZ,
                                  haloSpecs[i].haloWidth, haloSpecs[i].haloHeight, haloSpecs[i].haloDepth,
                                  haloSpecs[i].bufferWidth, haloSpecs[i].bufferHeight, haloSpecs[i].bufferDepth};

        try {
            cl_gpuToCpuSendBuffer[i] = cl::Buffer(cl_context, CL_MEM_READ_WRITE, bufsize*sizeof(buf_t));
            cl_localHaloSpecsGpu[i] = cl::Buffer(cl_context, &tmpHaloSpecs[0], &tmpHaloSpecs[9], true);
        } catch(cl::Error error) {
            std::cerr << "Tausch3D :: setLocalHaloInfoGpu() (2) :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

    }

}
#endif


////////////////////////
/// Set remote halo info

template <class buf_t> void Tausch3D<buf_t>::setRemoteHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumParts = numHaloParts;
    remoteHaloSpecsCpu = new TauschHaloSpec[numHaloParts];
    mpiRecvBuffer = new buf_t*[numHaloParts];
    mpiRecvRequests = new MPI_Request[numHaloParts];
    setupMpiRecv = new bool[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsCpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsCpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecsCpu[i].bufferDepth = haloSpecs[i].bufferDepth;
        remoteHaloSpecsCpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsCpu[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecsCpu[i].haloZ = haloSpecs[i].haloZ;
        remoteHaloSpecsCpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsCpu[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecsCpu[i].haloDepth = haloSpecs[i].haloDepth;
        remoteHaloSpecsCpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight*haloSpecs[i].haloDepth;

        mpiRecvBuffer[i] = new buf_t[bufsize]{};

        setupMpiRecv[i] = false;

    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::setRemoteHaloInfoCpuForGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsCpuForGpu = numHaloParts;
    remoteHaloSpecsCpuForGpu = new TauschHaloSpec[numHaloParts];
    gpuToCpuRecvBuffer = new buf_t*[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsCpuForGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsCpuForGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecsCpuForGpu[i].bufferDepth = haloSpecs[i].bufferDepth;
        remoteHaloSpecsCpuForGpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsCpuForGpu[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecsCpuForGpu[i].haloZ = haloSpecs[i].haloZ;
        remoteHaloSpecsCpuForGpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsCpuForGpu[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecsCpuForGpu[i].haloDepth = haloSpecs[i].haloDepth;
        remoteHaloSpecsCpuForGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight*haloSpecs[i].haloDepth;
        gpuToCpuRecvBuffer[i] = new buf_t[bufsize];

    }

}

template <class buf_t> void Tausch3D<buf_t>::setRemoteHaloInfoGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsGpu = numHaloParts;
    remoteHaloSpecsGpu = new TauschHaloSpec[numHaloParts];
    cpuToGpuRecvBuffer = new buf_t*[numHaloParts];

    try {
        cl_cpuToGpuRecvBuffer = new cl::Buffer[remoteHaloNumPartsGpu];
        cl_remoteHaloSpecsGpu = new cl::Buffer[numHaloParts];
    } catch(cl::Error error) {
        std::cerr << "Tausch3D :: enableOpenCL() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecsGpu[i].bufferDepth = haloSpecs[i].bufferDepth;
        remoteHaloSpecsGpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsGpu[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecsGpu[i].haloZ = haloSpecs[i].haloZ;
        remoteHaloSpecsGpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsGpu[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecsGpu[i].haloDepth = haloSpecs[i].haloDepth;
        remoteHaloSpecsGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight*haloSpecs[i].haloDepth;
        cpuToGpuRecvBuffer[i] = new buf_t[bufsize];

        size_t tmpHaloSpecs[9] = {haloSpecs[i].haloX, haloSpecs[i].haloY, haloSpecs[i].haloZ,
                                  haloSpecs[i].haloWidth, haloSpecs[i].haloHeight, haloSpecs[i].haloDepth,
                                  haloSpecs[i].bufferWidth, haloSpecs[i].bufferHeight, haloSpecs[i].bufferDepth};

        try {
            cl_cpuToGpuRecvBuffer[i] = cl::Buffer(cl_context, CL_MEM_READ_WRITE, bufsize*sizeof(double));
            cl_remoteHaloSpecsGpu[i] = cl::Buffer(cl_context, &tmpHaloSpecs[0], &tmpHaloSpecs[9], true);
        } catch(cl::Error error) {
            std::cerr << "Tausch3D :: setRemoteHaloInfo() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

    }

}
#endif


////////////////////////
/// Post Receives

template <class buf_t> void Tausch3D<buf_t>::postReceiveCpu(size_t haloId, int mpitag) {

    if(!setupMpiRecv[haloId]) {

        if(mpitag == -1) {
            std::cerr << "[Tausch3D] ERROR: MPI_Recv for halo region #" << haloId << " hasn't been posted before, missing mpitag... Abort!" << std::endl;
            exit(1);
        }

        setupMpiRecv[haloId] = true;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsCpu[haloId].haloWidth*remoteHaloSpecsCpu[haloId].haloHeight*remoteHaloSpecsCpu[haloId].haloDepth;

        MPI_Recv_init(&mpiRecvBuffer[haloId][0], bufsize,
                      mpiDataType, remoteHaloSpecsCpu[haloId].remoteMpiRank, mpitag, TAUSCH_COMM, &mpiRecvRequests[haloId]);

    }

    MPI_Start(&mpiRecvRequests[haloId]);

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::postReceiveCpuForGpu(size_t haloId, int msgtag) {
    msgtagsCpuToGpu[haloId].store(msgtag);
}

template <class buf_t> void Tausch3D<buf_t>::postReceiveGpu(size_t haloId, int msgtag) {
    msgtagsGpuToCpu[haloId].store(msgtag);
}
#endif


////////////////////////
/// Post ALL Receives

template <class buf_t> void Tausch3D<buf_t>::postAllReceivesCpu(int *mpitag) {

    if(mpitag == nullptr) {
        mpitag = new int[remoteHaloNumParts];
        for(int id = 0; id < remoteHaloNumParts; ++id)
            mpitag[id] = -1;
    }

    for(int id = 0; id < remoteHaloNumParts; ++id)
        postReceiveCpu(id,mpitag[id]);

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::postAllReceivesCpuForGpu(int *msgtag) {

    if(msgtag == nullptr) {
        std::cerr << "Tausch3D::postAllReceives :: ERROR :: msgtag cannot be nullptr for device " << TAUSCH_CPU+TAUSCH_WITHGPU << std::endl;
        return;
    }

    for(int id = 0; id < remoteHaloNumPartsCpuForGpu; ++id)
        postReceiveCpuForGpu(id, msgtag[id]);

}

template <class buf_t> void Tausch3D<buf_t>::postAllReceivesGpu(int *msgtag) {

    if(msgtag == nullptr) {
        std::cerr << "Tausch3D::postAllReceives :: ERROR :: msgtag cannot be nullptr for device " << TAUSCH_GPU << std::endl;
        return;
    }

    for(int id = 0; id < remoteHaloNumPartsGpu; ++id)
        postReceiveGpu(id, msgtag[id]);

}
#endif


////////////////////////
/// Pack send buffer

template <class buf_t> void Tausch3D<buf_t>::packSendBufferCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    for(int s = 0; s < region.width * region.height * region.depth; ++s) {
        int bufIndex = (s/(region.width*region.height) + localHaloSpecsCpu[haloId].haloZ) * localHaloSpecsCpu[haloId].bufferWidth * localHaloSpecsCpu[haloId].bufferHeight +
                    ((s%(region.width*region.height))/localHaloSpecsCpu[haloId].haloWidth + localHaloSpecsCpu[haloId].haloY) * localHaloSpecsCpu[haloId].bufferWidth +
                    s%region.width + localHaloSpecsCpu[haloId].haloX;
        int mpiIndex = (s/(region.width*region.height) + region.z)*(localHaloSpecsCpu[haloId].haloWidth*localHaloSpecsCpu[haloId].haloHeight) +
                       ((s%(region.width*region.height))/localHaloSpecsCpu[haloId].haloWidth + region.y) * localHaloSpecsCpu[haloId].haloWidth +
                       s%region.width + region.x;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b]*localHaloSpecsCpu[haloId].haloWidth*localHaloSpecsCpu[haloId].haloHeight*localHaloSpecsCpu[haloId].haloDepth;
            mpiSendBuffer[haloId][offset + valuesPerPointPerBuffer[bufferId]*mpiIndex + val] =
                    buf[valuesPerPointPerBuffer[bufferId]*bufIndex + val];
        }
    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::packSendBufferCpuToGpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    //HERE
    for(int s = 0; s < region.width*region.height*region.depth; ++s) {
        int bufIndex = (s/(region.width*region.height) + localHaloSpecsCpuForGpu[haloId].haloZ) * localHaloSpecsCpuForGpu[haloId].bufferWidth * localHaloSpecsCpuForGpu[haloId].bufferHeight +
                    ((s%(region.width*region.height))/localHaloSpecsCpuForGpu[haloId].haloWidth + localHaloSpecsCpuForGpu[haloId].haloY) * localHaloSpecsCpuForGpu[haloId].bufferWidth +
                    s%region.width + localHaloSpecsCpuForGpu[haloId].haloX;
        int mpiIndex = (s/(region.width*region.height) + region.z)*(localHaloSpecsCpuForGpu[haloId].haloWidth*localHaloSpecsCpuForGpu[haloId].haloHeight) +
                       ((s%(region.width*region.height))/localHaloSpecsCpuForGpu[haloId].haloWidth + region.y) * localHaloSpecsCpuForGpu[haloId].haloWidth +
                       s%region.width + region.x;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b] * localHaloSpecsCpuForGpu[haloId].haloWidth * localHaloSpecsCpuForGpu[haloId].haloHeight * localHaloSpecsCpuForGpu[haloId].haloDepth;
            cpuToGpuSendBuffer[haloId][offset + valuesPerPointPerBuffer[bufferId]*mpiIndex + val].store(buf[valuesPerPointPerBuffer[bufferId]*bufIndex + val]);
        }
    }

}

template <class buf_t> void Tausch3D<buf_t>::packSendBufferGpuToCpu(size_t haloId, size_t bufferId, cl::Buffer buf) {

    try {

        auto kernel_packNextSendBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                    (cl_programs, "packSendBuffer");

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsGpu[haloId].haloWidth*localHaloSpecsGpu[haloId].haloHeight*localHaloSpecsGpu[haloId].haloDepth;

        int globalsize = (bufsize/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        cl::Buffer cl_bufferId(cl_context, &bufferId, (&bufferId)+1, true);

        kernel_packNextSendBuffer(cl::EnqueueArgs(cl_queue, cl::NDRange(globalsize), cl::NDRange(cl_kernelLocalSize)),
                                  cl_localHaloSpecsGpu[haloId], cl_valuesPerPointPerBuffer,
                                  cl_bufferId, cl_gpuToCpuSendBuffer[haloId], buf);

    } catch(cl::Error error) {
        std::cerr << "Tausch3D :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}
#endif


////////////////////////
/// Send data off

template <class buf_t> void Tausch3D<buf_t>::sendCpu(size_t haloId, int mpitag) {


    if(!setupMpiSend[haloId]) {

        if(mpitag == -1) {
            std::cerr << "[Tausch3D] ERROR: MPI_Send for halo region #" << haloId << " hasn't been posted before, missing mpitag... Abort!" << std::endl;
            exit(1);
        }

        setupMpiSend[haloId] = true;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsCpu[haloId].haloWidth*localHaloSpecsCpu[haloId].haloHeight*localHaloSpecsCpu[haloId].haloDepth;

        MPI_Send_init(&mpiSendBuffer[haloId][0], bufsize,
                  mpiDataType, localHaloSpecsCpu[haloId].remoteMpiRank, mpitag, TAUSCH_COMM, &mpiSendRequests[haloId]);

    } else {
        int flag;
        MPI_Test(&mpiSendRequests[haloId], &flag, MPI_STATUS_IGNORE);
        if(flag == 0) MPI_Wait(&mpiSendRequests[haloId], MPI_STATUS_IGNORE);
    }

    MPI_Start(&mpiSendRequests[haloId]);

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::sendCpuToGpu(size_t haloId, int msgtag) {
    msgtagsCpuToGpu[haloId].store(msgtag);
    syncCpuAndGpu();
}

template <class buf_t> void Tausch3D<buf_t>::sendGpuToCpu(size_t haloId, int msgtag) {

    msgtagsGpuToCpu[haloId].store(msgtag);

    try {

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsGpu[haloId].haloWidth*localHaloSpecsGpu[haloId].haloHeight*localHaloSpecsGpu[haloId].haloDepth;

        buf_t *tmp = new buf_t[bufsize];
        cl::copy(cl_queue, cl_gpuToCpuSendBuffer[haloId], &tmp[0], &tmp[bufsize]);
        for(int i = 0; i < bufsize; ++i)
            gpuToCpuSendBuffer[haloId][i].store(tmp[i]);

    } catch(cl::Error error) {
        std::cerr << "Tausch3D :: sendGpuToCpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    syncCpuAndGpu();

}
#endif


////////////////////////
/// Receive data

template <class buf_t> void Tausch3D<buf_t>::recvCpu(size_t haloId) {
    MPI_Wait(&mpiRecvRequests[haloId], MPI_STATUS_IGNORE);
}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::recvCpuToGpu(size_t haloId) {

    syncCpuAndGpu();

    int remoteid = obtainRemoteId(msgtagsCpuToGpu[haloId]);

    size_t bufsize = 0;
    for(int n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsGpu[haloId].haloWidth*remoteHaloSpecsGpu[haloId].haloHeight*remoteHaloSpecsGpu[haloId].haloDepth;

    for(int j = 0; j < bufsize; ++j)
        cpuToGpuRecvBuffer[haloId][j] = cpuToGpuSendBuffer[remoteid][j].load();

    try {
        cl_cpuToGpuRecvBuffer[haloId] = cl::Buffer(cl_context, &cpuToGpuRecvBuffer[haloId][0], &cpuToGpuRecvBuffer[haloId][bufsize], false);
    } catch(cl::Error error) {
        std::cerr << "Tausch3D :: recvCpuToGpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch3D<buf_t>::recvGpuToCpu(size_t haloId) {

    syncCpuAndGpu();

    int remoteid = obtainRemoteId(msgtagsGpuToCpu[haloId]);

    size_t bufsize = 0;
    for(int n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsCpuForGpu[haloId].haloWidth*remoteHaloSpecsCpuForGpu[haloId].haloHeight*remoteHaloSpecsCpuForGpu[haloId].haloDepth;
    for(int i = 0; i < bufsize; ++i)
        gpuToCpuRecvBuffer[haloId][i] = gpuToCpuSendBuffer[remoteid][i].load();

}
#endif


////////////////////////
/// Unpack received data

template <class buf_t> void Tausch3D<buf_t>::unpackRecvBufferCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    for(int s = 0; s < region.width * region.height * region.depth; ++s) {
        int bufIndex = (s/(region.width*region.height) + remoteHaloSpecsCpu[haloId].haloZ) * remoteHaloSpecsCpu[haloId].bufferWidth * remoteHaloSpecsCpu[haloId].bufferHeight +
                    ((s%(region.width*region.height))/remoteHaloSpecsCpu[haloId].haloWidth + remoteHaloSpecsCpu[haloId].haloY) * remoteHaloSpecsCpu[haloId].bufferWidth +
                    s%region.width + remoteHaloSpecsCpu[haloId].haloX;
        int mpiIndex = (s/(region.width*region.height) + region.z)*(remoteHaloSpecsCpu[haloId].haloWidth*remoteHaloSpecsCpu[haloId].haloHeight) +
                       ((s%(region.width*region.height))/remoteHaloSpecsCpu[haloId].haloWidth + region.y) * remoteHaloSpecsCpu[haloId].haloWidth +
                       s%region.width + region.x;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b]*remoteHaloSpecsCpu[haloId].haloWidth*remoteHaloSpecsCpu[haloId].haloHeight*remoteHaloSpecsCpu[haloId].haloDepth;
            buf[valuesPerPointPerBuffer[bufferId]*bufIndex + val] =
                    mpiRecvBuffer[haloId][offset + valuesPerPointPerBuffer[bufferId]*mpiIndex + val];
        }
    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::unpackRecvBufferGpuToCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    for(int s = 0; s < region.width * region.height * region.depth; ++s) {
        int bufIndex = (s/(region.width*region.height) + remoteHaloSpecsCpuForGpu[haloId].haloZ) * remoteHaloSpecsCpuForGpu[haloId].bufferWidth * remoteHaloSpecsCpuForGpu[haloId].bufferHeight +
                    ((s%(region.width*region.height))/remoteHaloSpecsCpuForGpu[haloId].haloWidth + remoteHaloSpecsCpuForGpu[haloId].haloY) * remoteHaloSpecsCpuForGpu[haloId].bufferWidth +
                    s%region.width + remoteHaloSpecsCpuForGpu[haloId].haloX;
        int mpiIndex = (s/(region.width*region.height) + region.z)*(remoteHaloSpecsCpuForGpu[haloId].haloWidth*remoteHaloSpecsCpuForGpu[haloId].haloHeight) +
                       ((s%(region.width*region.height))/remoteHaloSpecsCpuForGpu[haloId].haloWidth + region.y) * remoteHaloSpecsCpuForGpu[haloId].haloWidth +
                       s%region.width + region.x;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b] * remoteHaloSpecsCpuForGpu[haloId].haloWidth * remoteHaloSpecsCpuForGpu[haloId].haloHeight * remoteHaloSpecsCpuForGpu[haloId].haloDepth;
            buf[valuesPerPointPerBuffer[bufferId]*bufIndex + val] = gpuToCpuRecvBuffer[haloId][offset + valuesPerPointPerBuffer[bufferId]*mpiIndex + val];
        }
    }

}

template <class buf_t> void Tausch3D<buf_t>::unpackRecvBufferCpuToGpu(size_t haloId, size_t bufferId, cl::Buffer buf) {

    try {
        auto kernel_unpackRecvBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                (cl_programs, "unpackRecvBuffer");

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsGpu[haloId].haloWidth*remoteHaloSpecsGpu[haloId].haloHeight*remoteHaloSpecsGpu[haloId].haloDepth;

        int globalsize = (bufsize/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        cl::Buffer cl_bufferId(cl_context, &bufferId, (&bufferId)+1, true);

        kernel_unpackRecvBuffer(cl::EnqueueArgs(cl_queue, cl::NDRange(globalsize), cl::NDRange(cl_kernelLocalSize)),
                                cl_remoteHaloSpecsGpu[haloId], cl_valuesPerPointPerBuffer, cl_bufferId,
                                cl_cpuToGpuRecvBuffer[haloId], buf);

    } catch(cl::Error error) {
        std::cerr << "Tausch3D :: unpackRecvBufferCpuToGpu() :: OpenCL exception caught: " << error.what()
                  << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}
#endif


////////////////////////
/// Pack buffer and send data off

template <class buf_t> void Tausch3D<buf_t>::packAndSendCpu(size_t haloId, buf_t *buf, TauschPackRegion region, int mpitag) {
    packSendBufferCpu(haloId, 0, buf, region);
    sendCpu(haloId, mpitag);
}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::packAndSendCpuForGpu(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag) {
    packSendBufferCpuToGpu(haloId, 0, buf, region);
    sendCpuToGpu(haloId, msgtag);
}
template <class buf_t> void Tausch3D<buf_t>::packAndSendGpu(size_t haloId, cl::Buffer buf, int msgtag) {
    packSendBufferGpuToCpu(haloId, 0, buf);
    sendGpuToCpu(haloId, msgtag);
}
#endif


////////////////////////
/// Receive data and unpack

template <class buf_t> void Tausch3D<buf_t>::recvAndUnpackCpu(size_t haloId, buf_t *buf, TauschPackRegion region) {
    recvCpu(haloId);
    unpackRecvBufferCpu(haloId, 0, buf, region);
}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::recvAndUnpackCpuForGpu(size_t haloId, buf_t *buf, TauschPackRegion region) {
    recvCpuToGpu(haloId);
    unpackRecvBufferGpuToCpu(haloId, 0, buf, region);
}
template <class buf_t> void Tausch3D<buf_t>::recvAndUnpackGpu(size_t haloId, cl::Buffer buf) {
    recvCpu(haloId);
    unpackRecvBufferCpuToGpu(haloId, 0, buf);
}
#endif




///////////////////////////////////////////////////////////////
/// SOME GENERAL PURPOSE OPENCL FUNCTIONS - PART OF PUBLIC API
///////////////////////////////////////////////////////////////

#ifdef TAUSCH_OPENCL

template <class buf_t> void Tausch3D<buf_t>::enableOpenCL(bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName, bool showOpenCLBuildLog) {

    this->blockingSyncCpuGpu = blockingSyncCpuGpu;
    cl_kernelLocalSize = clLocalWorkgroupSize;
    this->showOpenCLBuildLog = showOpenCLBuildLog;

    sync_counter[0].store(0); sync_counter[1].store(0);
    sync_lock[0].store(0); sync_lock[1].store(0);

    setupOpenCL(giveOpenCLDeviceName);

    try {
        cl_valuesPerPointPerBuffer = cl::Buffer(cl_context, &valuesPerPointPerBuffer[0], &valuesPerPointPerBuffer[numBuffers], true);
    } catch(cl::Error error) {
        std::cerr << "Tausch3D :: enableOpenCL() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch3D<buf_t>::enableOpenCL(cl::Device cl_defaultDevice, cl::Context cl_context, cl::CommandQueue cl_queue, bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool showOpenCLBuildLog) {

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
        std::cerr << "Tausch3D :: enableOpenCL() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch3D<buf_t>::syncCpuAndGpu() {

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

template <class buf_t> int Tausch3D<buf_t>::obtainRemoteId(int msgtag) {
    for(int j = 0; j < remoteHaloNumPartsCpuForGpu; ++j) {
        if(msgtagsCpuToGpu[j].load() == msgtag)
            return j;
    }
    return 0;
}

template <class buf_t> void Tausch3D<buf_t>::setupOpenCL(bool giveOpenCLDeviceName) {

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

template <class buf_t> void Tausch3D<buf_t>::compileKernels() {

  std::string oclstr = R"d(
kernel void packSendBuffer(global const size_t * restrict const haloSpecs,
                           global const size_t * restrict const valuesPerPointPerBuffer, global int * restrict const bufferId,
                           global double * restrict const haloBuffer, global const double * restrict const buffer) {

    const int current = get_global_id(0);

    int maxSize = haloSpecs[3]*haloSpecs[4]*haloSpecs[5];

    if(current >= maxSize) return;

    int index = (current/(haloSpecs[3]*haloSpecs[4]) + haloSpecs[2])*haloSpecs[6]*haloSpecs[7] +
                ((current%(haloSpecs[3]*haloSpecs[4]))/haloSpecs[3] + haloSpecs[1]) * haloSpecs[6] +
                current%haloSpecs[3] + haloSpecs[0];

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

    int maxSize = haloSpecs[3]*haloSpecs[4]*haloSpecs[5];

    if(current >= maxSize) return;

    int index = (current/(haloSpecs[3]*haloSpecs[4]) + haloSpecs[2])*haloSpecs[6]*haloSpecs[7] +
                ((current%(haloSpecs[3]*haloSpecs[4]))/haloSpecs[3] + haloSpecs[1]) * haloSpecs[6] +
                current%haloSpecs[3] + haloSpecs[0];

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
                std::cout << "Tausch3D :: compileKernels() :: getBuildInfo :: OpenCL exception caught: " << err.what() << " (" << err.err() << ")" << std::endl;
            }
        }
    } catch(cl::Error error) {
        std::cout << "Tausch3D :: compileKernels() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        if(error.err() == -11) {
            try {
                std::string log = cl_programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_defaultDevice);
                std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl
                          << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** "
                          << std::endl << std::endl;
            } catch(cl::Error err) {
                std::cout << "Tausch3D :: compileKernels() :: getBuildInfo :: OpenCL exception caught: " << err.what() << " (" << err.err() << ")" << std::endl;
            }
        }
    }

}


#endif


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
