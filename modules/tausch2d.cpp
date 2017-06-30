#include "tausch2d.h"

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
    delete[] mpiRecvRequests;

    delete[] setupMpiRecv;
    delete[] setupMpiSend;

    delete[] valuesPerPointPerBuffer;
}

template <class buf_t> void Tausch2D<buf_t>::setLocalHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumPartsCpu = numHaloParts;
    localHaloSpecsCpu = new TauschHaloSpec[numHaloParts];
    mpiSendBuffer = new buf_t*[numHaloParts];
    mpiSendRequests = new MPI_Request[numHaloParts];
    setupMpiSend = new bool[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecsCpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsCpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecsCpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsCpu[i].haloY = haloSpecs[i].haloY;
        localHaloSpecsCpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsCpu[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecsCpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        mpiSendBuffer[i] = new buf_t[bufsize]{};

        setupMpiSend[i] = false;

    }

}

template <class buf_t> void Tausch2D<buf_t>::setRemoteHaloInfoCpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsCpu = numHaloParts;
    remoteHaloSpecsCpu = new TauschHaloSpec[numHaloParts];
    mpiRecvBuffer = new buf_t*[numHaloParts];
    mpiRecvRequests = new MPI_Request[numHaloParts];
    setupMpiRecv = new bool[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsCpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsCpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecsCpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsCpu[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecsCpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsCpu[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecsCpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
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
            bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsCpu[id].haloWidth*remoteHaloSpecsCpu[id].haloHeight;

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

template <class buf_t> void Tausch2D<buf_t>::packSendBufferCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    int size = region.width * region.height;
    for(int s = 0; s < size; ++s) {
        int index = (region.startY + s/region.width + localHaloSpecsCpu[haloId].haloY)*localHaloSpecsCpu[haloId].bufferWidth +
                    s%region.width + localHaloSpecsCpu[haloId].haloX + region.startX;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b]*size;
            mpiSendBuffer[haloId][offset + valuesPerPointPerBuffer[bufferId]*s + val] =
                    buf[valuesPerPointPerBuffer[bufferId]*index + val];
        }
    }

}

template <class buf_t> void Tausch2D<buf_t>::packSendBufferCpu(size_t haloId, size_t bufferId, buf_t *buf) {
    TauschPackRegion region;
    region.startX = 0;
    region.startY = 0;
    region.width = localHaloSpecsCpu[haloId].haloWidth;
    region.height = localHaloSpecsCpu[haloId].haloHeight;
    packSendBufferCpu(haloId, bufferId, buf, region);
}

template <class buf_t> void Tausch2D<buf_t>::sendCpu(size_t id, int mpitag) {

    if(!setupMpiSend[id]) {

        if(mpitag == -1) {
            std::cerr << "[Tausch2D] ERROR: MPI_Send for halo region #" << id << " hasn't been posted before, missing mpitag... Abort!" << std::endl;
            exit(1);
        }

        setupMpiSend[id] = true;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsCpu[id].haloWidth*localHaloSpecsCpu[id].haloHeight;
        MPI_Send_init(&mpiSendBuffer[id][0], bufsize, mpiDataType, localHaloSpecsCpu[id].remoteMpiRank,
                  mpitag, TAUSCH_COMM, &mpiSendRequests[id]);

    }

    MPI_Start(&mpiSendRequests[id]);

}

template <class buf_t> void Tausch2D<buf_t>::recvCpu(size_t id) {
    MPI_Wait(&mpiRecvRequests[id], MPI_STATUS_IGNORE);
}

template <class buf_t> void Tausch2D<buf_t>::unpackRecvBufferCpu(size_t haloId, size_t bufferId, buf_t *buf) {

    int size = remoteHaloSpecsCpu[haloId].haloWidth * remoteHaloSpecsCpu[haloId].haloHeight;
    for(int s = 0; s < size; ++s) {
        int index = (s/remoteHaloSpecsCpu[haloId].haloWidth + remoteHaloSpecsCpu[haloId].haloY)*remoteHaloSpecsCpu[haloId].bufferWidth +
                    s%remoteHaloSpecsCpu[haloId].haloWidth + remoteHaloSpecsCpu[haloId].haloX;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b]*size;
            buf[valuesPerPointPerBuffer[bufferId]*index + val] =
                    mpiRecvBuffer[haloId][offset + valuesPerPointPerBuffer[bufferId]*s + val];
        }
    }

}

template <class buf_t> void Tausch2D<buf_t>::packAndSendCpu(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region, int mpitag) {
    packSendBufferCpu(haloId, bufferId, buf, region);
    sendCpu(haloId, mpitag);
}

template <class buf_t> void Tausch2D<buf_t>::packAndSendCpu(size_t haloId, size_t bufferId, buf_t *buf, int mpitag) {
    TauschPackRegion region;
    region.startX = 0;
    region.startY = 0;
    region.width = localHaloSpecsCpu[haloId].haloWidth;
    region.height = localHaloSpecsCpu[haloId].haloHeight;
    packSendBufferCpu(haloId, bufferId, buf, region);
    sendCpu(haloId, mpitag);
}

template <class buf_t> void Tausch2D<buf_t>::recvAndUnpackCpu(size_t haloId, size_t bufferId, buf_t *buf) {
    recvCpu(haloId);
    unpackRecvBufferCpu(haloId, bufferId, buf);
}


#ifdef TAUSCH_OPENCL

template <class buf_t> void Tausch2D<buf_t>::enableOpenCL(bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName, bool showOpenCLBuildLog) {

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

template <class buf_t> void Tausch2D<buf_t>::setLocalHaloInfoCpuForGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumPartsCpuForGpu = numHaloParts;
    localHaloSpecsCpuForGpu = new TauschHaloSpec[numHaloParts];
    cpuToGpuSendBuffer = new std::atomic<buf_t>*[numHaloParts];
    numBuffersPackedCpuToGpu =  new size_t[numHaloParts]{};
    msgtagsCpuToGpu = new std::atomic<int>[numHaloParts]{};

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecsCpuForGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsCpuForGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecsCpuForGpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsCpuForGpu[i].haloY = haloSpecs[i].haloY;
        localHaloSpecsCpuForGpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsCpuForGpu[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecsCpuForGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        cpuToGpuSendBuffer[i] = new std::atomic<buf_t>[bufsize]{};

    }

}

template <class buf_t> void Tausch2D<buf_t>::setRemoteHaloInfoCpuForGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsCpuForGpu = numHaloParts;
    remoteHaloSpecsCpuForGpu = new TauschHaloSpec[numHaloParts];
    gpuToCpuRecvBuffer = new buf_t*[numHaloParts];
    numBuffersUnpackedGpuToCpu = new size_t[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsCpuForGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsCpuForGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecsCpuForGpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsCpuForGpu[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecsCpuForGpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsCpuForGpu[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecsCpuForGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        numBuffersUnpackedGpuToCpu[i] = 0;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        gpuToCpuRecvBuffer[i] = new buf_t[bufsize];

    }

}

template <class buf_t> void Tausch2D<buf_t>::setLocalHaloInfoGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumPartsGpu = numHaloParts;
    localHaloSpecsGpu = new TauschHaloSpec[numHaloParts];
    gpuToCpuSendBuffer = new std::atomic<buf_t>*[numHaloParts];
    msgtagsGpuToCpu = new std::atomic<int>[numHaloParts]{};

    try {
        cl_gpuToCpuSendBuffer = new cl::Buffer[numHaloParts];
        cl_localHaloSpecsGpu = new cl::Buffer[numHaloParts];
        cl_numBuffersPackedGpuToCpu = new cl::Buffer[numHaloParts];
        for(int n = 0; n < numHaloParts; ++n) {
            cl_numBuffersPackedGpuToCpu[n] = cl::Buffer(cl_context, CL_MEM_READ_WRITE, sizeof(size_t));
            cl_queue.enqueueFillBuffer(cl_numBuffersPackedGpuToCpu[n], 0, 0, sizeof(size_t));
        }
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: setLocalHaloInfoGpu() (1) :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecsGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecsGpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsGpu[i].haloY = haloSpecs[i].haloY;
        localHaloSpecsGpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsGpu[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecsGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        gpuToCpuSendBuffer[i] = new std::atomic<buf_t>[bufsize]{};

        size_t tmpHaloSpecs[6] = {haloSpecs[i].haloX, haloSpecs[i].haloY, haloSpecs[i].haloWidth, haloSpecs[i].haloHeight, haloSpecs[i].bufferWidth, haloSpecs[i].bufferHeight};

        try {
            cl_gpuToCpuSendBuffer[i] = cl::Buffer(cl_context, CL_MEM_READ_WRITE, bufsize*sizeof(buf_t));
            cl_localHaloSpecsGpu[i] = cl::Buffer(cl_context, &tmpHaloSpecs[0], &tmpHaloSpecs[6], true);
        } catch(cl::Error error) {
            std::cerr << "Tausch2D :: setLocalHaloInfoGpu() (2) :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

    }

}

template <class buf_t> void Tausch2D<buf_t>::setRemoteHaloInfoGpu(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsGpu = numHaloParts;
    remoteHaloSpecsGpu = new TauschHaloSpec[numHaloParts];
    cpuToGpuRecvBuffer = new buf_t*[numHaloParts];

    try {
        cl_cpuToGpuRecvBuffer = new cl::Buffer[remoteHaloNumPartsGpu];
        cl_numBuffersUnpackedCpuToGpu = new cl::Buffer[remoteHaloNumPartsGpu];
        for(int i = 0; i < remoteHaloNumPartsGpu; ++i)
            cl_numBuffersUnpackedCpuToGpu[i] = cl::Buffer(cl_context, CL_MEM_READ_WRITE, sizeof(double));
        cl_remoteHaloSpecsGpu = new cl::Buffer[numHaloParts];
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: enableOpenCL() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecsGpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsGpu[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecsGpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsGpu[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecsGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        cpuToGpuRecvBuffer[i] = new buf_t[bufsize];

        size_t tmpHaloSpecs[6] = {haloSpecs[i].haloX, haloSpecs[i].haloY, haloSpecs[i].haloWidth, haloSpecs[i].haloHeight, haloSpecs[i].bufferWidth, haloSpecs[i].bufferHeight, };

        try {
            cl_cpuToGpuRecvBuffer[i] = cl::Buffer(cl_context, CL_MEM_READ_WRITE, bufsize*sizeof(double));
            cl_remoteHaloSpecsGpu[i] = cl::Buffer(cl_context, &tmpHaloSpecs[0], &tmpHaloSpecs[6], true);
        } catch(cl::Error error) {
            std::cerr << "Tausch2D :: setRemoteHaloInfo() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

    }

}

template <class buf_t> void Tausch2D<buf_t>::packNextSendBufferCpuToGpu(size_t id, buf_t *buf) {

    int size = localHaloSpecsCpuForGpu[id].haloWidth * localHaloSpecsCpuForGpu[id].haloHeight;
    for(int s = 0; s < size; ++s) {
        int index = (s/localHaloSpecsCpuForGpu[id].haloWidth + localHaloSpecsCpuForGpu[id].haloY)*localHaloSpecsCpuForGpu[id].bufferWidth+
                    s%localHaloSpecsCpuForGpu[id].haloWidth +localHaloSpecsCpuForGpu[id].haloX;
        for(int val = 0; val < valuesPerPointPerBuffer[numBuffersPackedCpuToGpu[id]]; ++val) {
            int offset = 0;
            for(int b = 0; b < numBuffersPackedCpuToGpu[id]; ++b)
                offset += valuesPerPointPerBuffer[b]*size;
            cpuToGpuSendBuffer[id][offset + valuesPerPointPerBuffer[numBuffersPackedCpuToGpu[id]]*s + val].store(buf[valuesPerPointPerBuffer[numBuffersPackedCpuToGpu[id]]*index + val]);
        }
    }
    ++numBuffersPackedCpuToGpu[id];

}

template <class buf_t> void Tausch2D<buf_t>::packNextSendBufferGpuToCpu(size_t id, cl::Buffer buf) {

    try {

        auto kernel_packNextSendBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                    (cl_programs, "packNextSendBuffer");

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsGpu[id].haloWidth*localHaloSpecsGpu[id].haloHeight;

        int globalsize = (bufsize/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_packNextSendBuffer(cl::EnqueueArgs(cl_queue, cl::NDRange(globalsize), cl::NDRange(cl_kernelLocalSize)),
                                  cl_localHaloSpecsGpu[id], cl_valuesPerPointPerBuffer,
                                  cl_numBuffersPackedGpuToCpu[id], cl_gpuToCpuSendBuffer[id], buf);

        auto kernel_inc = cl::make_kernel<cl::Buffer>(cl_programs, "incrementBuffer");
        kernel_inc(cl::EnqueueArgs(cl_queue, cl::NDRange(1), cl::NDRange(1)), cl_numBuffersPackedGpuToCpu[id]);

    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch2D<buf_t>::sendCpuToGpu(size_t id, int msgtag) {
    numBuffersPackedCpuToGpu[id] = 0;
    msgtagsCpuToGpu[id].store(msgtag);
    syncCpuAndGpu();
}

template <class buf_t> void Tausch2D<buf_t>::sendGpuToCpu(size_t id, int msgtag) {

    msgtagsGpuToCpu[id].store(msgtag);

    try {

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsGpu[id].haloWidth*localHaloSpecsGpu[id].haloHeight;

        buf_t *tmp = new buf_t[bufsize];
        cl::copy(cl_queue, cl_gpuToCpuSendBuffer[id], &tmp[0], &tmp[bufsize]);
        for(int i = 0; i < bufsize; ++i)
            gpuToCpuSendBuffer[id][i].store(tmp[i]);

    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: sendGpuToCpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    syncCpuAndGpu();

}

template <class buf_t> void Tausch2D<buf_t>::recvCpuToGpu(size_t id, int msgtag) {

    int remoteid = 0;
    for(int j = 0; j < remoteHaloNumPartsGpu; ++j) {
        if(msgtagsCpuToGpu[j].load() == msgtag) {
            remoteid = j;
            break;
        }
    }

    syncCpuAndGpu();

    size_t bufsize = 0;
    for(int n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsGpu[id].haloWidth*remoteHaloSpecsGpu[id].haloHeight;

    for(int j = 0; j < bufsize; ++j)
        cpuToGpuRecvBuffer[id][j] = cpuToGpuSendBuffer[remoteid][j].load();

    try {
        cl_cpuToGpuRecvBuffer[id] = cl::Buffer(cl_context, &cpuToGpuRecvBuffer[id][0], &cpuToGpuRecvBuffer[id][bufsize], false);
        cl_queue.enqueueFillBuffer(cl_numBuffersUnpackedCpuToGpu[id], 0, 0, sizeof(double));
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: recvCpuToGpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch2D<buf_t>::recvGpuToCpu(size_t id, int msgtag) {

    int remoteid = 0;
    for(int j = 0; j < remoteHaloNumPartsCpuForGpu; ++j) {
        if(msgtagsCpuToGpu[j].load() == msgtag) {
            remoteid = j;
            break;
        }
    }

    size_t bufsize = 0;
    for(int n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsCpuForGpu[id].haloWidth*remoteHaloSpecsCpuForGpu[id].haloHeight;
    for(int i = 0; i < bufsize; ++i)
        gpuToCpuRecvBuffer[id][i] = gpuToCpuSendBuffer[remoteid][i].load();

    syncCpuAndGpu();
    numBuffersUnpackedGpuToCpu[id] = 0;

}

template <class buf_t> void Tausch2D<buf_t>::unpackNextRecvBufferGpuToCpu(size_t id, buf_t *buf) {

    int size = remoteHaloSpecsCpuForGpu[id].haloWidth * remoteHaloSpecsCpuForGpu[id].haloHeight;
    for(int s = 0; s < size; ++s) {
        int index = (s/remoteHaloSpecsCpuForGpu[id].haloWidth + remoteHaloSpecsCpuForGpu[id].haloY)*remoteHaloSpecsCpuForGpu[id].bufferWidth +
                    s%remoteHaloSpecsCpuForGpu[id].haloWidth +remoteHaloSpecsCpuForGpu[id].haloX;
        for(int val = 0; val < valuesPerPointPerBuffer[numBuffersUnpackedGpuToCpu[id]]; ++val) {
            int offset = 0;
            for(int b = 0; b < numBuffersUnpackedGpuToCpu[id]; ++b)
                offset += valuesPerPointPerBuffer[b]*size;
            buf[valuesPerPointPerBuffer[numBuffersUnpackedGpuToCpu[id]]*index + val] = gpuToCpuRecvBuffer[id][offset + valuesPerPointPerBuffer[numBuffersUnpackedGpuToCpu[id]]*s + val];
        }
    }
    ++numBuffersUnpackedGpuToCpu[id];

}

template <class buf_t> void Tausch2D<buf_t>::unpackNextRecvBufferCpuToGpu(size_t id, cl::Buffer buf) {

    try {
        auto kernel_unpackRecvBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                (cl_programs, "unpackNextRecvBuffer");

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsGpu[id].haloWidth*remoteHaloSpecsGpu[id].haloHeight;

        int globalsize = (bufsize/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        kernel_unpackRecvBuffer(cl::EnqueueArgs(cl_queue, cl::NDRange(globalsize), cl::NDRange(cl_kernelLocalSize)),
                                cl_remoteHaloSpecsGpu[id], cl_valuesPerPointPerBuffer,
                                cl_numBuffersUnpackedCpuToGpu[id], cl_cpuToGpuRecvBuffer[id], buf);

        auto kernel_inc = cl::make_kernel<cl::Buffer>(cl_programs, "incrementBuffer");
        kernel_inc(cl::EnqueueArgs(cl_queue, cl::NDRange(1), cl::NDRange(1)), cl_numBuffersUnpackedCpuToGpu[id]);

    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: unpackNextRecvBufferCpuToGpu() :: OpenCL exception caught: " << error.what()
                  << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch2D<buf_t>::syncCpuAndGpu() {

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

kernel void packNextSendBuffer(global const size_t * restrict const haloSpecs,
                               global const size_t * restrict const valuesPerPointPerBuffer, global int * restrict const numBuffersPacked,
                               global double * restrict const haloBuffer, global const double * restrict const buffer) {

    const int current = get_global_id(0);

    int maxSize = haloSpecs[2]*haloSpecs[3];

    if(current >= maxSize) return;

    int index = (current/haloSpecs[2] + haloSpecs[1])*haloSpecs[4] +
                 current%haloSpecs[2] + haloSpecs[0];

    for(int val = 0; val < valuesPerPointPerBuffer[*numBuffersPacked]; ++val) {
        int offset = 0;
        for(int b = 0; b < *numBuffersPacked; ++b)
            offset += valuesPerPointPerBuffer[b]*maxSize;
        haloBuffer[offset + valuesPerPointPerBuffer[*numBuffersPacked]*current + val] = buffer[valuesPerPointPerBuffer[*numBuffersPacked]*index + val];
    }

}

kernel void unpackNextRecvBuffer(global const size_t * restrict const haloSpecs,
                                 global const size_t * restrict const valuesPerPointPerBuffer, global int * restrict const numBuffersUnpacked,
                                 global const double * restrict const haloBuffer, global double * restrict const buffer) {

    const int current = get_global_id(0);

    int maxSize = haloSpecs[2]*haloSpecs[3];

    if(current >= maxSize) return;

    int index = (current/haloSpecs[2] + haloSpecs[1])*haloSpecs[4] +
                 current%haloSpecs[2] + haloSpecs[0];

    for(int val = 0; val < valuesPerPointPerBuffer[*numBuffersUnpacked]; ++val) {
        int offset = 0;
        for(int b = 0; b < *numBuffersUnpacked; ++b)
            offset += valuesPerPointPerBuffer[b]*maxSize;
        buffer[valuesPerPointPerBuffer[*numBuffersUnpacked]*index + val] =
                haloBuffer[offset + valuesPerPointPerBuffer[*numBuffersUnpacked]*current + val];
    }

}

kernel void incrementBuffer(global int * restrict const buffer) {
    if(get_global_id(0) > 0) return;
    *buffer += 1;
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
                std::cout << "Tausch2D :: compileKernels() :: getBuildInfo :: OpenCL exception caught: " << err.what() << " (" << err.err() << ")" << std::endl;
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
                std::cout << "Tausch2D :: compileKernels() :: getBuildInfo :: OpenCL exception caught: " << err.what() << " (" << err.err() << ")" << std::endl;
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
