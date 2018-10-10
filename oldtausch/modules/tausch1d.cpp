#include "tausch1d.h"

template <class buf_t> Tausch1D<buf_t>::Tausch1D(MPI_Datatype mpiDataType,
                                                   size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm, bool duplicateCommunicator) {

    if(duplicateCommunicator)
        MPI_Comm_dup(comm, &TAUSCH_COMM);
    else
        TAUSCH_COMM = comm;

    // get MPI info
    MPI_Comm_rank(TAUSCH_COMM, &mpiRank);
    MPI_Comm_size(TAUSCH_COMM, &mpiSize);

    this->numBuffers = numBuffers;

    this->valuesPerPointPerBuffer = new size_t[numBuffers];
    if(valuesPerPointPerBuffer == NULL)
        for(int i = 0; i < numBuffers; ++i)
            this->valuesPerPointPerBuffer[i] = 1;
    else
        for(int i = 0; i < numBuffers; ++i)
            this->valuesPerPointPerBuffer[i] = valuesPerPointPerBuffer[i];

    this->mpiDataType = mpiDataType;

#ifdef TAUSCH_OPENCL
    setupCpuWithGpu = false;
    setupGpuWithCpu = false;
#endif

}

template <class buf_t> Tausch1D<buf_t>::~Tausch1D() {

    for(int i = 0; i < mpiSendBuffer.size(); ++i) {
        if(std::find(alreadyDeletedLocalHaloIds.begin(), alreadyDeletedLocalHaloIds.end(), i) == alreadyDeletedLocalHaloIds.end())
            delete[] mpiSendBuffer[i];
    }

    for(int i = 0; i < mpiRecvBuffer.size(); ++i) {
        if(std::find(alreadyDeletedRemoteHaloIds.begin(), alreadyDeletedRemoteHaloIds.end(), i) == alreadyDeletedRemoteHaloIds.end())
            delete[] mpiRecvBuffer[i];
    }

#ifdef TAUSCH_OPENCL

    if(setupCpuWithGpu) {

        for(int i = 0; i < localHaloNumPartsCpuForGpu; ++i)
            delete[] cpuToGpuSendBuffer[i];
        delete[] localHaloSpecsCpuForGpu;
        delete[] cpuToGpuSendBuffer;
        delete[] msgtagsCpuToGpu;

        for(int i = 0; i < remoteHaloNumPartsCpuForGpu; ++i)
            delete[] cpuToGpuRecvBuffer[i];
        delete[] remoteHaloSpecsCpuForGpu;
        delete[] cpuToGpuRecvBuffer;

    }

    if(setupGpuWithCpu) {

        for(int i = 0; i < localHaloNumPartsGpu; ++i)
            delete[] gpuToCpuSendBuffer[i];
        delete[] localHaloSpecsGpu;
        delete[] gpuToCpuSendBuffer;
        delete[] msgtagsGpuToCpu;
        delete[] cl_gpuToCpuSendBuffer;
        delete[] cl_localHaloSpecsGpu;

        for(int i = 0; i < remoteHaloNumPartsGpu; ++i)
            delete[] gpuToCpuRecvBuffer[i];
        delete[] remoteHaloSpecsGpu;
        delete[] gpuToCpuRecvBuffer;
        delete[] cl_gpuToCpuSendBuffer;
        delete[] cl_remoteHaloSpecsGpu;

    }

#endif

    delete[] valuesPerPointPerBuffer;
}

/////////////////////////////////////////////
/// PUBLIC API FUNCTION
/////////////////////////////////////////////


////////////////////////
/// Set local halo info

template <class buf_t> int Tausch1D<buf_t>::addLocalHaloInfoCwC(TauschHaloSpec haloSpec) {

    localHaloSpecsCpu.push_back(haloSpec);

    size_t haloSize = haloSpec.haloWidth;
    if(haloSpec.haloIndicesInBuffer.size() > 0)
        haloSize = haloSpec.haloIndicesInBuffer.size();

    size_t bufsize = 0;
    for(size_t n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*haloSize;
    mpiSendBuffer.push_back(new buf_t[bufsize]());

    setupMpiSend.push_back(false);

    mpiSendRequests.push_back(MPI_Request());

    return mpiSendBuffer.size()-1;

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch1D<buf_t>::setLocalHaloInfoCwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumPartsCpuForGpu = numHaloParts;
    localHaloSpecsCpuForGpu = new TauschHaloSpec[numHaloParts];
    cpuToGpuSendBuffer = new std::atomic<buf_t>*[numHaloParts];
    msgtagsCpuToGpu = new std::atomic<int>[numHaloParts]{};

    setupCpuWithGpu = true;

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecsCpuForGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsCpuForGpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsCpuForGpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsCpuForGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth;
        cpuToGpuSendBuffer[i] = new std::atomic<buf_t>[bufsize]{};

    }

}

template <class buf_t> void Tausch1D<buf_t>::setLocalHaloInfoGwC(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumPartsGpu = numHaloParts;
    localHaloSpecsGpu = new TauschHaloSpec[numHaloParts];
    gpuToCpuSendBuffer = new std::atomic<buf_t>*[numHaloParts];
    msgtagsGpuToCpu = new std::atomic<int>[numHaloParts]{};

    setupGpuWithCpu = true;

    try {
        cl_gpuToCpuSendBuffer = new cl::Buffer[numHaloParts];
        cl_localHaloSpecsGpu = new cl::Buffer[numHaloParts];
    } catch(cl::Error error) {
        std::cerr << "Tausch1D :: setLocalHaloInfoGpu() (1) :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    for(int i = 0; i < numHaloParts; ++i) {

        localHaloSpecsGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsGpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsGpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth;
        gpuToCpuSendBuffer[i] = new std::atomic<buf_t>[bufsize]{};

        size_t tmpHaloSpecs[3] = {haloSpecs[i].haloX, haloSpecs[i].haloWidth, haloSpecs[i].bufferWidth};

        try {
            cl_gpuToCpuSendBuffer[i] = cl::Buffer(cl_context, CL_MEM_READ_WRITE, bufsize*sizeof(buf_t));
            cl_localHaloSpecsGpu[i] = cl::Buffer(cl_context, &tmpHaloSpecs[0], &tmpHaloSpecs[3], true);
        } catch(cl::Error error) {
            std::cerr << "Tausch1D :: setLocalHaloInfoGpu() (2) :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")"
                      << std::endl;
            exit(1);
        }

    }

}
#endif


///////////////////////////
/// Destroy local halo info

template <class buf_t> void Tausch1D<buf_t>::delLocalHaloInfoCwC(size_t haloId) {
    delete[] mpiSendBuffer[haloId];
    mpiSendRequests[haloId] = MPI_Request();
    alreadyDeletedLocalHaloIds.push_back(haloId);
}


////////////////////////
/// Set remote halo info

template <class buf_t> int Tausch1D<buf_t>::addRemoteHaloInfoCwC(TauschHaloSpec haloSpec) {

    remoteHaloSpecsCpu.push_back(haloSpec);

    size_t haloSize = haloSpec.haloWidth;
    if(haloSpec.haloIndicesInBuffer.size() > 0)
        haloSize = haloSpec.haloIndicesInBuffer.size();

    size_t bufsize = 0;
    for(size_t n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*haloSize;
    mpiRecvBuffer.push_back(new buf_t[bufsize]());

    setupMpiRecv.push_back(false);

    mpiRecvRequests.push_back(MPI_Request());

    return mpiRecvBuffer.size()-1;

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch1D<buf_t>::setRemoteHaloInfoCwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsCpuForGpu = numHaloParts;
    remoteHaloSpecsCpuForGpu = new TauschHaloSpec[numHaloParts];
    gpuToCpuRecvBuffer = new buf_t*[numHaloParts];

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsCpuForGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsCpuForGpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsCpuForGpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsCpuForGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth;
        gpuToCpuRecvBuffer[i] = new buf_t[bufsize];

    }

}

template <class buf_t> void Tausch1D<buf_t>::setRemoteHaloInfoGwC(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsGpu = numHaloParts;
    remoteHaloSpecsGpu = new TauschHaloSpec[numHaloParts];
    cpuToGpuRecvBuffer = new buf_t*[numHaloParts];

    try {
        cl_cpuToGpuRecvBuffer = new cl::Buffer[remoteHaloNumPartsGpu];
        cl_remoteHaloSpecsGpu = new cl::Buffer[numHaloParts];
    } catch(cl::Error error) {
        std::cerr << "Tausch1D :: enableOpenCL() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    for(int i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsGpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsGpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth;
        cpuToGpuRecvBuffer[i] = new buf_t[bufsize];

        size_t tmpHaloSpecs[3] = {haloSpecs[i].haloX, haloSpecs[i].haloWidth, haloSpecs[i].bufferWidth};

        try {
            cl_cpuToGpuRecvBuffer[i] = cl::Buffer(cl_context, CL_MEM_READ_WRITE, bufsize*sizeof(double));
            cl_remoteHaloSpecsGpu[i] = cl::Buffer(cl_context, &tmpHaloSpecs[0], &tmpHaloSpecs[3], true);
        } catch(cl::Error error) {
            std::cerr << "Tausch1D :: setRemoteHaloInfo() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

    }

}
#endif


///////////////////////////
/// Destroy remote halo info

template <class buf_t> void Tausch1D<buf_t>::delRemoteHaloInfoCwC(size_t haloId) {
    delete[] mpiRecvBuffer[haloId];
    mpiRecvRequests[haloId] = MPI_Request();
    alreadyDeletedRemoteHaloIds.push_back(haloId);
}


////////////////////////
/// Post Receives

template <class buf_t> void Tausch1D<buf_t>::postReceiveCwC(size_t haloId, int msgtag, int remoteMpiRank) {

    if(!setupMpiRecv[haloId]) {

        if(msgtag == -1) {
            std::cerr << "[Tausch1D] ERROR: MPI_Recv for halo region #" << haloId << " hasn't been posted before, missing mpitag... Abort!"
                      << std::endl;
            exit(1);
        }

        setupMpiRecv[haloId] = true;

        size_t haloSize = remoteHaloSpecsCpu[haloId].haloWidth;
        if(remoteHaloSpecsCpu[haloId].haloIndicesInBuffer.size() > 0)
            haloSize = remoteHaloSpecsCpu[haloId].haloIndicesInBuffer.size();

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSize;

        if(remoteMpiRank == -1)
            remoteMpiRank = remoteHaloSpecsCpu[haloId].remoteMpiRank;

        MPI_Recv_init(&mpiRecvBuffer[haloId][0], bufsize, mpiDataType, remoteMpiRank,
                      msgtag, TAUSCH_COMM, &mpiRecvRequests[haloId]);

    }

    MPI_Start(&mpiRecvRequests[haloId]);

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch1D<buf_t>::postReceiveCwG(size_t haloId, int msgtag) {
    msgtagsCpuToGpu[haloId].store(msgtag);
}

template <class buf_t> void Tausch1D<buf_t>::postReceiveGwC(size_t haloId, int msgtag) {
    msgtagsGpuToCpu[haloId].store(msgtag);
}
#endif


////////////////////////
/// Post ALL Receives

template <class buf_t> void Tausch1D<buf_t>::postAllReceivesCwC(int *msgtag) {

    if(msgtag == NULL) {
        msgtag = new int[remoteHaloSpecsCpu.size()];
        for(int id = 0; id < remoteHaloSpecsCpu.size(); ++id)
            msgtag[id] = -1;
    }

    for(int id = 0; id < remoteHaloSpecsCpu.size(); ++id) {
        if(std::find(alreadyDeletedLocalHaloIds.begin(), alreadyDeletedLocalHaloIds.end(), id) == alreadyDeletedLocalHaloIds.end())
            postReceiveCwC(id,msgtag[id]);
    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch1D<buf_t>::postAllReceivesCwG(int *msgtag) {

    if(msgtag == NULL) {
        std::cerr << "Tausch1D::postAllReceives :: ERROR :: msgtag cannot be NULL for device " << TAUSCH_CPU+TAUSCH_WITHGPU << std::endl;
        return;
    }

    for(int id = 0; id < remoteHaloNumPartsCpuForGpu; ++id)
        postReceiveCwG(id, msgtag[id]);

}

template <class buf_t> void Tausch1D<buf_t>::postAllReceivesGwC(int *msgtag) {

    if(msgtag == NULL) {
        std::cerr << "Tausch1D::postAllReceives :: ERROR :: msgtag cannot be NULL for device " << TAUSCH_GPU << std::endl;
        return;
    }

    for(int id = 0; id < remoteHaloNumPartsGpu; ++id)
        postReceiveGwC(id, msgtag[id]);

}
#endif


////////////////////////
/// Pack send buffer

template <class buf_t> void Tausch1D<buf_t>::packSendBufferCwC(size_t haloId, size_t bufferId, buf_t *buf) {
    TauschPackRegion region;
    if(localHaloSpecsCpu[haloId].haloIndicesInBuffer.size() == 0) {
        region.x = 0;
        region.width = localHaloSpecsCpu[haloId].haloWidth;
    } else {
        region.startAtIndex = 0;
        region.endAtIndex = localHaloSpecsCpu[haloId].haloIndicesInBuffer.size();
    }
    packSendBufferCwC(haloId, bufferId, buf, region);
}

template <class buf_t> void Tausch1D<buf_t>::packSendBufferCwC(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    if(localHaloSpecsCpu[haloId].haloIndicesInBuffer.size() > 0) {

        for(size_t index = region.startAtIndex; index < region.endAtIndex; ++index)
            mpiSendBuffer[haloId][index] = buf[localHaloSpecsCpu[haloId].haloIndicesInBuffer[index]];

        return;

    }
    for(int s = 0; s < region.width; ++s) {
        int bufIndex = localHaloSpecsCpu[haloId].haloX + s + region.x;
        int mpiIndex = s + region.x;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b]*localHaloSpecsCpu[haloId].haloWidth;
            mpiSendBuffer[haloId][offset + valuesPerPointPerBuffer[bufferId]*mpiIndex + val] =
                    buf[valuesPerPointPerBuffer[bufferId]*bufIndex + val];
        }
    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch1D<buf_t>::packSendBufferCwG(size_t haloId, size_t bufferId, buf_t *buf) {
    TauschPackRegion region;
    region.x = 0;
    region.width = localHaloSpecsCpuForGpu[haloId].haloWidth;
    packSendBufferCwG(haloId, bufferId, buf, region);
}

template <class buf_t> void Tausch1D<buf_t>::packSendBufferCwG(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    for(int s = 0; s < region.width; ++s) {
        int bufIndex = s+localHaloSpecsCpuForGpu[haloId].haloX + region.x;
        int mpiIndex = s + region.x;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b]*localHaloSpecsCpuForGpu[haloId].haloWidth;
            cpuToGpuSendBuffer[haloId][offset + valuesPerPointPerBuffer[bufferId]*mpiIndex + val]
                    .store(buf[valuesPerPointPerBuffer[bufferId]*bufIndex + val]);
        }
    }

}

template <class buf_t> void Tausch1D<buf_t>::packSendBufferGwC(size_t haloId, size_t bufferId, cl::Buffer buf) {

    try {

        auto kernel_packNextSendBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                    (cl_programs, "packSendBuffer");

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsGpu[haloId].haloWidth;

        int globalsize = (bufsize/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        cl::Buffer cl_bufferId(cl_context, &bufferId, (&bufferId)+1, true);

        kernel_packNextSendBuffer(cl::EnqueueArgs(cl_queue, cl::NDRange(globalsize), cl::NDRange(cl_kernelLocalSize)),
                                  cl_localHaloSpecsGpu[haloId], cl_valuesPerPointPerBuffer,
                                  cl_bufferId, cl_gpuToCpuSendBuffer[haloId], buf);

    } catch(cl::Error error) {
        std::cerr << "Tausch1D :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}
#endif


////////////////////////
/// Send data off

template <class buf_t> void Tausch1D<buf_t>::sendCwC(size_t haloId, int msgtag, int remoteMpiRank) {

    if(!setupMpiSend[haloId]) {

        if(msgtag == -1) {
            std::cerr << "[Tausch1D] ERROR: MPI_Send for halo region #" << haloId << " hasn't been posted before, missing mpitag... Abort!"
                      << std::endl;
            exit(1);
        }

        setupMpiSend[haloId] = true;

        size_t haloSize = localHaloSpecsCpu[haloId].haloWidth;
        if(localHaloSpecsCpu[haloId].haloIndicesInBuffer.size() > 0)
            haloSize = localHaloSpecsCpu[haloId].haloIndicesInBuffer.size();

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSize;

        int receiver = localHaloSpecsCpu[haloId].remoteMpiRank;
        if(remoteMpiRank != -1)
            receiver = remoteMpiRank;

        MPI_Send_init(&mpiSendBuffer[haloId][0], bufsize, mpiDataType, receiver,
                      msgtag, TAUSCH_COMM, &mpiSendRequests[haloId]);

    } else
        MPI_Wait(&mpiSendRequests[haloId], MPI_STATUS_IGNORE);

    MPI_Start(&mpiSendRequests[haloId]);

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch1D<buf_t>::sendCwG(size_t haloId, int msgtag) {
    msgtagsCpuToGpu[haloId].store(msgtag);
    syncCpuAndGpu();
}

template <class buf_t> void Tausch1D<buf_t>::sendGwC(size_t haloId, int msgtag) {

    msgtagsGpuToCpu[haloId].store(msgtag);

    try {

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsGpu[haloId].haloWidth;

        buf_t *tmp = new buf_t[bufsize];
        cl::copy(cl_queue, cl_gpuToCpuSendBuffer[haloId], &tmp[0], &tmp[bufsize]);
        for(int i = 0; i < bufsize; ++i)
            gpuToCpuSendBuffer[haloId][i].store(tmp[i]);

    } catch(cl::Error error) {
        std::cerr << "Tausch1D :: sendGpuToCpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    syncCpuAndGpu();

}
#endif


////////////////////////
/// Receive data

template <class buf_t> void Tausch1D<buf_t>::recvCwC(size_t haloId) {
    MPI_Wait(&mpiRecvRequests[haloId], MPI_STATUS_IGNORE);
}

template <class buf_t> void Tausch1D<buf_t>::recvAllCwC() {
    MPI_Waitall(mpiRecvRequests.size(), mpiRecvRequests.data(), MPI_STATUSES_IGNORE);
}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch1D<buf_t>::recvGwC(size_t haloId) {

    syncCpuAndGpu();

    int remoteid = obtainRemoteId(msgtagsCpuToGpu[haloId]);

    size_t bufsize = 0;
    for(int n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsGpu[haloId].haloWidth;

    for(int j = 0; j < bufsize; ++j)
        cpuToGpuRecvBuffer[haloId][j] = cpuToGpuSendBuffer[remoteid][j].load();

    try {
        cl_cpuToGpuRecvBuffer[haloId] = cl::Buffer(cl_context, &cpuToGpuRecvBuffer[haloId][0], &cpuToGpuRecvBuffer[haloId][bufsize], false);
    } catch(cl::Error error) {
        std::cerr << "Tausch1D :: recvCpuToGpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch1D<buf_t>::recvCwG(size_t haloId) {

    syncCpuAndGpu();

    int remoteid = obtainRemoteId(msgtagsGpuToCpu[haloId]);

    size_t bufsize = 0;
    for(int n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsCpuForGpu[haloId].haloWidth;
    for(int i = 0; i < bufsize; ++i)
        gpuToCpuRecvBuffer[haloId][i] = gpuToCpuSendBuffer[remoteid][i].load();

}
#endif


////////////////////////
/// Unpack received data

template <class buf_t> void Tausch1D<buf_t>::unpackRecvBufferCwC(size_t haloId, size_t bufferId, buf_t *buf) {
    TauschPackRegion region;
    if(remoteHaloSpecsCpu[haloId].haloIndicesInBuffer.size() == 0) {
        region.x = 0;
        region.width = remoteHaloSpecsCpu[haloId].haloWidth;
    } else {
        region.startAtIndex = 0;
        region.endAtIndex = remoteHaloSpecsCpu[haloId].haloIndicesInBuffer.size();
    }
    unpackRecvBufferCwC(haloId, bufferId, buf, region);
}

template <class buf_t> void Tausch1D<buf_t>::unpackRecvBufferCwC(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    if(remoteHaloSpecsCpu[haloId].haloIndicesInBuffer.size() != 0) {

        for(size_t index = region.startAtIndex; index < region.endAtIndex; ++index)
            buf[remoteHaloSpecsCpu[haloId].haloIndicesInBuffer[index]] = mpiRecvBuffer[haloId][index];

        return;

    }

    for(int s = 0; s < region.width; ++s) {
        int bufIndex = remoteHaloSpecsCpu[haloId].haloX + s + region.x;
        int mpiIndex = s + region.x;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b]*remoteHaloSpecsCpu[haloId].haloWidth;
            buf[valuesPerPointPerBuffer[bufferId]*bufIndex + val] =
                    mpiRecvBuffer[haloId][offset + valuesPerPointPerBuffer[bufferId]*mpiIndex + val];
        }
    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch1D<buf_t>::unpackRecvBufferCwG(size_t haloId, size_t bufferId, buf_t *buf) {
    TauschPackRegion region;
    region.x = 0;
    region.width = remoteHaloSpecsCpuForGpu[haloId].haloWidth;
    unpackRecvBufferCwG(haloId, bufferId, buf, region);
}

template <class buf_t> void Tausch1D<buf_t>::unpackRecvBufferCwG(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    for(int s = 0; s < region.width; ++s) {
        int bufIndex = s+remoteHaloSpecsCpuForGpu[haloId].haloX + region.x;
        int mpiIndex = s + region.x;
        for(int val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            int offset = 0;
            for(int b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b]*remoteHaloSpecsCpuForGpu[haloId].haloWidth;
            buf[valuesPerPointPerBuffer[bufferId]*bufIndex + val]
                    = gpuToCpuRecvBuffer[haloId][offset + valuesPerPointPerBuffer[bufferId]*mpiIndex + val];
        }
    }

}

template <class buf_t> void Tausch1D<buf_t>::unpackRecvBufferGwC(size_t haloId, size_t bufferId, cl::Buffer buf) {

    try {
        auto kernel_unpackRecvBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                (cl_programs, "unpackRecvBuffer");

        size_t bufsize = 0;
        for(int n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsGpu[haloId].haloWidth;

        int globalsize = (bufsize/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        cl::Buffer cl_bufferId(cl_context, &bufferId, (&bufferId)+1, true);

        kernel_unpackRecvBuffer(cl::EnqueueArgs(cl_queue, cl::NDRange(globalsize), cl::NDRange(cl_kernelLocalSize)),
                                cl_remoteHaloSpecsGpu[haloId], cl_valuesPerPointPerBuffer, cl_bufferId,
                                cl_cpuToGpuRecvBuffer[haloId], buf);

    } catch(cl::Error error) {
        std::cerr << "Tausch1D :: unpackRecvBufferCpuToGpu() :: OpenCL exception caught: " << error.what()
                  << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}
#endif


////////////////////////
/// Pack buffer and send data off

template <class buf_t> void Tausch1D<buf_t>::packAndSendCwC(size_t haloId, buf_t *buf, TauschPackRegion region, int mpitag) {
    packSendBufferCwC(haloId, 0, buf, region);
    sendCwC(haloId, mpitag);
}
template <class buf_t> void Tausch1D<buf_t>::packAndSendCwC(size_t haloId, buf_t *buf, int mpitag) {
    packSendBufferCwC(haloId, 0, buf);
    sendCwC(haloId, mpitag);
}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch1D<buf_t>::packAndSendCwG(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag) {
    packSendBufferCwG(haloId, 0, buf, region);
    sendCwG(haloId, msgtag);
}
template <class buf_t> void Tausch1D<buf_t>::packAndSendGwC(size_t haloId, cl::Buffer buf, int msgtag) {
    packSendBufferGwC(haloId, 0, buf);
    sendGwC(haloId, msgtag);
}
#endif


////////////////////////
/// Receive data and unpack

template <class buf_t> void Tausch1D<buf_t>::recvAndUnpackCwC(size_t haloId, buf_t *buf, TauschPackRegion region) {
    recvCwC(haloId);
    unpackRecvBufferCwC(haloId, 0, buf, region);
}
template <class buf_t> void Tausch1D<buf_t>::recvAndUnpackCwC(size_t haloId, buf_t *buf) {
    recvCwC(haloId);
    unpackRecvBufferCwC(haloId, 0, buf);
}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch1D<buf_t>::recvAndUnpackCwG(size_t haloId, buf_t *buf, TauschPackRegion region) {
    recvGwC(haloId);
    unpackRecvBufferCwG(haloId, 0, buf, region);
}
template <class buf_t> void Tausch1D<buf_t>::recvAndUnpackGwC(size_t haloId, cl::Buffer buf) {
    recvCwC(haloId);
    unpackRecvBufferGwC(haloId, 0, buf);
}
#endif



template <class buf_t> TauschPackRegion Tausch1D<buf_t>::createFilledPackRegion(size_t x, size_t width) {
    TauschPackRegion region;
    region.x = x;
    region.width = width;
    return region;
}

template <class buf_t> TauschHaloSpec Tausch1D<buf_t>::createFilledHaloSpec(size_t bufferWidth, size_t haloX, size_t haloWidth, int remoteMpiRank) {
    TauschHaloSpec halo;
    halo.bufferWidth = bufferWidth;
    halo.haloX = haloX;
    halo.haloWidth = haloWidth;
    halo.remoteMpiRank = remoteMpiRank;
    return halo;
}

template <class buf_t> TauschHaloSpec Tausch1D<buf_t>::createFilledHaloSpec(std::vector<size_t> haloIndicesInBuffer) {
    TauschHaloSpec halo;
    halo.haloIndicesInBuffer = haloIndicesInBuffer;
    return halo;
}


///////////////////////////////////////////////////////////////
/// SOME GENERAL PURPOSE OPENCL FUNCTIONS - PART OF PUBLIC API
///////////////////////////////////////////////////////////////

#ifdef TAUSCH_OPENCL

template <class buf_t> void Tausch1D<buf_t>::enableOpenCL(bool blockingSyncCpuGpu, int clLocalWorkgroupSize, bool giveOpenCLDeviceName,
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
        std::cerr << "Tausch1D :: enableOpenCL() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch1D<buf_t>::enableOpenCL(cl::Device cl_defaultDevice, cl::Context cl_context, cl::CommandQueue cl_queue,
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
        std::cerr << "Tausch1D :: enableOpenCL() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch1D<buf_t>::syncCpuAndGpu() {

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

template <class buf_t> int Tausch1D<buf_t>::obtainRemoteId(int msgtag) {
    for(int j = 0; j < remoteHaloNumPartsCpuForGpu; ++j) {
        if(msgtagsCpuToGpu[j].load() == msgtag)
            return j;
    }
    return 0;
}

template <class buf_t> void Tausch1D<buf_t>::setupOpenCL(bool giveOpenCLDeviceName) {

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

template <class buf_t> void Tausch1D<buf_t>::compileKernels() {

    std::string oclstr =
"kernel void packSendBuffer(global const size_t * restrict const haloSpecs,"
"                           global const size_t * restrict const valuesPerPointPerBuffer, global int * restrict const bufferId,"
"                           global double * restrict const haloBuffer, global const double * restrict const buffer) {"

"    const int current = get_global_id(0);"

"    int maxSize = haloSpecs[1];"

"    if(current >= maxSize) return;"

"    int index = current + haloSpecs[0];"

"    for(int val = 0; val < valuesPerPointPerBuffer[*bufferId]; ++val) {"
"        int offset = 0;"
"        for(int b = 0; b < *bufferId; ++b)"
"            offset += valuesPerPointPerBuffer[b]*maxSize;"
"        haloBuffer[offset+ valuesPerPointPerBuffer[*bufferId]*current + val] = buffer[valuesPerPointPerBuffer[*bufferId]*index + val];"
"    }"

"}"

"kernel void unpackRecvBuffer(global const size_t * restrict const haloSpecs,"
"                             global const size_t * restrict const valuesPerPointPerBuffer, global int * restrict const bufferId,"
"                             global const double * restrict const haloBuffer, global double * restrict const buffer) {"

"    const int current = get_global_id(0);"

"    int maxSize = haloSpecs[1];"

"    if(current >= maxSize) return;"

"    int index = current + haloSpecs[0];"

"    for(int val = 0; val < valuesPerPointPerBuffer[*bufferId]; ++val) {"
"        int offset = 0;"
"        for(int b = 0; b < *bufferId; ++b)"
"            offset += valuesPerPointPerBuffer[b]*maxSize;"
"        buffer[valuesPerPointPerBuffer[*bufferId]*index + val] ="
"                haloBuffer[offset + valuesPerPointPerBuffer[*bufferId]*current + val];"
"    }"

"}";

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
                std::cout << "Tausch1D :: compileKernels() :: getBuildInfo :: OpenCL exception caught: " << err.what() << " (" << err.err() << ")"
                          << std::endl;
            }
        }
    } catch(cl::Error error) {
        std::cout << "Tausch1D :: compileKernels() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        if(error.err() == -11) {
            try {
                std::string log = cl_programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_defaultDevice);
                std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl
                          << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** "
                          << std::endl << std::endl;
            } catch(cl::Error err) {
                std::cout << "Tausch1D :: compileKernels() :: getBuildInfo :: OpenCL exception caught: " << err.what() << " (" << err.err() << ")"
                          << std::endl;
            }
        }
    }

}


#endif

template class Tausch1D<char>;
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
