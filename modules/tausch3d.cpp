#include "tausch3d.h"

template <class buf_t> Tausch3D<buf_t>::Tausch3D(MPI_Datatype mpiDataType,
                                                 size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm) {

    MPI_Comm_dup(comm, &TAUSCH_COMM);

    // get MPI info
    MPI_Comm_rank(TAUSCH_COMM, &mpiRank);
    MPI_Comm_size(TAUSCH_COMM, &mpiSize);

    this->numBuffers = numBuffers;

    this->valuesPerPointPerBuffer = new size_t[numBuffers];
#if __cplusplus >= 201103L
    if(valuesPerPointPerBuffer == nullptr) {
#else
    if(valuesPerPointPerBuffer == NULL) {
#endif
        valuesPerPointPerBufferAllOne = true;
        for(size_t i = 0; i < numBuffers; ++i)
            this->valuesPerPointPerBuffer[i] = 1;
    } else {
        valuesPerPointPerBufferAllOne = true;
        for(size_t i = 0; i < numBuffers; ++i) {
            this->valuesPerPointPerBuffer[i] = valuesPerPointPerBuffer[i];
            if(valuesPerPointPerBuffer[i] != 1)
                valuesPerPointPerBufferAllOne = false;
        }
    }

    this->mpiDataType = mpiDataType;

#ifdef TAUSCH_OPENCL
    setupCpuWithGpu = false;
    setupGpuWithCpu = false;
#endif

}

template <class buf_t> Tausch3D<buf_t>::~Tausch3D() {

    for(size_t i = 0; i < mpiSendBufferCpuWithCpu.size(); ++i) {
        if(std::find(alreadyDeletedLocalHaloIds.begin(), alreadyDeletedLocalHaloIds.end(), i) == alreadyDeletedLocalHaloIds.end())
            delete[] mpiSendBufferCpuWithCpu[i];
    }

    for(size_t i = 0; i < mpiRecvBufferCpuWithCpu.size(); ++i) {
        if(std::find(alreadyDeletedRemoteHaloIds.begin(), alreadyDeletedRemoteHaloIds.end(), i) == alreadyDeletedRemoteHaloIds.end())
            delete[] mpiRecvBufferCpuWithCpu[i];
    }

#ifdef TAUSCH_OPENCL

    if(setupCpuWithGpu) {

        for(size_t i = 0; i < localHaloNumPartsCpuWithGpu; ++i)
            delete[] sendBufferCpuWithGpu[i];
        delete[] localHaloSpecsCpuWithGpu;
        delete[] sendBufferCpuWithGpu;
        delete[] msgtagsCpuToGpu;

        for(size_t i = 0; i < remoteHaloNumPartsCpuWithGpu; ++i)
            delete[] recvBufferGpuWithCpu[i];
        delete[] remoteHaloSpecsCpuWithGpu;
        delete[] recvBufferGpuWithCpu;

        delete[] localBufferOffsetCwG;
        delete[] remoteBufferOffsetCwG;
        delete[] localTotalBufferSizeCwG;
        delete[] remoteTotalBufferSizeCwG;

    }

    if(setupGpuWithCpu) {

        for(size_t i = 0; i < localHaloNumPartsGpuWithCpu; ++i)
            delete[] sendBufferGpuWithCpu[i];
        delete[] localHaloSpecsGpuWithCpu;
        delete[] sendBufferGpuWithCpu;
        delete[] msgtagsGpuToCpu;

        for(size_t i = 0; i < remoteHaloNumPartsGpuWithCpu; ++i)
            delete[] recvBufferCpuWithGpu[i];
        delete[] remoteHaloSpecsGpuWithCpu;
        delete[] recvBufferCpuWithGpu;

        delete[] localBufferOffsetGwC;
        delete[] remoteBufferOffsetGwC;
        delete[] localTotalBufferSizeGwC;
        delete[] remoteTotalBufferSizeGwC;

        delete[] cl_localHaloSpecsGpuWithCpu;
        delete[] cl_sendBufferGpuWithCpu;
        delete[] cl_remoteHaloSpecsGpuWithCpu;

    }

#endif

    delete[] valuesPerPointPerBuffer;

}

/////////////////////////////////////////////
/// PUBLIC API FUNCTION
/////////////////////////////////////////////


////////////////////////
/// Set local halo info

template <class buf_t> size_t Tausch3D<buf_t>::addLocalHaloInfoCwC(TauschHaloSpec haloSpec) {

    localHaloSpecsCpuWithCpu.push_back(haloSpec);

    size_t bufsize = 0;
    for(size_t n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*haloSpec.haloWidth*haloSpec.haloHeight*haloSpec.haloDepth;
    mpiSendBufferCpuWithCpu.push_back(new buf_t[bufsize]());

    setupMpiSendCpuWithCpu.push_back(false);

    mpiSendRequestsCpuWithCpu.push_back(MPI_Request());

    // These are computed once as they don't change below
    size_t o = 0;
    for(size_t nb = 0; nb < numBuffers; ++nb) {
        size_t offset = 0;
        for(size_t b = 0; b < nb; ++b)
            offset += valuesPerPointPerBuffer[b] * haloSpec.haloWidth * haloSpec.haloHeight * haloSpec.haloDepth;
        o += offset;
        localBufferOffsetCwC.push_back(o);
    }

    // The buffer sizes also do not change anymore
    size_t s = 0;
    for(size_t nb = 0; nb < numBuffers; ++nb)
        s += valuesPerPointPerBuffer[nb]*haloSpec.haloWidth*haloSpec.haloHeight*haloSpec.haloDepth;
    localTotalBufferSizeCwC.push_back(s);

    return mpiSendBufferCpuWithCpu.size()-1;

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::setLocalHaloInfoCwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumPartsCpuWithGpu = numHaloParts;
    localHaloSpecsCpuWithGpu = new TauschHaloSpec[numHaloParts];
    sendBufferCpuWithGpu = new std::atomic<buf_t>*[numHaloParts];
    msgtagsCpuToGpu = new std::atomic<int>[numHaloParts]{};

    setupCpuWithGpu = true;

    for(size_t i = 0; i < numHaloParts; ++i) {

        localHaloSpecsCpuWithGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsCpuWithGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecsCpuWithGpu[i].bufferDepth = haloSpecs[i].bufferDepth;
        localHaloSpecsCpuWithGpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsCpuWithGpu[i].haloY = haloSpecs[i].haloY;
        localHaloSpecsCpuWithGpu[i].haloZ = haloSpecs[i].haloZ;
        localHaloSpecsCpuWithGpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsCpuWithGpu[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecsCpuWithGpu[i].haloDepth = haloSpecs[i].haloDepth;
        localHaloSpecsCpuWithGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(size_t n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight*haloSpecs[i].haloDepth;
        sendBufferCpuWithGpu[i] = new std::atomic<buf_t>[bufsize]{};

    }

    // These are computed once as they don't change below
    localBufferOffsetCwG = new int[numHaloParts*numBuffers]{};
    for(size_t nb = 0; nb < numBuffers; ++nb) {
        for(size_t nh = 0; nh < numHaloParts; ++nh) {
            int offset = 0;
            for(size_t b = 0; b < nb; ++b)
                offset += valuesPerPointPerBuffer[b] * localHaloSpecsCpuWithGpu[nh].haloWidth * localHaloSpecsCpuWithGpu[nh].haloHeight * localHaloSpecsCpuWithGpu[nh].haloDepth;
            localBufferOffsetCwG[nb*numHaloParts + nh] = offset;
        }
    }

    // The buffer sizes also do not change anymore
    localTotalBufferSizeCwG = new int[numHaloParts]{};
    for(size_t nh = 0; nh < numHaloParts; ++nh)
        for(size_t nb = 0; nb < numBuffers; ++nb)
            localTotalBufferSizeCwG[nh] += valuesPerPointPerBuffer[nb]*localHaloSpecsCpuWithGpu[nh].haloWidth*localHaloSpecsCpuWithGpu[nh].haloHeight*localHaloSpecsCpuWithGpu[nh].haloDepth;

}

template <class buf_t> void Tausch3D<buf_t>::setLocalHaloInfoGwC(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumPartsGpuWithCpu = numHaloParts;
    localHaloSpecsGpuWithCpu = new TauschHaloSpec[numHaloParts];
    sendBufferGpuWithCpu = new std::atomic<buf_t>*[numHaloParts];
    msgtagsGpuToCpu = new std::atomic<int>[numHaloParts]{};

    setupGpuWithCpu = true;

    try {
        cl_sendBufferGpuWithCpu = new cl::Buffer[numHaloParts];
        cl_localHaloSpecsGpuWithCpu = new cl::Buffer[numHaloParts];
    } catch(cl::Error error) {
        std::cerr << "Tausch3D :: setLocalHaloInfoGpu() (1) :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    for(size_t i = 0; i < numHaloParts; ++i) {

        localHaloSpecsGpuWithCpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsGpuWithCpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecsGpuWithCpu[i].bufferDepth = haloSpecs[i].bufferDepth;
        localHaloSpecsGpuWithCpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsGpuWithCpu[i].haloY = haloSpecs[i].haloY;
        localHaloSpecsGpuWithCpu[i].haloZ = haloSpecs[i].haloZ;
        localHaloSpecsGpuWithCpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsGpuWithCpu[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecsGpuWithCpu[i].haloDepth = haloSpecs[i].haloDepth;
        localHaloSpecsGpuWithCpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(size_t n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight*haloSpecs[i].haloDepth;
        sendBufferGpuWithCpu[i] = new std::atomic<buf_t>[bufsize]{};

        size_t tmpHaloSpecs[9] = {haloSpecs[i].haloX, haloSpecs[i].haloY, haloSpecs[i].haloZ,
                                  haloSpecs[i].haloWidth, haloSpecs[i].haloHeight, haloSpecs[i].haloDepth,
                                  haloSpecs[i].bufferWidth, haloSpecs[i].bufferHeight, haloSpecs[i].bufferDepth};

        try {
            cl_sendBufferGpuWithCpu[i] = cl::Buffer(cl_context, CL_MEM_READ_WRITE, bufsize*sizeof(buf_t));
            cl_localHaloSpecsGpuWithCpu[i] = cl::Buffer(cl_context, &tmpHaloSpecs[0], &tmpHaloSpecs[9], true);
        } catch(cl::Error error) {
            std::cerr << "Tausch3D :: setLocalHaloInfoGpu() (2) :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")"
                      << std::endl;
            exit(1);
        }

    }

    // These are computed once as they don't change below
    localBufferOffsetGwC = new int[numHaloParts*numBuffers]{};
    for(size_t nb = 0; nb < numBuffers; ++nb) {
        for(size_t nh = 0; nh < numHaloParts; ++nh) {
            int offset = 0;
            for(size_t b = 0; b < nb; ++b)
                offset += valuesPerPointPerBuffer[b] * localHaloSpecsGpuWithCpu[nh].haloWidth * localHaloSpecsGpuWithCpu[nh].haloHeight * localHaloSpecsGpuWithCpu[nh].haloDepth;
            localBufferOffsetGwC[nb*numHaloParts + nh] = offset;
        }
    }

    // The buffer sizes also do not change anymore
    localTotalBufferSizeGwC = new int[numHaloParts]{};
    for(size_t nh = 0; nh < numHaloParts; ++nh)
        for(size_t nb = 0; nb < numBuffers; ++nb)
            localTotalBufferSizeGwC[nh] += valuesPerPointPerBuffer[nb]*localHaloSpecsGpuWithCpu[nh].haloWidth*localHaloSpecsGpuWithCpu[nh].haloHeight*localHaloSpecsGpuWithCpu[nh].haloDepth;

}
#endif

template <class buf_t> void Tausch3D<buf_t>::delLocalHaloInfoCwC(size_t haloId) {
    delete[] mpiSendBufferCpuWithCpu[haloId];
    mpiSendRequestsCpuWithCpu[haloId] = MPI_Request();
    localBufferOffsetCwC[haloId] = 0;
    localTotalBufferSizeCwC[haloId] = 0;
    alreadyDeletedLocalHaloIds.push_back(haloId);
}


////////////////////////
/// Set remote halo info

template <class buf_t> size_t Tausch3D<buf_t>::addRemoteHaloInfoCwC(TauschHaloSpec haloSpec) {

    remoteHaloSpecsCpuWithCpu.push_back(haloSpec);

    size_t bufsize = 0;
    for(size_t n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*haloSpec.haloWidth*haloSpec.haloHeight*haloSpec.haloDepth;
    mpiRecvBufferCpuWithCpu.push_back(new buf_t[bufsize]());

    setupMpiRecvCpuWithCpu.push_back(false);

    mpiRecvRequestsCpuWithCpu.push_back(MPI_Request());

    // These are computed once as they don't change below
    size_t o = 0;
    for(size_t nb = 0; nb < numBuffers; ++nb) {
        size_t offset = 0;
        for(size_t b = 0; b < nb; ++b)
            offset += valuesPerPointPerBuffer[b] * haloSpec.haloWidth * haloSpec.haloHeight * haloSpec.haloDepth;
        o += offset;
        remoteBufferOffsetCwC.push_back(o);
    }

    // The buffer sizes also do not change anymore
    size_t s = 0;
    for(size_t nb = 0; nb < numBuffers; ++nb)
        s += valuesPerPointPerBuffer[nb]*haloSpec.haloWidth*haloSpec.haloHeight*haloSpec.haloDepth;
    remoteTotalBufferSizeCwC.push_back(s);

    return mpiRecvBufferCpuWithCpu.size()-1;

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::setRemoteHaloInfoCwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsCpuWithGpu = numHaloParts;
    remoteHaloSpecsCpuWithGpu = new TauschHaloSpec[numHaloParts];
    recvBufferCpuWithGpu = new buf_t*[numHaloParts];

    for(size_t i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsCpuWithGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsCpuWithGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecsCpuWithGpu[i].bufferDepth = haloSpecs[i].bufferDepth;
        remoteHaloSpecsCpuWithGpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsCpuWithGpu[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecsCpuWithGpu[i].haloZ = haloSpecs[i].haloZ;
        remoteHaloSpecsCpuWithGpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsCpuWithGpu[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecsCpuWithGpu[i].haloDepth = haloSpecs[i].haloDepth;
        remoteHaloSpecsCpuWithGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(size_t n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight*haloSpecs[i].haloDepth;
        recvBufferCpuWithGpu[i] = new buf_t[bufsize];

    }

    // These are computed once as they don't change below
    remoteBufferOffsetCwG = new size_t[numHaloParts*numBuffers]{};
    for(size_t nb = 0; nb < numBuffers; ++nb) {
        for(size_t nh = 0; nh < numHaloParts; ++nh) {
            size_t offset = 0;
            for(size_t b = 0; b < nb; ++b)
                offset += valuesPerPointPerBuffer[b] * remoteHaloSpecsCpuWithGpu[nh].haloWidth * remoteHaloSpecsCpuWithGpu[nh].haloHeight * remoteHaloSpecsCpuWithGpu[nh].haloDepth;
            remoteBufferOffsetCwG[nb*numHaloParts + nh] = offset;
        }
    }

    // The buffer sizes also do not change anymore
    remoteTotalBufferSizeCwG = new size_t[numHaloParts]{};
    for(size_t nh = 0; nh < numHaloParts; ++nh)
        for(size_t nb = 0; nb < numBuffers; ++nb)
            remoteTotalBufferSizeCwG[nh] += valuesPerPointPerBuffer[nb]*remoteHaloSpecsCpuWithGpu[nh].haloWidth*remoteHaloSpecsCpuWithGpu[nh].haloHeight*remoteHaloSpecsCpuWithGpu[nh].haloDepth;

}

template <class buf_t> void Tausch3D<buf_t>::setRemoteHaloInfoGwC(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsGpuWithCpu = numHaloParts;
    remoteHaloSpecsGpuWithCpu = new TauschHaloSpec[numHaloParts];
    recvBufferGpuWithCpu = new buf_t*[numHaloParts];

    try {
        cl_recvBufferGpuWithCpu = new cl::Buffer[numHaloParts];
        cl_remoteHaloSpecsGpuWithCpu = new cl::Buffer[numHaloParts];
    } catch(cl::Error error) {
        std::cerr << "Tausch3D :: enableOpenCL() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    for(size_t i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsGpuWithCpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsGpuWithCpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecsGpuWithCpu[i].bufferDepth = haloSpecs[i].bufferDepth;
        remoteHaloSpecsGpuWithCpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsGpuWithCpu[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecsGpuWithCpu[i].haloZ = haloSpecs[i].haloZ;
        remoteHaloSpecsGpuWithCpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsGpuWithCpu[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecsGpuWithCpu[i].haloDepth = haloSpecs[i].haloDepth;
        remoteHaloSpecsGpuWithCpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(size_t n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight*haloSpecs[i].haloDepth;
        recvBufferGpuWithCpu[i] = new buf_t[bufsize];

        size_t tmpHaloSpecs[9] = {haloSpecs[i].haloX, haloSpecs[i].haloY, haloSpecs[i].haloZ,
                                  haloSpecs[i].haloWidth, haloSpecs[i].haloHeight, haloSpecs[i].haloDepth,
                                  haloSpecs[i].bufferWidth, haloSpecs[i].bufferHeight, haloSpecs[i].bufferDepth};

        try {
            cl_recvBufferGpuWithCpu[i] = cl::Buffer(cl_context, CL_MEM_READ_WRITE, bufsize*sizeof(double));
            cl_remoteHaloSpecsGpuWithCpu[i] = cl::Buffer(cl_context, &tmpHaloSpecs[0], &tmpHaloSpecs[9], true);
        } catch(cl::Error error) {
            std::cerr << "Tausch3D :: setRemoteHaloInfo() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

    }

    // These are computed once as they don't change below
    remoteBufferOffsetGwC = new int[numHaloParts*numBuffers]{};
    for(size_t nb = 0; nb < numBuffers; ++nb) {
        for(size_t nh = 0; nh < numHaloParts; ++nh) {
            int offset = 0;
            for(size_t b = 0; b < nb; ++b)
                offset += valuesPerPointPerBuffer[b] * remoteHaloSpecsGpuWithCpu[nh].haloWidth * remoteHaloSpecsGpuWithCpu[nh].haloHeight * remoteHaloSpecsGpuWithCpu[nh].haloDepth;
            remoteBufferOffsetGwC[nb*numHaloParts + nh] = offset;
        }
    }

    // The buffer sizes also do not change anymore
    remoteTotalBufferSizeGwC = new int[numHaloParts]{};
    for(size_t nh = 0; nh < numHaloParts; ++nh)
        for(size_t nb = 0; nb < numBuffers; ++nb)
            remoteTotalBufferSizeGwC[nh] += valuesPerPointPerBuffer[nb]*remoteHaloSpecsGpuWithCpu[nh].haloWidth*remoteHaloSpecsGpuWithCpu[nh].haloHeight*remoteHaloSpecsGpuWithCpu[nh].haloDepth;

}
#endif

template <class buf_t> void Tausch3D<buf_t>::delRemoteHaloInfoCwC(size_t haloId) {
    delete[] mpiRecvBufferCpuWithCpu[haloId];
    mpiRecvRequestsCpuWithCpu[haloId] = MPI_Request();
    remoteBufferOffsetCwC[haloId] = 0;
    remoteTotalBufferSizeCwC[haloId] = 0;
    alreadyDeletedRemoteHaloIds.push_back(haloId);
}


////////////////////////
/// Post Receives

template <class buf_t> void Tausch3D<buf_t>::postReceiveCwC(size_t haloId, int msgtag) {

    if(!setupMpiRecvCpuWithCpu[haloId]) {

        if(msgtag == -1) {
            std::cerr << "[Tausch3D] ERROR: MPI_Recv for halo region #" << haloId << " hasn't been posted before, missing mpitag... Abort!"
                      << std::endl;
            exit(1);
        }

        setupMpiRecvCpuWithCpu[haloId] = true;

        MPI_Recv_init(&mpiRecvBufferCpuWithCpu[haloId][0], int(remoteTotalBufferSizeCwC[haloId]), mpiDataType,
                      remoteHaloSpecsCpuWithCpu[haloId].remoteMpiRank, msgtag, TAUSCH_COMM, &mpiRecvRequestsCpuWithCpu[haloId]);

    }

    MPI_Start(&mpiRecvRequestsCpuWithCpu[haloId]);

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::postReceiveCwG(size_t haloId, int msgtag) {
    msgtagsCpuToGpu[haloId].store(msgtag);
}

template <class buf_t> void Tausch3D<buf_t>::postReceiveGwC(size_t haloId, int msgtag) {
    msgtagsGpuToCpu[haloId].store(msgtag);
}
#endif


////////////////////////
/// Post ALL Receives

template <class buf_t> void Tausch3D<buf_t>::postAllReceivesCwC(int *msgtag) {

#if __cplusplus >= 201103L
    if(msgtag == nullptr) {
#else
    if(msgtag == NULL) {
#endif
        msgtag = new int[remoteHaloSpecsCpuWithCpu.size()];
        for(size_t id = 0; id < remoteHaloSpecsCpuWithCpu.size(); ++id)
            msgtag[id] = -1;
    }

    for(size_t id = 0; id < remoteHaloSpecsCpuWithCpu.size(); ++id) {
        if(std::find(alreadyDeletedRemoteHaloIds.begin(), alreadyDeletedRemoteHaloIds.end(), id) == alreadyDeletedRemoteHaloIds.end())
            postReceiveCwC(id, msgtag[id]);
    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::postAllReceivesCwG(int *msgtag) {

#if __cplusplus >= 201103L
    if(msgtag == nullptr) {
#else
    if(msgtag == NULL) {
#endif
        std::cerr << "Tausch3D::postAllReceives :: ERROR :: msgtag cannot be NULL for CpuWithGpu" << std::endl;
        return;
    }

    for(size_t id = 0; id < remoteHaloNumPartsCpuWithGpu; ++id)
        postReceiveCwG(id, msgtag[id]);

}

template <class buf_t> void Tausch3D<buf_t>::postAllReceivesGwC(int *msgtag) {

#if __cplusplus >= 201103L
    if(msgtag == nullptr) {
#else
    if(msgtag == NULL) {
#endif
        std::cerr << "Tausch3D::postAllReceives :: ERROR :: msgtag cannot be NULL for GpuWithGpu" << std::endl;
        return;
    }

    for(size_t id = 0; id < remoteHaloNumPartsGpuWithCpu; ++id)
        postReceiveGwC(id, msgtag[id]);

}
#endif


////////////////////////
/// Pack send buffer

template <class buf_t> void Tausch3D<buf_t>::packSendBufferCwC(size_t haloId, size_t bufferId, buf_t *buf) {
    TauschPackRegion region;
    region.x = 0;
    region.y = 0;
    region.z = 0;
    region.width = localHaloSpecsCpuWithCpu[haloId].haloWidth;
    region.height = localHaloSpecsCpuWithCpu[haloId].haloHeight;
    region.depth = localHaloSpecsCpuWithCpu[haloId].haloDepth;
    packSendBufferCwC(haloId, bufferId, buf, region);
}

template <class buf_t> void Tausch3D<buf_t>::packSendBufferCwC(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    size_t bufIndexBase = (region.z + localHaloSpecsCpuWithCpu[haloId].haloZ)*localHaloSpecsCpuWithCpu[haloId].bufferWidth*localHaloSpecsCpuWithCpu[haloId].bufferHeight +
                             (region.y + localHaloSpecsCpuWithCpu[haloId].haloY)*localHaloSpecsCpuWithCpu[haloId].bufferWidth +
                              region.x + localHaloSpecsCpuWithCpu[haloId].haloX;
    size_t mpiIndexBase = region.z*localHaloSpecsCpuWithCpu[haloId].haloWidth*localHaloSpecsCpuWithCpu[haloId].haloHeight +
                             region.y*localHaloSpecsCpuWithCpu[haloId].haloWidth +
                             region.x;

    if(valuesPerPointPerBufferAllOne) {

        mpiIndexBase += localBufferOffsetCwC[numBuffers*haloId + bufferId];

        for(size_t d = 0; d < region.depth; ++d) {

            size_t buff_addZOffset = bufIndexBase + d*localHaloSpecsCpuWithCpu[haloId].bufferWidth*localHaloSpecsCpuWithCpu[haloId].bufferHeight;
            size_t halo_addZOffset = mpiIndexBase + d*localHaloSpecsCpuWithCpu[haloId].haloWidth*localHaloSpecsCpuWithCpu[haloId].haloHeight;

            for(size_t h = 0; h < region.height; ++h) {

                size_t buff_addZYOffset = buff_addZOffset + h*localHaloSpecsCpuWithCpu[haloId].bufferWidth;
                size_t halo_addZYOffset = halo_addZOffset + h*localHaloSpecsCpuWithCpu[haloId].haloWidth;

                for(size_t w = 0; w < region.width; ++w) {

                    mpiSendBufferCpuWithCpu[haloId][halo_addZYOffset+w] = buf[buff_addZYOffset + w];

                }

            }

        }

    } else {

        for(size_t d = 0; d < region.depth; ++d) {

            size_t buff_addZOffset = bufIndexBase + d * localHaloSpecsCpuWithCpu[haloId].bufferWidth*localHaloSpecsCpuWithCpu[haloId].bufferHeight;
            size_t halo_addZOffset = mpiIndexBase + d * localHaloSpecsCpuWithCpu[haloId].haloWidth*localHaloSpecsCpuWithCpu[haloId].haloHeight;

            for(size_t h = 0; h < region.height; ++h) {

                size_t buff_addZYOffset = buff_addZOffset + h*localHaloSpecsCpuWithCpu[haloId].bufferWidth;
                size_t halo_addZYOffset = halo_addZOffset + h*localHaloSpecsCpuWithCpu[haloId].haloWidth;

                for(size_t w = 0; w < region.width; ++w) {

                    size_t bufIndex = valuesPerPointPerBuffer[bufferId] * (buff_addZYOffset + w);
                    size_t mpiIndex = localBufferOffsetCwC[numBuffers*haloId + bufferId] + valuesPerPointPerBuffer[bufferId] * (halo_addZYOffset + w);

                    for(size_t val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val)
                        mpiSendBufferCpuWithCpu[haloId][mpiIndex + val] = buf[bufIndex + val];

                }

            }

        }

    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::packSendBufferCwG(size_t haloId, size_t bufferId, buf_t *buf) {
    TauschPackRegion region;
    region.x = 0;
    region.y = 0;
    region.z = 0;
    region.width = localHaloSpecsCpuWithGpu[haloId].haloWidth;
    region.height = localHaloSpecsCpuWithGpu[haloId].haloHeight;
    region.depth = localHaloSpecsCpuWithGpu[haloId].haloDepth;
    packSendBufferCwG(haloId, bufferId, buf, region);
}

template <class buf_t> void Tausch3D<buf_t>::packSendBufferCwG(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    size_t bufIndexBase = (region.z + localHaloSpecsCpuWithCpu[haloId].haloZ)*localHaloSpecsCpuWithCpu[haloId].bufferWidth*localHaloSpecsCpuWithCpu[haloId].bufferHeight +
                       (region.y + localHaloSpecsCpuWithCpu[haloId].haloY)*localHaloSpecsCpuWithCpu[haloId].bufferWidth +
                        region.x + localHaloSpecsCpuWithCpu[haloId].haloX;
    size_t mpiIndexBase = region.z*localHaloSpecsCpuWithCpu[haloId].haloWidth*localHaloSpecsCpuWithCpu[haloId].haloHeight +
                       region.y*localHaloSpecsCpuWithCpu[haloId].haloWidth +
                       region.x;

    if(valuesPerPointPerBufferAllOne) {

        mpiIndexBase += localBufferOffsetCwC[numBuffers*haloId + bufferId];

        for(size_t d = 0; d < region.depth; ++d) {

            size_t dbwh = bufIndexBase + d*localHaloSpecsCpuWithCpu[haloId].bufferWidth*localHaloSpecsCpuWithCpu[haloId].bufferHeight;
            size_t dhwh = mpiIndexBase + d*localHaloSpecsCpuWithCpu[haloId].haloWidth*localHaloSpecsCpuWithCpu[haloId].haloHeight;

            for(size_t h = 0; h < region.height; ++h) {

                size_t hbw = dbwh + h*localHaloSpecsCpuWithCpu[haloId].bufferWidth;
                size_t hhw = dhwh + h*localHaloSpecsCpuWithCpu[haloId].haloWidth;

                for(size_t w = 0; w < region.width; ++w)
                    mpiSendBufferCpuWithCpu[haloId][hhw + w] = buf[hbw + w];

            }

        }

    } else {

        for(size_t d = 0; d < region.depth; ++d) {

            size_t dbwh = d*localHaloSpecsCpuWithCpu[haloId].bufferWidth*localHaloSpecsCpuWithCpu[haloId].bufferHeight;
            size_t dhwh = d*localHaloSpecsCpuWithCpu[haloId].haloWidth*localHaloSpecsCpuWithCpu[haloId].haloHeight;

            for(size_t h = 0; h < region.height; ++h) {

                size_t hbw = dbwh + h*localHaloSpecsCpuWithCpu[haloId].bufferWidth;
                size_t hhw = dhwh + h*localHaloSpecsCpuWithCpu[haloId].haloWidth;

                for(size_t w = 0; w < region.width; ++w) {

                    size_t bufIndex = bufIndexBase + hbw + w;
                    size_t mpiIndex = localBufferOffsetCwC[numBuffers*haloId + bufferId] + valuesPerPointPerBuffer[bufferId]*(mpiIndexBase + hhw + w);

                    for(size_t val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val)
                        mpiSendBufferCpuWithCpu[haloId][mpiIndex + val] = buf[valuesPerPointPerBuffer[bufferId]*bufIndex + val];

                }

            }

        }

    }

}

template <class buf_t> void Tausch3D<buf_t>::packSendBufferGwC(size_t haloId, size_t bufferId, cl::Buffer buf) {

    try {

        auto kernel_packNextSendBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                    (cl_programs, "packSendBuffer");

        size_t bufsize = 0;
        for(size_t n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsGpuWithCpu[haloId].haloWidth
                       *localHaloSpecsGpuWithCpu[haloId].haloHeight*localHaloSpecsGpuWithCpu[haloId].haloDepth;

        size_t globalsize = (bufsize/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        cl::Buffer cl_bufferId(cl_context, &bufferId, (&bufferId)+1, true);

        kernel_packNextSendBuffer(cl::EnqueueArgs(cl_queue, cl::NDRange(globalsize), cl::NDRange(cl_kernelLocalSize)),
                                  cl_localHaloSpecsGpuWithCpu[haloId], cl_valuesPerPointPerBuffer,
                                  cl_bufferId, cl_sendBufferGpuWithCpu[haloId], buf);

    } catch(cl::Error error) {
        std::cerr << "Tausch3D :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}
#endif


////////////////////////
/// Send data off

template <class buf_t> void Tausch3D<buf_t>::sendCwC(size_t haloId, int msgtag) {


    if(!setupMpiSendCpuWithCpu[haloId]) {

        if(msgtag == -1) {
            std::cerr << "[Tausch3D] ERROR: MPI_Send for halo region #" << haloId << " hasn't been posted before, missing mpitag... Abort!"
                      << std::endl;
            exit(1);
        }

        setupMpiSendCpuWithCpu[haloId] = true;

        MPI_Send_init(&mpiSendBufferCpuWithCpu[haloId][0], int(localTotalBufferSizeCwC[haloId]),
                  mpiDataType, localHaloSpecsCpuWithCpu[haloId].remoteMpiRank, msgtag, TAUSCH_COMM, &mpiSendRequestsCpuWithCpu[haloId]);

    } else
        MPI_Wait(&mpiSendRequestsCpuWithCpu[haloId], MPI_STATUS_IGNORE);

    MPI_Start(&mpiSendRequestsCpuWithCpu[haloId]);

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::sendCwG(size_t haloId, int msgtag) {
    msgtagsCpuToGpu[haloId].store(msgtag);
    syncCpuAndGpu();
}

template <class buf_t> void Tausch3D<buf_t>::sendGwC(size_t haloId, int msgtag) {

    msgtagsGpuToCpu[haloId].store(msgtag);

    try {

        size_t bufsize = 0;
        for(size_t n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*localHaloSpecsGpuWithCpu[haloId].haloWidth
                       *localHaloSpecsGpuWithCpu[haloId].haloHeight*localHaloSpecsGpuWithCpu[haloId].haloDepth;

        buf_t *tmp = new buf_t[bufsize];
        cl::copy(cl_queue, cl_sendBufferGpuWithCpu[haloId], &tmp[0], &tmp[bufsize]);
        for(size_t i = 0; i < bufsize; ++i)
            sendBufferGpuWithCpu[haloId][i].store(tmp[i]);

    } catch(cl::Error error) {
        std::cerr << "Tausch3D :: sendGpuToCpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    syncCpuAndGpu();

}
#endif


////////////////////////
/// Receive data

template <class buf_t> void Tausch3D<buf_t>::recvCwC(size_t haloId) {
    MPI_Wait(&mpiRecvRequestsCpuWithCpu[haloId], MPI_STATUS_IGNORE);
}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::recvGwC(size_t haloId) {

    syncCpuAndGpu();

    size_t remoteid = obtainRemoteId(msgtagsCpuToGpu[haloId]);

    size_t bufsize = 0;
    for(size_t n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsGpuWithCpu[haloId].haloWidth
                   *remoteHaloSpecsGpuWithCpu[haloId].haloHeight*remoteHaloSpecsGpuWithCpu[haloId].haloDepth;

    for(size_t j = 0; j < bufsize; ++j)
        recvBufferGpuWithCpu[haloId][j] = sendBufferCpuWithGpu[remoteid][j].load();

    try {
        cl_recvBufferGpuWithCpu[haloId] = cl::Buffer(cl_context, &recvBufferGpuWithCpu[haloId][0], &recvBufferGpuWithCpu[haloId][bufsize], false);
    } catch(cl::Error error) {
        std::cerr << "Tausch3D :: recvCpuToGpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch3D<buf_t>::recvCwG(size_t haloId) {

    syncCpuAndGpu();

    size_t remoteid = obtainRemoteId(msgtagsGpuToCpu[haloId]);

    size_t bufsize = 0;
    for(size_t n = 0; n < numBuffers; ++n)
        bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsCpuWithGpu[haloId].haloWidth
                   *remoteHaloSpecsCpuWithGpu[haloId].haloHeight*remoteHaloSpecsCpuWithGpu[haloId].haloDepth;
    for(size_t i = 0; i < bufsize; ++i)
        recvBufferCpuWithGpu[haloId][i] = sendBufferGpuWithCpu[remoteid][i].load();

}
#endif


////////////////////////
/// Unpack received data

template <class buf_t> void Tausch3D<buf_t>::unpackRecvBufferCwC(size_t haloId, size_t bufferId, buf_t *buf) {
    TauschPackRegion region;
    region.x = 0;
    region.y = 0;
    region.z = 0;
    region.width = remoteHaloSpecsCpuWithCpu[haloId].haloWidth;
    region.height = remoteHaloSpecsCpuWithCpu[haloId].haloHeight;
    region.depth = remoteHaloSpecsCpuWithCpu[haloId].haloDepth;
    unpackRecvBufferCwC(haloId, bufferId, buf, region);
}

template <class buf_t> void Tausch3D<buf_t>::unpackRecvBufferCwC(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    size_t bufIndexBase = (region.z + remoteHaloSpecsCpuWithCpu[haloId].haloZ)*remoteHaloSpecsCpuWithCpu[haloId].bufferWidth*remoteHaloSpecsCpuWithCpu[haloId].bufferHeight +
                       (region.y + remoteHaloSpecsCpuWithCpu[haloId].haloY)*remoteHaloSpecsCpuWithCpu[haloId].bufferWidth +
                        remoteHaloSpecsCpuWithCpu[haloId].haloX + region.x;
    size_t mpiIndexBase = region.z*remoteHaloSpecsCpuWithCpu[haloId].haloWidth*remoteHaloSpecsCpuWithCpu[haloId].haloHeight +
                       region.y*remoteHaloSpecsCpuWithCpu[haloId].haloWidth +
                       region.x;

    if(valuesPerPointPerBufferAllOne) {

        mpiIndexBase += remoteBufferOffsetCwC[numBuffers*haloId + bufferId];

        for(size_t d = 0; d < region.depth; ++d) {

            size_t dbwh = bufIndexBase + d*remoteHaloSpecsCpuWithCpu[haloId].bufferWidth*remoteHaloSpecsCpuWithCpu[haloId].bufferHeight;
            size_t dhwh = mpiIndexBase + d*remoteHaloSpecsCpuWithCpu[haloId].haloWidth*remoteHaloSpecsCpuWithCpu[haloId].haloHeight;

            for(size_t h = 0; h < region.height; ++h) {

                size_t hbw = dbwh + h*remoteHaloSpecsCpuWithCpu[haloId].bufferWidth;
                size_t hhw = dhwh + h*remoteHaloSpecsCpuWithCpu[haloId].haloWidth;

                for(size_t w = 0; w < region.width; ++w)
                    buf[hbw + w] = mpiRecvBufferCpuWithCpu[haloId][hhw + w];

            }

        }

    } else {

        for(size_t d = 0; d < region.depth; ++d) {

            size_t dbwh = d*remoteHaloSpecsCpuWithCpu[haloId].bufferWidth*remoteHaloSpecsCpuWithCpu[haloId].bufferHeight;
            size_t dhwh = d*remoteHaloSpecsCpuWithCpu[haloId].haloWidth*remoteHaloSpecsCpuWithCpu[haloId].haloHeight;

            for(size_t h = 0; h < region.height; ++h) {

                size_t hbw = dbwh + h*remoteHaloSpecsCpuWithCpu[haloId].bufferWidth;
                size_t hhw = dhwh + h*remoteHaloSpecsCpuWithCpu[haloId].haloWidth;

                for(size_t w = 0; w < region.width; ++w) {

                    size_t bufIndex = valuesPerPointPerBuffer[bufferId]* ( hbw + w );
                    size_t fullerOffset = remoteBufferOffsetCwC[numBuffers*haloId + bufferId] + valuesPerPointPerBuffer[bufferId]* ( hhw + w );

                    for(size_t val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val)
                        buf[bufIndex + val] = mpiRecvBufferCpuWithCpu[haloId][fullerOffset + val];

                }

            }

        }

    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::unpackRecvBufferCwG(size_t haloId, size_t bufferId, buf_t *buf) {
    TauschPackRegion region;
    region.x = 0;
    region.y = 0;
    region.z = 0;
    region.width = remoteHaloSpecsCpuWithGpu[haloId].haloWidth;
    region.height = remoteHaloSpecsCpuWithGpu[haloId].haloHeight;
    region.depth = remoteHaloSpecsCpuWithGpu[haloId].haloDepth;
    unpackRecvBufferCwG(haloId, bufferId, buf, region);
}

template <class buf_t> void Tausch3D<buf_t>::unpackRecvBufferCwG(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    for(size_t s = 0; s < region.width * region.height * region.depth; ++s) {
        size_t bufIndex = (s/(region.width*region.height) + remoteHaloSpecsCpuWithGpu[haloId].haloZ)
                          * remoteHaloSpecsCpuWithGpu[haloId].bufferWidth * remoteHaloSpecsCpuWithGpu[haloId].bufferHeight +
                          ((s%(region.width*region.height))/remoteHaloSpecsCpuWithGpu[haloId].haloWidth + remoteHaloSpecsCpuWithGpu[haloId].haloY)
                          * remoteHaloSpecsCpuWithGpu[haloId].bufferWidth +
                          s%region.width + remoteHaloSpecsCpuWithGpu[haloId].haloX;
        size_t mpiIndex = (s/(region.width*region.height) + region.z)
                          *(remoteHaloSpecsCpuWithGpu[haloId].haloWidth*remoteHaloSpecsCpuWithGpu[haloId].haloHeight) +
                          ((s%(region.width*region.height))/remoteHaloSpecsCpuWithGpu[haloId].haloWidth + region.y)
                          * remoteHaloSpecsCpuWithGpu[haloId].haloWidth +
                          s%region.width + region.x;
        for(size_t val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val) {
            size_t offset = 0;
            for(size_t b = 0; b < bufferId; ++b)
                offset += valuesPerPointPerBuffer[b] * remoteHaloSpecsCpuWithGpu[haloId].haloWidth * remoteHaloSpecsCpuWithGpu[haloId].haloHeight
                          * remoteHaloSpecsCpuWithGpu[haloId].haloDepth;
            buf[valuesPerPointPerBuffer[bufferId]*bufIndex + val]
                    = recvBufferCpuWithGpu[haloId][offset + valuesPerPointPerBuffer[bufferId]*mpiIndex + val];
        }
    }

}

template <class buf_t> void Tausch3D<buf_t>::unpackRecvBufferGwC(size_t haloId, size_t bufferId, cl::Buffer buf) {

    try {
        auto kernel_unpackRecvBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                (cl_programs, "unpackRecvBuffer");

        size_t bufsize = 0;
        for(size_t n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*remoteHaloSpecsGpuWithCpu[haloId].haloWidth
                       *remoteHaloSpecsGpuWithCpu[haloId].haloHeight*remoteHaloSpecsGpuWithCpu[haloId].haloDepth;

        size_t globalsize = (bufsize/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        cl::Buffer cl_bufferId(cl_context, &bufferId, (&bufferId)+1, true);

        kernel_unpackRecvBuffer(cl::EnqueueArgs(cl_queue, cl::NDRange(globalsize), cl::NDRange(cl_kernelLocalSize)),
                                cl_remoteHaloSpecsGpuWithCpu[haloId], cl_valuesPerPointPerBuffer, cl_bufferId,
                                cl_recvBufferGpuWithCpu[haloId], buf);

    } catch(cl::Error error) {
        std::cerr << "Tausch3D :: unpackRecvBufferCpuToGpu() :: OpenCL exception caught: " << error.what()
                  << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}
#endif


////////////////////////
/// Pack buffer and send data off

template <class buf_t> void Tausch3D<buf_t>::packAndSendCwC(size_t haloId, buf_t *buf, TauschPackRegion region, int mpitag) {
    packSendBufferCwC(haloId, 0, buf, region);
    sendCwC(haloId, mpitag);
}
template <class buf_t> void Tausch3D<buf_t>::packAndSendCwC(size_t haloId, buf_t *buf, int mpitag) {
    packSendBufferCwC(haloId, 0, buf);
    sendCwC(haloId, mpitag);
}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::packAndSendCwG(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag) {
    packSendBufferCwG(haloId, 0, buf, region);
    sendCwG(haloId, msgtag);
}
template <class buf_t> void Tausch3D<buf_t>::packAndSendGwC(size_t haloId, cl::Buffer buf, int msgtag) {
    packSendBufferGwC(haloId, 0, buf);
    sendGwC(haloId, msgtag);
}
#endif


////////////////////////
/// Receive data and unpack

template <class buf_t> void Tausch3D<buf_t>::recvAndUnpackCwC(size_t haloId, buf_t *buf, TauschPackRegion region) {
    recvCwC(haloId);
    unpackRecvBufferCwC(haloId, 0, buf, region);
}
template <class buf_t> void Tausch3D<buf_t>::recvAndUnpackCwC(size_t haloId, buf_t *buf) {
    recvCwC(haloId);
    unpackRecvBufferCwC(haloId, 0, buf);
}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch3D<buf_t>::recvAndUnpackCwG(size_t haloId, buf_t *buf, TauschPackRegion region) {
    recvCwG(haloId);
    unpackRecvBufferCwG(haloId, 0, buf, region);
}
template <class buf_t> void Tausch3D<buf_t>::recvAndUnpackGwC(size_t haloId, cl::Buffer buf) {
    recvGwC(haloId);
    unpackRecvBufferGwC(haloId, 0, buf);
}
#endif


template <class buf_t> TauschPackRegion Tausch3D<buf_t>::createFilledPackRegion(size_t x, size_t y, size_t z,
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

template <class buf_t> TauschHaloSpec Tausch3D<buf_t>::createFilledHaloSpec(size_t bufferWidth, size_t bufferHeight, size_t bufferDepth,
                                                                            size_t haloX, size_t haloY, size_t haloZ,
                                                                            size_t haloWidth, size_t haloHeight, size_t haloDepth, int remoteMpiRank) {
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



///////////////////////////////////////////////////////////////
/// SOME GENERAL PURPOSE OPENCL FUNCTIONS - PART OF PUBLIC API
///////////////////////////////////////////////////////////////

#ifdef TAUSCH_OPENCL

template <class buf_t> void Tausch3D<buf_t>::enableOpenCL(bool blockingSyncCpuGpu, size_t clLocalWorkgroupSize, bool giveOpenCLDeviceName,
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
        std::cerr << "Tausch3D :: enableOpenCL() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch3D<buf_t>::enableOpenCL(cl::Device cl_defaultDevice, cl::Context cl_context, cl::CommandQueue cl_queue,
                                                          bool blockingSyncCpuGpu, size_t clLocalWorkgroupSize, bool showOpenCLBuildLog) {

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

template <class buf_t> size_t Tausch3D<buf_t>::obtainRemoteId(int msgtag) {
    for(size_t j = 0; j < remoteHaloNumPartsCpuWithGpu; ++j) {
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
        size_t platform_length = all_platforms.size();

        // We need at most mpiSize many devices
        size_t *platform_num = new size_t[mpiSize]{};
        size_t *device_num = new size_t[mpiSize]{};

        // Counter so that we know when to stop
        int num = 0;

        // Loop over platforms
        for(size_t i = 0; i < platform_length; ++i) {
            // Get devices on platform
            std::vector<cl::Device> all_devices;
            all_platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
            size_t device_length = all_devices.size();
            // Loop over platforms
            for(size_t j = 0; j < device_length; ++j) {
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

    std::string oclstr =
"kernel void packSendBuffer(global const size_t * restrict const haloSpecs,"
"                           global const size_t * restrict const valuesPerPointPerBuffer, global int * restrict const bufferId,"
"                           global double * restrict const haloBuffer, global const double * restrict const buffer) {"

"    const int current = get_global_id(0);"

"    int maxSize = haloSpecs[3]*haloSpecs[4]*haloSpecs[5];"

"    if(current >= maxSize) return;"

"    int index = (current/(haloSpecs[3]*haloSpecs[4]) + haloSpecs[2])*haloSpecs[6]*haloSpecs[7] +"
"                ((current%(haloSpecs[3]*haloSpecs[4]))/haloSpecs[3] + haloSpecs[1]) * haloSpecs[6] +"
"                current%haloSpecs[3] + haloSpecs[0];"

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

"    int maxSize = haloSpecs[3]*haloSpecs[4]*haloSpecs[5];"

"    if(current >= maxSize) return;"

"    int index = (current/(haloSpecs[3]*haloSpecs[4]) + haloSpecs[2])*haloSpecs[6]*haloSpecs[7] +"
"                ((current%(haloSpecs[3]*haloSpecs[4]))/haloSpecs[3] + haloSpecs[1]) * haloSpecs[6] +"
"                current%haloSpecs[3] + haloSpecs[0];"

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
                std::cout << "Tausch3D :: compileKernels() :: getBuildInfo :: OpenCL exception caught: " << err.what() << " (" << err.err() << ")"
                          << std::endl;
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
                std::cout << "Tausch3D :: compileKernels() :: getBuildInfo :: OpenCL exception caught: " << err.what() << " (" << err.err() << ")"
                          << std::endl;
            }
        }
    }

}


#endif


template class Tausch3D<char>;
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
