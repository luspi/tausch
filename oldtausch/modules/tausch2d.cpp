#include "tausch2d.h"

template <class buf_t> Tausch2D<buf_t>::Tausch2D(MPI_Datatype mpiDataType,
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
    setupGpuWithGpu = false;
#endif

}

template <class buf_t> Tausch2D<buf_t>::~Tausch2D() {

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
            delete[] recvBufferCpuWithGpu[i];
        delete[] remoteHaloSpecsCpuWithGpu;
        delete[] recvBufferCpuWithGpu;

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
            delete[] recvBufferGpuWithCpu[i];
        delete[] remoteHaloSpecsGpuWithCpu;
        delete[] recvBufferGpuWithCpu;

        delete[] localBufferOffsetGwC;
        delete[] remoteBufferOffsetGwC;
        delete[] localTotalBufferSizeGwC;
        delete[] remoteTotalBufferSizeGwC;

    }

    if(setupGpuWithGpu) {

        for(size_t i = 0; i < localHaloNumPartsGpuWithGpu; ++i)
            delete[] mpiSendBufferGpuWithGpu[i];
        delete[] localHaloSpecsGpuWithGpu;
        delete[] mpiSendBufferGpuWithGpu;
        delete[] mpiSendRequestsGpuWithGpu;
        delete[] setupMpiSendGpuWithGpu;
        delete[] cl_sendBufferGpuWithGpu;
        delete[] cl_localHaloSpecsGpuWithGpu;

        for(size_t i = 0; i < remoteHaloNumPartsGpuWithGpu; ++i)
            delete[] mpiRecvBufferGpuWithGpu[i];
        delete[] remoteHaloSpecsGpuWithGpu;
        delete[] mpiRecvBufferGpuWithGpu;
        delete[] mpiRecvRequestsGpuWithGpu;
        delete[] setupMpiRecvGpuWithGpu;
        delete[] cl_recvBufferGpuWithGpu;
        delete[] cl_remoteHaloSpecsGpuWithGpu;

        delete[] localBufferOffsetGwG;
        delete[] remoteBufferOffsetGwG;
        delete[] localTotalBufferSizeGwG;
        delete[] remoteTotalBufferSizeGwG;

    }

#endif

    delete[] valuesPerPointPerBuffer;
}

/////////////////////////////////////////////
/// PUBLIC API FUNCTION
/////////////////////////////////////////////


////////////////////////
/// Set local halo info
///

template <class buf_t> size_t Tausch2D<buf_t>::addLocalHaloInfoCwC(TauschHaloSpec haloSpec) {

    localHaloSpecsCpuWithCpu.push_back(haloSpec);

    size_t haloSize = haloSpec.haloWidth*haloSpec.haloHeight;
    if(haloSpec.haloIndicesInBuffer.size() > 0)
        haloSize = haloSpec.haloIndicesInBuffer.size();

    size_t bufsize = 0;
    size_t offset = 0;

    for(size_t n = 0; n < numBuffers; ++n) {
        bufsize += valuesPerPointPerBuffer[n]*haloSize;
        offset += valuesPerPointPerBuffer[n-1] * haloSize;
        localBufferOffsetCwC.push_back(offset);
    }

    localTotalBufferSizeCwC.push_back(bufsize);
    mpiSendBufferCpuWithCpu.push_back(new buf_t[bufsize]());
    mpiSendRequestsCpuWithCpu.push_back(MPI_Request());

    setupMpiSendCpuWithCpu.push_back(false);

    return mpiSendBufferCpuWithCpu.size()-1;

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::setLocalHaloInfoCwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    localHaloNumPartsCpuWithGpu = numHaloParts;
    localHaloSpecsCpuWithGpu = new TauschHaloSpec[numHaloParts];
    sendBufferCpuWithGpu = new std::atomic<buf_t>*[numHaloParts];
    msgtagsCpuToGpu = new std::atomic<int>[numHaloParts]{};

    setupCpuWithGpu = true;

    for(size_t i = 0; i < numHaloParts; ++i) {

        localHaloSpecsCpuWithGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsCpuWithGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecsCpuWithGpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsCpuWithGpu[i].haloY = haloSpecs[i].haloY;
        localHaloSpecsCpuWithGpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsCpuWithGpu[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecsCpuWithGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(size_t n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        sendBufferCpuWithGpu[i] = new std::atomic<buf_t>[bufsize]{};

    }

    // These are computed once as they don't change below
    localBufferOffsetCwG = new size_t[numHaloParts*numBuffers]{};
    for(size_t nb = 0; nb < numBuffers; ++nb) {
        for(size_t nh = 0; nh < numHaloParts; ++nh) {
            size_t offset = 0;
            for(size_t b = 0; b < nb; ++b)
                offset += valuesPerPointPerBuffer[b] * localHaloSpecsCpuWithGpu[nh].haloWidth * localHaloSpecsCpuWithGpu[nh].haloHeight;
            localBufferOffsetCwG[nb*numHaloParts + nh] = offset;
        }
    }

    // The buffer sizes also do not change anymore
    localTotalBufferSizeCwG = new int[numHaloParts]{};
    for(size_t nh = 0; nh < numHaloParts; ++nh)
        for(size_t nb = 0; nb < numBuffers; ++nb)
            localTotalBufferSizeCwG[nh] += valuesPerPointPerBuffer[nb]*localHaloSpecsCpuWithGpu[nh].haloWidth*localHaloSpecsCpuWithGpu[nh].haloHeight;

}

template <class buf_t> void Tausch2D<buf_t>::setLocalHaloInfoGwC(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

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

    for(size_t i = 0; i < numHaloParts; ++i) {

        localHaloSpecsGpuWithCpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsGpuWithCpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecsGpuWithCpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsGpuWithCpu[i].haloY = haloSpecs[i].haloY;
        localHaloSpecsGpuWithCpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsGpuWithCpu[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecsGpuWithCpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(size_t n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        sendBufferGpuWithCpu[i] = new std::atomic<buf_t>[bufsize]{};

        size_t tmpHaloSpecs[6] = {haloSpecs[i].haloX, haloSpecs[i].haloY,
                                  haloSpecs[i].haloWidth, haloSpecs[i].haloHeight,
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

    // These are computed once as they don't change below
    localBufferOffsetGwC = new int[numHaloParts*numBuffers]{};
    for(size_t nb = 0; nb < numBuffers; ++nb) {
        for(size_t nh = 0; nh < numHaloParts; ++nh) {
            int offset = 0;
            for(size_t b = 0; b < nb; ++b)
                offset += valuesPerPointPerBuffer[b] * localHaloSpecsGpuWithCpu[nh].haloWidth * localHaloSpecsGpuWithCpu[nh].haloHeight;
            localBufferOffsetGwC[nb*numHaloParts + nh] = offset;
        }
    }

    // The buffer sizes also do not change anymore
    localTotalBufferSizeGwC = new size_t[numHaloParts]{};
    for(size_t nh = 0; nh < numHaloParts; ++nh)
        for(size_t nb = 0; nb < numBuffers; ++nb)
            localTotalBufferSizeGwC[nh] += valuesPerPointPerBuffer[nb]*localHaloSpecsGpuWithCpu[nh].haloWidth*localHaloSpecsGpuWithCpu[nh].haloHeight;

}

template <class buf_t> void Tausch2D<buf_t>::setLocalHaloInfoGwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

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

    for(size_t i = 0; i < numHaloParts; ++i) {

        localHaloSpecsGpuWithGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        localHaloSpecsGpuWithGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        localHaloSpecsGpuWithGpu[i].haloX = haloSpecs[i].haloX;
        localHaloSpecsGpuWithGpu[i].haloY = haloSpecs[i].haloY;
        localHaloSpecsGpuWithGpu[i].haloWidth = haloSpecs[i].haloWidth;
        localHaloSpecsGpuWithGpu[i].haloHeight = haloSpecs[i].haloHeight;
        localHaloSpecsGpuWithGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(size_t n = 0; n < numBuffers; ++n)
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

    // These are computed once as they don't change below
    localBufferOffsetGwG = new int[numHaloParts*numBuffers]{};
    for(size_t nb = 0; nb < numBuffers; ++nb) {
        for(size_t nh = 0; nh < numHaloParts; ++nh) {
            int offset = 0;
            for(size_t b = 0; b < nb; ++b)
                offset += valuesPerPointPerBuffer[b] * localHaloSpecsGpuWithGpu[nh].haloWidth * localHaloSpecsGpuWithGpu[nh].haloHeight;
            localBufferOffsetGwG[nb*numHaloParts + nh] = offset;
        }
    }

    // The buffer sizes also do not change anymore
    localTotalBufferSizeGwG = new size_t[numHaloParts]{};
    for(size_t nh = 0; nh < numHaloParts; ++nh)
        for(size_t nb = 0; nb < numBuffers; ++nb)
            localTotalBufferSizeGwG[nh] += valuesPerPointPerBuffer[nb]*localHaloSpecsGpuWithGpu[nh].haloWidth*localHaloSpecsGpuWithGpu[nh].haloHeight;

}
#endif


////////////////////////
/// Destroy local halo info

template <class buf_t> void Tausch2D<buf_t>::delLocalHaloInfoCwC(size_t haloId) {
    delete[] mpiSendBufferCpuWithCpu[haloId];
    mpiSendRequestsCpuWithCpu[haloId] = MPI_Request();
    localBufferOffsetCwC[haloId] = 0;
    localTotalBufferSizeCwC[haloId] = 0;
    alreadyDeletedLocalHaloIds.push_back(haloId);
}


////////////////////////
/// Add remote halo info

template <class buf_t> size_t Tausch2D<buf_t>::addRemoteHaloInfoCwC(TauschHaloSpec haloSpec) {

    remoteHaloSpecsCpuWithCpu.push_back(haloSpec);

    size_t haloSize = haloSpec.haloWidth*haloSpec.haloHeight;
    if(haloSpec.haloIndicesInBuffer.size() > 0)
        haloSize = haloSpec.haloIndicesInBuffer.size();

    size_t bufsize = 0;
    size_t offset = 0;

    for(size_t n = 0; n < numBuffers; ++n) {
        bufsize += valuesPerPointPerBuffer[n]*haloSize;
        offset += valuesPerPointPerBuffer[n-1] * haloSize;
        remoteBufferOffsetCwC.push_back(offset);
    }

    remoteTotalBufferSizeCwC.push_back(bufsize);
    mpiRecvBufferCpuWithCpu.push_back(new buf_t[bufsize]());
    mpiRecvRequestsCpuWithCpu.push_back(MPI_Request());

    setupMpiRecvCpuWithCpu.push_back(false);

    return mpiRecvBufferCpuWithCpu.size()-1;

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::setRemoteHaloInfoCwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsCpuWithGpu = numHaloParts;
    remoteHaloSpecsCpuWithGpu = new TauschHaloSpec[numHaloParts];
    recvBufferCpuWithGpu = new buf_t*[numHaloParts];

    for(size_t i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsCpuWithGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsCpuWithGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecsCpuWithGpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsCpuWithGpu[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecsCpuWithGpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsCpuWithGpu[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecsCpuWithGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(size_t n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        recvBufferCpuWithGpu[i] = new buf_t[bufsize];

    }

    // These are computed once as they don't change below
    remoteBufferOffsetCwG = new size_t[numHaloParts*numBuffers]{};
    for(size_t nb = 0; nb < numBuffers; ++nb) {
        for(size_t nh = 0; nh < numHaloParts; ++nh) {
            size_t offset = 0;
            for(size_t b = 0; b < nb; ++b)
                offset += valuesPerPointPerBuffer[b] * remoteHaloSpecsCpuWithGpu[nh].haloWidth * remoteHaloSpecsCpuWithGpu[nh].haloHeight;
            remoteBufferOffsetCwG[nb*numHaloParts + nh] = offset;
        }
    }

    // The buffer sizes also do not change anymore
    remoteTotalBufferSizeCwG = new int[numHaloParts]{};
    for(size_t nh = 0; nh < numHaloParts; ++nh)
        for(size_t nb = 0; nb < numBuffers; ++nb)
            remoteTotalBufferSizeCwG[nh] += valuesPerPointPerBuffer[nb]*remoteHaloSpecsCpuWithGpu[nh].haloWidth*remoteHaloSpecsCpuWithGpu[nh].haloHeight;

}

template <class buf_t> void Tausch2D<buf_t>::setRemoteHaloInfoGwC(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

    remoteHaloNumPartsGpuWithCpu = numHaloParts;
    remoteHaloSpecsGpuWithCpu = new TauschHaloSpec[numHaloParts];
    recvBufferGpuWithCpu = new buf_t*[numHaloParts];

    try {
        cl_recvBufferGpuWithCpu = new cl::Buffer[numHaloParts];
        cl_remoteHaloSpecsGpuWithCpu = new cl::Buffer[numHaloParts];
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: enableOpenCL() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    for(size_t i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsGpuWithCpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsGpuWithCpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecsGpuWithCpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsGpuWithCpu[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecsGpuWithCpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsGpuWithCpu[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecsGpuWithCpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(size_t n = 0; n < numBuffers; ++n)
            bufsize += valuesPerPointPerBuffer[n]*haloSpecs[i].haloWidth*haloSpecs[i].haloHeight;
        recvBufferGpuWithCpu[i] = new buf_t[bufsize];

        size_t tmpHaloSpecs[6] = {haloSpecs[i].haloX, haloSpecs[i].haloY,
                                  haloSpecs[i].haloWidth, haloSpecs[i].haloHeight,
                                  haloSpecs[i].bufferWidth, haloSpecs[i].bufferHeight, };

        try {
            cl_recvBufferGpuWithCpu[i] = cl::Buffer(cl_context, CL_MEM_READ_WRITE, bufsize*sizeof(double));
            cl_remoteHaloSpecsGpuWithCpu[i] = cl::Buffer(cl_context, &tmpHaloSpecs[0], &tmpHaloSpecs[6], true);
        } catch(cl::Error error) {
            std::cerr << "Tausch2D :: setRemoteHaloInfo() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

    }

    // These are computed once as they don't change below
    remoteBufferOffsetGwC = new int[numHaloParts*numBuffers]{};
    for(size_t nb = 0; nb < numBuffers; ++nb) {
        for(size_t nh = 0; nh < numHaloParts; ++nh) {
            int offset = 0;
            for(size_t b = 0; b < nb; ++b)
                offset += valuesPerPointPerBuffer[b] * remoteHaloSpecsGpuWithCpu[nh].haloWidth * remoteHaloSpecsGpuWithCpu[nh].haloHeight;
            remoteBufferOffsetGwC[nb*numHaloParts + nh] = offset;
        }
    }

    // The buffer sizes also do not change anymore
    remoteTotalBufferSizeGwC = new size_t[numHaloParts]{};
    for(size_t nh = 0; nh < numHaloParts; ++nh)
        for(size_t nb = 0; nb < numBuffers; ++nb)
            remoteTotalBufferSizeGwC[nh] += valuesPerPointPerBuffer[nb]*remoteHaloSpecsGpuWithCpu[nh].haloWidth*remoteHaloSpecsGpuWithCpu[nh].haloHeight;

}

template <class buf_t> void Tausch2D<buf_t>::setRemoteHaloInfoGwG(size_t numHaloParts, TauschHaloSpec *haloSpecs) {

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

    for(size_t i = 0; i < numHaloParts; ++i) {

        remoteHaloSpecsGpuWithGpu[i].bufferWidth = haloSpecs[i].bufferWidth;
        remoteHaloSpecsGpuWithGpu[i].bufferHeight = haloSpecs[i].bufferHeight;
        remoteHaloSpecsGpuWithGpu[i].haloX = haloSpecs[i].haloX;
        remoteHaloSpecsGpuWithGpu[i].haloY = haloSpecs[i].haloY;
        remoteHaloSpecsGpuWithGpu[i].haloWidth = haloSpecs[i].haloWidth;
        remoteHaloSpecsGpuWithGpu[i].haloHeight = haloSpecs[i].haloHeight;
        remoteHaloSpecsGpuWithGpu[i].remoteMpiRank = haloSpecs[i].remoteMpiRank;

        size_t bufsize = 0;
        for(size_t n = 0; n < numBuffers; ++n)
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

    // These are computed once as they don't change below
    remoteBufferOffsetGwG = new int[numHaloParts*numBuffers]{};
    for(size_t nb = 0; nb < numBuffers; ++nb) {
        for(size_t nh = 0; nh < numHaloParts; ++nh) {
            int offset = 0;
            for(size_t b = 0; b < nb; ++b)
                offset += valuesPerPointPerBuffer[b] * remoteHaloSpecsGpuWithGpu[nh].haloWidth * remoteHaloSpecsGpuWithGpu[nh].haloHeight;
            remoteBufferOffsetGwG[nb*numHaloParts + nh] = offset;
        }
    }

    // The buffer sizes also do not change anymore
    remoteTotalBufferSizeGwG = new size_t[numHaloParts]{};
    for(size_t nh = 0; nh < numHaloParts; ++nh)
        for(size_t nb = 0; nb < numBuffers; ++nb)
            remoteTotalBufferSizeGwG[nh] += valuesPerPointPerBuffer[nb]*remoteHaloSpecsGpuWithGpu[nh].haloWidth*remoteHaloSpecsGpuWithGpu[nh].haloHeight;

}
#endif


/////////////////////////////
/// Destroy remote halo info

template <class buf_t> void Tausch2D<buf_t>::delRemoteHaloInfoCwC(size_t haloId) {
    delete[] mpiRecvBufferCpuWithCpu[haloId];
    mpiRecvRequestsCpuWithCpu[haloId] = MPI_Request();
    remoteBufferOffsetCwC[haloId] = 0;
    remoteTotalBufferSizeCwC[haloId] = 0;
    alreadyDeletedRemoteHaloIds.push_back(haloId);
}


////////////////////////
/// Post Receives

template <class buf_t> void Tausch2D<buf_t>::postReceiveCwC(size_t haloId, int msgtag, int remoteMpiRank) {

    if(!setupMpiRecvCpuWithCpu[haloId]) {

        setupMpiRecvCpuWithCpu[haloId] = true;

        if(remoteMpiRank == -1)
            remoteMpiRank = remoteHaloSpecsCpuWithCpu[haloId].remoteMpiRank;

        // MPI_Recv_init expects 'count' parameter as int
        MPI_Recv_init(&mpiRecvBufferCpuWithCpu[haloId][0], int(remoteTotalBufferSizeCwC[haloId]), mpiDataType,
                      remoteMpiRank, msgtag, TAUSCH_COMM, &mpiRecvRequestsCpuWithCpu[haloId]);

    }

    MPI_Start(&mpiRecvRequestsCpuWithCpu[haloId]);

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::postReceiveCwG(size_t haloId, int msgtag) {
    msgtagsCpuToGpu[haloId].store(msgtag);
}

template <class buf_t> void Tausch2D<buf_t>::postReceiveGwC(size_t haloId, int msgtag) {
    msgtagsGpuToCpu[haloId].store(msgtag);
}

template <class buf_t> void Tausch2D<buf_t>::postReceiveGwG(size_t haloId, int msgtag) {

    if(!setupMpiRecvGpuWithGpu[haloId]) {

        if(msgtag == -1) {
            std::cerr << "[Tausch2D] ERROR: MPI_Recv for halo region #" << haloId << " hasn't been posted before, missing mpitag... Abort!"
                      << std::endl;
            exit(1);
        }

        setupMpiRecvGpuWithGpu[haloId] = true;

        MPI_Recv_init(&mpiRecvBufferGpuWithGpu[haloId][0], int(remoteTotalBufferSizeGwG[haloId]), mpiDataType,
                      remoteHaloSpecsGpuWithGpu[haloId].remoteMpiRank, msgtag, TAUSCH_COMM, &mpiRecvRequestsGpuWithGpu[haloId]);

    }

    MPI_Start(&mpiRecvRequestsGpuWithGpu[haloId]);
}
#endif


////////////////////////
/// Post ALL Receives

template <class buf_t> void Tausch2D<buf_t>::postAllReceivesCwC(int *msgtag) {

    for(size_t id = 0; id < remoteHaloSpecsCpuWithCpu.size(); ++id) {
        if(std::find(alreadyDeletedRemoteHaloIds.begin(), alreadyDeletedRemoteHaloIds.end(), id) == alreadyDeletedRemoteHaloIds.end())
            postReceiveCwC(id, msgtag[id]);
    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::postAllReceivesCwG(int *msgtag) {

#if __cplusplus >= 201103L
    if(msgtag == nullptr) {
#else
    if(msgtag == NULL) {
#endif
        std::cerr << "Tausch2D::postAllReceives :: ERROR :: msgtag cannot be NULL for CpuWithGpu" << std::endl;
        return;
    }

    for(size_t id = 0; id < remoteHaloNumPartsCpuWithGpu; ++id)
        postReceiveCwG(id, msgtag[id]);

}

template <class buf_t> void Tausch2D<buf_t>::postAllReceivesGwC(int *msgtag) {

#if __cplusplus >= 201103L
    if(msgtag == nullptr) {
#else
    if(msgtag == NULL) {
#endif
        std::cerr << "Tausch2D::postAllReceives :: ERROR :: msgtag cannot be NULL for GpuWithGpu" << std::endl;
        return;
    }

    for(size_t id = 0; id < remoteHaloNumPartsGpuWithCpu; ++id)
        postReceiveGwC(id, msgtag[id]);

}

template <class buf_t> void Tausch2D<buf_t>::postAllReceivesGwG(int *msgtag) {

#if __cplusplus >= 201103L
    if(msgtag == nullptr) {
#else
    if(msgtag == NULL) {
#endif
        msgtag = new int[remoteHaloNumPartsGpuWithGpu];
        for(size_t id = 0; id < remoteHaloNumPartsGpuWithGpu; ++id)
            msgtag[id] = -1;
    }

    for(size_t id = 0; id < remoteHaloNumPartsGpuWithGpu; ++id)
        postReceiveGwG(id, msgtag[id]);

}
#endif


////////////////////////
/// Pack send buffer

template <class buf_t> void Tausch2D<buf_t>::packSendBufferCwC(size_t haloId, size_t bufferId, buf_t *buf) {
    TauschPackRegion region;
    if(localHaloSpecsCpuWithCpu[haloId].haloIndicesInBuffer.size() == 0) {
        region.x = 0;
        region.y = 0;
        region.width = localHaloSpecsCpuWithCpu[haloId].haloWidth;
        region.height = localHaloSpecsCpuWithCpu[haloId].haloHeight;
    } else {
        region.startAtIndex = 0;
        region.endAtIndex = localHaloSpecsCpuWithCpu[haloId].haloIndicesInBuffer.size();
    }
    packSendBufferCwC(haloId, bufferId, buf, region);
}

template <class buf_t> void Tausch2D<buf_t>::packSendBufferCwC(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    if(localHaloSpecsCpuWithCpu[haloId].haloIndicesInBuffer.size() > 0) {

        size_t haloSize = localHaloSpecsCpuWithCpu[haloId].haloIndicesInBuffer.size();

        for(size_t index = region.startAtIndex; index < region.endAtIndex; ++index)
            mpiSendBufferCpuWithCpu[haloId][bufferId*haloSize + index] = buf[localHaloSpecsCpuWithCpu[haloId].haloIndicesInBuffer[index]];

        return;

    }

    size_t bufIndexBase = (region.y + localHaloSpecsCpuWithCpu[haloId].haloY)*localHaloSpecsCpuWithCpu[haloId].bufferWidth +
                        localHaloSpecsCpuWithCpu[haloId].haloX + region.x;
    size_t mpiIndexBase = region.y*localHaloSpecsCpuWithCpu[haloId].haloWidth + region.x;

    if(valuesPerPointPerBufferAllOne) {

        mpiIndexBase += localBufferOffsetCwC[numBuffers*haloId + bufferId];

        for(size_t h = 0; h < region.height; ++h) {
            size_t hbw = bufIndexBase + h*localHaloSpecsCpuWithCpu[haloId].bufferWidth;
            size_t hhw = mpiIndexBase + h*localHaloSpecsCpuWithCpu[haloId].haloWidth;

#if defined (USEMEMCPY)
            memcpy(&mpiSendBufferCpuWithCpu[haloId][hhw], &buf[hbw], region.width*sizeof(double));
#elif defined (USESTLCPY)
            std::copy(&buf[hbw], &buf[hbw+region.width], &mpiSendBufferCpuWithCpu[haloId][hhw]);
#else
            for(size_t w = 0; w < region.width; ++w)
                mpiSendBufferCpuWithCpu[haloId][hhw + w] = buf[hbw + w];
#endif

        }

    } else {

        for(size_t h = 0; h < region.height; ++h) {

            size_t hbw = h*localHaloSpecsCpuWithCpu[haloId].bufferWidth;
            size_t hhw = h*localHaloSpecsCpuWithCpu[haloId].haloWidth;

            for(size_t w = 0; w < region.width; ++w) {

                size_t bufIndex = valuesPerPointPerBuffer[bufferId] * (bufIndexBase + hbw + w);
                size_t mpiIndex = localBufferOffsetCwC[numBuffers*haloId + bufferId] + valuesPerPointPerBuffer[bufferId]*(mpiIndexBase + hhw + w);

#if defined (USEMEMCPY)
                memcpy(&mpiSendBufferCpuWithCpu[haloId][mpiIndex], &buf[bufIndex], valuesPerPointPerBuffer[bufferId]*sizeof(double));
#elif defined (USESTLCPY)
                std::copy(&buf[bufIndex], &buf[bufIndex + valuesPerPointPerBuffer[bufferId]], &mpiSendBufferCpuWithCpu[haloId][mpiIndex]);
#else
                for(size_t val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val)
                    mpiSendBufferCpuWithCpu[haloId][mpiIndex + val] =
                            buf[bufIndex + val];
#endif

            }
        }

    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::packSendBufferCwG(size_t haloId, size_t bufferId, buf_t *buf) {
    TauschPackRegion region;
    region.x = 0;
    region.y = 0;
    region.width = localHaloSpecsCpuWithGpu[haloId].haloWidth;
    region.height = localHaloSpecsCpuWithGpu[haloId].haloHeight;
    packSendBufferCwG(haloId, bufferId, buf, region);
}

template <class buf_t> void Tausch2D<buf_t>::packSendBufferCwG(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    size_t bufIndexBase = (localHaloSpecsCpuWithGpu[haloId].haloY + region.y)*localHaloSpecsCpuWithGpu[haloId].bufferWidth+
                       +localHaloSpecsCpuWithGpu[haloId].haloX + region.x;
    size_t mpiIndexBase = region.y*localHaloSpecsCpuWithGpu[haloId].haloWidth + region.x;

    if(valuesPerPointPerBufferAllOne) {

        mpiIndexBase += localBufferOffsetCwG[localHaloNumPartsCpuWithGpu*bufferId + haloId];

        for(size_t h = 0; h < region.height; ++h) {

            size_t hbw = bufIndexBase + h*localHaloSpecsCpuWithGpu[haloId].bufferWidth;
            size_t hhw = mpiIndexBase + h*localHaloSpecsCpuWithGpu[haloId].haloWidth;

            for(size_t w = 0; w < region.width; ++w)
                sendBufferCpuWithGpu[haloId][hhw + w] = buf[hbw + w];

        }

    } else {

        size_t offset = localBufferOffsetCwG[localHaloNumPartsCpuWithGpu*bufferId + haloId];

        for(size_t h = 0; h < region.height; ++h) {

            size_t hbw = h*localHaloSpecsCpuWithGpu[haloId].bufferWidth;
            size_t hhw = h*localHaloSpecsCpuWithGpu[haloId].haloWidth;

            for(size_t w = 0; w < region.width; ++w) {

                size_t bufIndex = bufIndexBase + hbw + w;
                size_t mpiIndex = mpiIndexBase + hhw + w;

                for(size_t val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val)
                    sendBufferCpuWithGpu[haloId][offset + valuesPerPointPerBuffer[bufferId]*mpiIndex + val]
                            .store(buf[valuesPerPointPerBuffer[bufferId]*bufIndex + val]);

            }

        }

    }

}

template <class buf_t> void Tausch2D<buf_t>::packSendBufferGwC(size_t haloId, size_t bufferId, cl::Buffer buf) {

    try {

        auto kernel_packNextSendBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                    (cl_programs, "packSendBuffer");

        size_t globalsize = (localTotalBufferSizeGwC[haloId]/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        cl::Buffer cl_bufferId(cl_context, &bufferId, (&bufferId)+1, true);

        kernel_packNextSendBuffer(cl::EnqueueArgs(cl_queue, cl::NDRange(globalsize), cl::NDRange(cl_kernelLocalSize)),
                                  cl_localHaloSpecsGpuWithCpu[haloId], cl_valuesPerPointPerBuffer,
                                  cl_bufferId, cl_sendBufferGpuWithCpu[haloId], buf);

    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch2D<buf_t>::packSendBufferGwG(size_t haloId, size_t bufferId, cl::Buffer buf) {

    try {

        auto kernel_packSendBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                    (cl_programs, "packSendBuffer");

        size_t globalsize = (localTotalBufferSizeGwG[haloId]/cl_kernelLocalSize +1)*cl_kernelLocalSize;

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

template <class buf_t> void Tausch2D<buf_t>::sendCwC(size_t haloId, int msgtag, int remoteMpiRank) {

    if(!setupMpiSendCpuWithCpu[haloId]) {

        setupMpiSendCpuWithCpu[haloId] = true;

        if(remoteMpiRank == -1)
            remoteMpiRank = localHaloSpecsCpuWithCpu[haloId].remoteMpiRank;

        // MPI_Send_init expects 'count' parameter to be of type int
        MPI_Send_init(&mpiSendBufferCpuWithCpu[haloId][0], int(localTotalBufferSizeCwC[haloId]), mpiDataType, remoteMpiRank,
                  msgtag, TAUSCH_COMM, &mpiSendRequestsCpuWithCpu[haloId]);

    } else
        MPI_Wait(&mpiSendRequestsCpuWithCpu[haloId], MPI_STATUS_IGNORE);

    MPI_Start(&mpiSendRequestsCpuWithCpu[haloId]);

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::sendCwG(size_t haloId, int msgtag) {
    msgtagsCpuToGpu[haloId].store(msgtag);
    syncTwoThreads();
}

template <class buf_t> void Tausch2D<buf_t>::sendGwC(size_t haloId, int msgtag) {

    msgtagsGpuToCpu[haloId].store(msgtag);

    try {

        buf_t *tmp = new buf_t[localTotalBufferSizeGwC[haloId]];
        cl::copy(cl_queue, cl_sendBufferGpuWithCpu[haloId], &tmp[0], &tmp[localTotalBufferSizeGwC[haloId]]);
        for(size_t i = 0; i < localTotalBufferSizeGwC[haloId]; ++i)
            sendBufferGpuWithCpu[haloId][i].store(tmp[i]);
        delete[] tmp;

    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: sendGpuToCpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    syncTwoThreads();

}

template <class buf_t> void Tausch2D<buf_t>::sendGwG(size_t haloId, int msgtag) {

    if(!setupMpiSendGpuWithGpu[haloId]) {

        if(msgtag == -1) {
            std::cerr << "[Tausch2D] ERROR: MPI_Send for halo region #" << haloId << " hasn't been posted before, missing msgtag... Abort!"
                      << std::endl;
            exit(1);
        }

        setupMpiSendGpuWithGpu[haloId] = true;

        MPI_Send_init(&mpiSendBufferGpuWithGpu[haloId][0], int(localTotalBufferSizeGwG[haloId]), mpiDataType, localHaloSpecsGpuWithGpu[haloId].remoteMpiRank,
                  msgtag, TAUSCH_COMM, &mpiSendRequestsGpuWithGpu[haloId]);

    } else
        MPI_Wait(&mpiSendRequestsGpuWithGpu[haloId], MPI_STATUS_IGNORE);

    try {
        cl::copy(cl_queue, cl_sendBufferGpuWithGpu[haloId], &mpiSendBufferGpuWithGpu[haloId][0], &mpiSendBufferGpuWithGpu[haloId][localTotalBufferSizeGwG[haloId]]);
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: sendGpuWithGpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

    MPI_Start(&mpiSendRequestsGpuWithGpu[haloId]);

}
#endif


////////////////////////
/// Receive data

template <class buf_t> void Tausch2D<buf_t>::recvCwC(size_t haloId) {
    MPI_Wait(&mpiRecvRequestsCpuWithCpu[haloId], MPI_STATUS_IGNORE);
}

template <class buf_t> void Tausch2D<buf_t>::recvAllCwC() {
    MPI_Waitall(mpiRecvRequestsCpuWithCpu.size(), mpiRecvRequestsCpuWithCpu.data(), MPI_STATUSES_IGNORE);
}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::recvCwG(size_t haloId) {

    syncTwoThreads();

    size_t remoteid = obtainRemoteId(msgtagsGpuToCpu[haloId]);

    for(int i = 0; i < remoteTotalBufferSizeCwG[haloId]; ++i)
        recvBufferCpuWithGpu[haloId][i] = sendBufferGpuWithCpu[remoteid][i].load();

}

template <class buf_t> void Tausch2D<buf_t>::recvGwC(size_t haloId) {

    syncTwoThreads();

    size_t remoteid = obtainRemoteId(msgtagsCpuToGpu[haloId]);

    for(size_t j = 0; j < remoteTotalBufferSizeGwC[haloId]; ++j)
        recvBufferGpuWithCpu[haloId][j] = sendBufferCpuWithGpu[remoteid][j].load();

    try {
        cl_recvBufferGpuWithCpu[haloId] = cl::Buffer(cl_context, &recvBufferGpuWithCpu[haloId][0], &recvBufferGpuWithCpu[haloId][remoteTotalBufferSizeGwC[haloId]], false);
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: recvCpuToGpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch2D<buf_t>::recvGwG(size_t haloId) {

    MPI_Wait(&mpiRecvRequestsGpuWithGpu[haloId], MPI_STATUS_IGNORE);

    try {
        cl_recvBufferGpuWithGpu[haloId] = cl::Buffer(cl_context, &mpiRecvBufferGpuWithGpu[haloId][0], &mpiRecvBufferGpuWithGpu[haloId][remoteTotalBufferSizeGwG[haloId]], false);
    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: recvGpuWithGpu() :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}
#endif


////////////////////////
/// Unpack received data

template <class buf_t> void Tausch2D<buf_t>::unpackRecvBufferCwC(size_t haloId, size_t bufferId, buf_t *buf) {
    TauschPackRegion region;
    if(remoteHaloSpecsCpuWithCpu[haloId].haloIndicesInBuffer.size() == 0) {
        region.x = 0;
        region.y = 0;
        region.width = remoteHaloSpecsCpuWithCpu[haloId].haloWidth;
        region.height = remoteHaloSpecsCpuWithCpu[haloId].haloHeight;
    } else {
        region.startAtIndex = 0;
        region.endAtIndex = remoteHaloSpecsCpuWithCpu[haloId].haloIndicesInBuffer.size();
    }
    unpackRecvBufferCwC(haloId, bufferId, buf, region);
}

template <class buf_t> void Tausch2D<buf_t>::unpackRecvBufferCwC(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    if(remoteHaloSpecsCpuWithCpu[haloId].haloIndicesInBuffer.size() != 0) {

        size_t haloSize = remoteHaloSpecsCpuWithCpu[haloId].haloIndicesInBuffer.size();

        for(size_t index = region.startAtIndex; index < region.endAtIndex; ++index)
            buf[remoteHaloSpecsCpuWithCpu[haloId].haloIndicesInBuffer[index]] = mpiRecvBufferCpuWithCpu[haloId][bufferId*haloSize + index];

        return;

    }

    size_t bufIndexBase = (region.y + remoteHaloSpecsCpuWithCpu[haloId].haloY)*remoteHaloSpecsCpuWithCpu[haloId].bufferWidth +
                        remoteHaloSpecsCpuWithCpu[haloId].haloX + region.x;
    size_t mpiIndexBase = (region.y)*remoteHaloSpecsCpuWithCpu[haloId].haloWidth + region.x;

    if(valuesPerPointPerBufferAllOne) {

        mpiIndexBase += remoteBufferOffsetCwC[numBuffers*haloId + bufferId];

        for(size_t h = 0; h < region.height; ++h) {

            size_t hbw = bufIndexBase + h*remoteHaloSpecsCpuWithCpu[haloId].bufferWidth;
            size_t hhw = mpiIndexBase + h*remoteHaloSpecsCpuWithCpu[haloId].haloWidth;

#if defined (USEMEMCPY)
            memcpy(&buf[hbw], &mpiRecvBufferCpuWithCpu[haloId][hhw], region.width*sizeof(double));
#elif defined (USESTLCPY)
            std::copy(&mpiRecvBufferCpuWithCpu[haloId][hhw], &mpiRecvBufferCpuWithCpu[haloId][hhw+region.width], &buf[hbw]);
#else
            for(size_t w = 0; w < region.width; ++w)
                buf[hbw + w] = mpiRecvBufferCpuWithCpu[haloId][hhw + w];
#endif

        }

    } else {

        for(size_t h = 0; h < region.height; ++h) {

            size_t hbw = bufIndexBase + h*remoteHaloSpecsCpuWithCpu[haloId].bufferWidth;
            size_t hhw = mpiIndexBase + h*remoteHaloSpecsCpuWithCpu[haloId].haloWidth;

            for(size_t w = 0; w < region.width; ++w) {

                size_t bufIndex = valuesPerPointPerBuffer[bufferId]* ( hbw + w );
                size_t fullerOffset = remoteBufferOffsetCwC[numBuffers*haloId + bufferId] + valuesPerPointPerBuffer[bufferId]* ( hhw + w );

#if defined (USEMEMCPY)
                memcpy(&buf[bufIndex], &mpiRecvBufferCpuWithCpu[haloId][fullerOffset], valuesPerPointPerBuffer[bufferId]*sizeof(double));
#elif defined (USESTLCPY)
                std::copy(&mpiRecvBufferCpuWithCpu[haloId][fullerOffset], &mpiRecvBufferCpuWithCpu[haloId][fullerOffset+valuesPerPointPerBuffer[bufferId]], &buf[bufIndex]);
#else
                for(size_t val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val)
                    buf[bufIndex + val] = mpiRecvBufferCpuWithCpu[haloId][fullerOffset + val];
#endif

            }

        }

    }

}

#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::unpackRecvBufferCwG(size_t haloId, size_t bufferId, buf_t *buf) {
    TauschPackRegion region;
    region.x = 0;
    region.y = 0;
    region.width = remoteHaloSpecsCpuWithGpu[haloId].haloWidth;
    region.height = remoteHaloSpecsCpuWithGpu[haloId].haloHeight;
    unpackRecvBufferCwG(haloId, bufferId, buf, region);
}

template <class buf_t> void Tausch2D<buf_t>::unpackRecvBufferCwG(size_t haloId, size_t bufferId, buf_t *buf, TauschPackRegion region) {

    size_t bufIndexBase = (remoteHaloSpecsCpuWithGpu[haloId].haloY + region.y)*remoteHaloSpecsCpuWithGpu[haloId].bufferWidth
                        + remoteHaloSpecsCpuWithGpu[haloId].haloX + region.x;
    size_t mpiIndexBase = region.y*remoteHaloSpecsCpuWithGpu[haloId].haloWidth + region.x;

    if(valuesPerPointPerBufferAllOne) {

        mpiIndexBase += remoteBufferOffsetCwG[remoteHaloNumPartsCpuWithGpu*bufferId + haloId];

        for(size_t h = 0; h < region.height; ++h) {

            size_t hbw = bufIndexBase + h*remoteHaloSpecsCpuWithGpu[haloId].bufferWidth;
            size_t hhw = mpiIndexBase + h*remoteHaloSpecsCpuWithGpu[haloId].haloWidth;

            for(size_t w = 0; w < region.width; ++w)
                buf[hbw + w] = recvBufferCpuWithGpu[haloId][hhw + w];

        }

    } else {

        for(size_t h = 0; h < region.height; ++h) {

            size_t hbw = bufIndexBase + h*remoteHaloSpecsCpuWithGpu[haloId].bufferWidth;
            size_t hhw = mpiIndexBase + h*remoteHaloSpecsCpuWithGpu[haloId].haloWidth;

            for(size_t w = 0; w < region.width; ++w) {

                size_t bufIndex = valuesPerPointPerBuffer[bufferId]* ( hbw + w );
                size_t mpiIndex = remoteBufferOffsetCwG[remoteHaloNumPartsCpuWithGpu*bufferId + haloId] + valuesPerPointPerBuffer[bufferId]* ( hhw + w );

                for(size_t val = 0; val < valuesPerPointPerBuffer[bufferId]; ++val)
                    buf[bufIndex + val] = recvBufferCpuWithGpu[haloId][mpiIndex + val];

            }

        }

    }

}

template <class buf_t> void Tausch2D<buf_t>::unpackRecvBufferGwC(size_t haloId, size_t bufferId, cl::Buffer buf) {

    try {
        auto kernel_unpackRecvBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                (cl_programs, "unpackRecvBuffer");

        size_t bufsize = remoteTotalBufferSizeGwC[haloId];

        size_t globalsize = (bufsize/cl_kernelLocalSize +1)*cl_kernelLocalSize;

        cl::Buffer cl_bufferId(cl_context, &bufferId, (&bufferId)+1, true);

        kernel_unpackRecvBuffer(cl::EnqueueArgs(cl_queue, cl::NDRange(globalsize), cl::NDRange(cl_kernelLocalSize)),
                                cl_remoteHaloSpecsGpuWithCpu[haloId], cl_valuesPerPointPerBuffer, cl_bufferId,
                                cl_recvBufferGpuWithCpu[haloId], buf);

    } catch(cl::Error error) {
        std::cerr << "Tausch2D :: unpackRecvBufferCpuToGpu() :: OpenCL exception caught: " << error.what()
                  << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

template <class buf_t> void Tausch2D<buf_t>::unpackRecvBufferGwG(size_t haloId, size_t bufferId, cl::Buffer buf) {

    try {
        auto kernel_unpackRecvBuffer = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                (cl_programs, "unpackRecvBuffer");

        size_t globalsize = (remoteTotalBufferSizeGwG[haloId]/cl_kernelLocalSize +1)*cl_kernelLocalSize;

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

template <class buf_t> void Tausch2D<buf_t>::packAndSendCwC(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag) {
    packSendBufferCwC(haloId, 0, buf, region);
    sendCwC(haloId, msgtag);
}
template <class buf_t> void Tausch2D<buf_t>::packAndSendCwC(size_t haloId, buf_t *buf, int msgtag) {
    packSendBufferCwC(haloId, 0, buf);
    sendCwC(haloId, msgtag);
}
#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::packAndSendCwG(size_t haloId, buf_t *buf, TauschPackRegion region, int msgtag) {
    packSendBufferCwG(haloId, 0, buf, region);
    sendCwG(haloId, msgtag);
}
template <class buf_t> void Tausch2D<buf_t>::packAndSendGwC(size_t haloId, cl::Buffer buf, int msgtag) {
    packSendBufferGwC(haloId, 0, buf);
    sendGwC(haloId, msgtag);
}
template <class buf_t> void Tausch2D<buf_t>::packAndSendGwG(size_t haloId, cl::Buffer buf, int msgtag) {
    packSendBufferGwG(haloId, 0, buf);
    sendGwG(haloId, msgtag);
}
#endif


////////////////////////
/// Receive data and unpack

template <class buf_t> void Tausch2D<buf_t>::recvAndUnpackCwC(size_t haloId, buf_t *buf, TauschPackRegion region) {
    recvCwC(haloId);
    unpackRecvBufferCwC(haloId, 0, buf, region);
}
template <class buf_t> void Tausch2D<buf_t>::recvAndUnpackCwC(size_t haloId, buf_t *buf) {
    recvCwC(haloId);
    unpackRecvBufferCwC(haloId, 0, buf);
}
#ifdef TAUSCH_OPENCL
template <class buf_t> void Tausch2D<buf_t>::recvAndUnpackCwG(size_t haloId, buf_t *buf, TauschPackRegion region) {
    recvGwC(haloId);
    unpackRecvBufferCwG(haloId, 0, buf, region);
}
template <class buf_t> void Tausch2D<buf_t>::recvAndUnpackGwC(size_t haloId, cl::Buffer buf) {
    recvCwC(haloId);
    unpackRecvBufferGwC(haloId, 0, buf);
}
template <class buf_t> void Tausch2D<buf_t>::recvAndUnpackGwG(size_t haloId, cl::Buffer buf) {
    recvGwG(haloId);
    unpackRecvBufferGwG(haloId, 0, buf);
}
#endif


template <class buf_t> TauschPackRegion Tausch2D<buf_t>::createFilledPackRegion(size_t x, size_t y, size_t width, size_t height) {
    TauschPackRegion region;
    region.x = x;
    region.y = y;
    region.width = width;
    region.height = height;
    return region;
}

template <class buf_t> TauschPackRegion Tausch2D<buf_t>::createFilledPackRegion(size_t startAtIndex, size_t endAtIndex) {
    TauschPackRegion region;
    region.startAtIndex = startAtIndex;
    region.endAtIndex = endAtIndex;
    return region;
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

template <class buf_t> TauschHaloSpec Tausch2D<buf_t>::createFilledHaloSpec(std::vector<size_t> haloIndicesInBuffer) {
    TauschHaloSpec halo;
    halo.haloIndicesInBuffer = haloIndicesInBuffer;
    return halo;
}


///////////////////////////////////////////////////////////////
/// SOME GENERAL PURPOSE OPENCL FUNCTIONS - PART OF PUBLIC API
///////////////////////////////////////////////////////////////

#ifdef TAUSCH_OPENCL

template <class buf_t> void Tausch2D<buf_t>::enableOpenCL(bool blockingSyncCpuGpu, size_t clLocalWorkgroupSize, bool giveOpenCLDeviceName,
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

template <class buf_t> size_t Tausch2D<buf_t>::obtainRemoteId(int msgtag) {
    for(size_t j = 0; j < remoteHaloNumPartsCpuWithGpu; ++j) {
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

template <class buf_t> void Tausch2D<buf_t>::compileKernels() {

    std::string oclstr =
"kernel void packSendBuffer(global const size_t * restrict const haloSpecs,"
"                           global const size_t * restrict const valuesPerPointPerBuffer, global int * restrict const bufferId,"
"                           global double * restrict const haloBuffer, global const double * restrict const buffer) {"

"    const int current = get_global_id(0);"

"    int maxSize = haloSpecs[2]*haloSpecs[3];"

"    if(current >= maxSize) return;"

"    int index = (current/haloSpecs[2] + haloSpecs[1])*haloSpecs[4] +"
"                 current%haloSpecs[2] + haloSpecs[0];"

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

"    int maxSize = haloSpecs[2]*haloSpecs[3];"

"    if(current >= maxSize) return;"

"    int index = (current/haloSpecs[2] + haloSpecs[1])*haloSpecs[4] +"
"                 current%haloSpecs[2] + haloSpecs[0];"

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
