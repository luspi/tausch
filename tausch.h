#ifndef TAUSCH_H
#define TAUSCH_H

#include <mpi.h>
#include <vector>

#ifdef TAUSCH_OPENCL
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#endif

#ifdef TAUSCH_CUDA
#include <cuda_runtime.h>
#endif

enum TauschOptimizationHint {
    NoHints = 1,
    StaysOnDevice = 2,
    DoesNotStayOnDevice = 4,
};

template <class buf_t>
class Tausch {

public:
#ifdef TAUSCH_OPENCL
    Tausch(cl::Device device, cl::Context context, cl::CommandQueue queue, std::string cName4BufT,
           const MPI_Datatype mpiDataType, const MPI_Comm comm = MPI_COMM_WORLD, const bool useDuplicateOfCommunicator = true) {
#else
    Tausch(const MPI_Datatype mpiDataType, const MPI_Comm comm = MPI_COMM_WORLD, const bool useDuplicateOfCommunicator = true) {
#endif

        if(useDuplicateOfCommunicator)
            MPI_Comm_dup(comm, &TAUSCH_COMM);
        else
            TAUSCH_COMM = comm;

        this->mpiDataType = mpiDataType;

#ifdef TAUSCH_OPENCL
        this->device = device;
        this->context = context;
        this->queue = queue;
        this->cName4BufT = cName4BufT;

        clKernelLocalSize = 256;

        std::string oclstr = "typedef "+cName4BufT+" buf_t;";

        oclstr += R"d(

kernel void packSubRegion(global const buf_t * restrict inBuf, global buf_t * restrict outBuf,
                          global const int * restrict inIndices, const int numIndices,
                          const int bufferOffset) {

    int gid = get_global_id(0);

    if(gid < numIndices)
        outBuf[gid] = inBuf[bufferOffset + inIndices[gid]];

}

kernel void unpackSubRegion(global const buf_t * restrict inBuf, global buf_t * restrict outBuf,
                            global const int * restrict outIndices, const int numIndices,
                            const int bufferOffset) {

    int gid = get_global_id(0);

    if(gid < numIndices)
      outBuf[bufferOffset + outIndices[gid]] = inBuf[gid];

}
                             )d";

        try {

            programs = cl::Program(context, oclstr, false);
            programs.build("");

        } catch(cl::Error &e) {

            std::cout << "Tausch::Tausch(): OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;

            if(e.err() == -11) {
                try {
                    std::string log = programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                    std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl
                              << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** "
                              << std::endl << std::endl;
                } catch(cl::Error &e) {
                    std::cout << "Tausch::Tausch()::getBuildInfo(): OpenCL exception caught: " << e.what() << " (" << e.err() << ")"
                              << std::endl;
                }
            }

        }
#endif

    }

    ~Tausch() { }

    inline int addLocalHaloInfo(std::vector<int> haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {
        return addLocalHaloInfo(extractHaloIndicesWithStride(haloIndices), numBuffers, remoteMpiRank, hints);
    }

    inline int addLocalHaloInfo(std::vector<size_t> haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {
        return addLocalHaloInfo(extractHaloIndicesWithStride(std::vector<int>(haloIndices.begin(), haloIndices.end())),
                                numBuffers, remoteMpiRank, hints);
    }

    inline int addLocalHaloInfo(std::vector<std::array<int, 4> > haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {

        int haloSize = 0;
        for(auto tuple : haloIndices)
            haloSize += tuple[1];

        localHaloIndices.push_back(haloIndices);
        localHaloIndicesSize.push_back(haloSize);
        localHaloNumBuffers.push_back(numBuffers);
        localHaloRemoteMpiRank.push_back(remoteMpiRank);

        localOptHints.push_back(hints);

        void *newbuf = NULL;
        posix_memalign(&newbuf, 64, numBuffers*haloSize*sizeof(buf_t));
        buf_t *newbuf_buft = reinterpret_cast<buf_t*>(newbuf);
        double zero = 0;
        std::fill_n(newbuf_buft, numBuffers*haloSize, zero);
        sendBuffer.push_back(newbuf_buft);

        mpiSendRequests.push_back(new MPI_Request());

        setupMpiSend.push_back(false);

        return sendBuffer.size()-1;

    }


    inline int addRemoteHaloInfo(std::vector<int> haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {
        return addRemoteHaloInfo(extractHaloIndicesWithStride(haloIndices), numBuffers, remoteMpiRank, hints);
    }

    inline int addRemoteHaloInfo(std::vector<size_t> haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {
        return addRemoteHaloInfo(extractHaloIndicesWithStride(std::vector<int>(haloIndices.begin(), haloIndices.end())),
                                 numBuffers, remoteMpiRank, hints);
    }

    inline int addRemoteHaloInfo(std::vector<std::array<int, 4> > haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {

        int haloSize = 0;
        for(auto tuple : haloIndices)
            haloSize += tuple[1];

        remoteHaloIndices.push_back(haloIndices);
        remoteHaloIndicesSize.push_back(haloSize);
        remoteHaloNumBuffers.push_back(numBuffers);
        remoteHaloRemoteMpiRank.push_back(remoteMpiRank);

        remoteOptHints.push_back(hints);

        void *newbuf = NULL;
        posix_memalign(&newbuf, 64, numBuffers*haloSize*sizeof(buf_t));
        buf_t *newbuf_buft = reinterpret_cast<buf_t*>(newbuf);
        double zero = 0;
        std::fill_n(newbuf_buft, numBuffers*haloSize, zero);
        recvBuffer.push_back(newbuf_buft);

        mpiRecvRequests.push_back(new MPI_Request());

        setupMpiRecv.push_back(false);

        return recvBuffer.size()-1;

    }

    inline void packSendBuffer(const size_t haloId, const size_t bufferId, const buf_t *buf) const {

        const size_t haloSize = localHaloIndicesSize[haloId];

        size_t mpiSendBufferIndex = 0;
        for(size_t region = 0; region < localHaloIndices[haloId].size(); ++region) {
            const std::array<int, 4> vals = localHaloIndices[haloId][region];

            const int val_start = vals[0];
            const int val_howmany = vals[1];
            const int val_stride1 = vals[2];
            const int val_stride2 = vals[3];

            if(val_stride1 == 1 && val_stride2 > 1) {

                for(int c = 0; c < val_howmany/2; ++c) {
                    sendBuffer[haloId][bufferId*haloSize + mpiSendBufferIndex + 0] = buf[val_start+c*(val_stride1+val_stride2)   ];
                    sendBuffer[haloId][bufferId*haloSize + mpiSendBufferIndex + 1] = buf[val_start+c*(val_stride1+val_stride2) +1];
                    mpiSendBufferIndex += 2;
                }

            } else if(val_stride1 == 1 && val_stride2 == 0) {
                memcpy(&sendBuffer[haloId][bufferId*haloSize + mpiSendBufferIndex], &buf[val_start], val_howmany*sizeof(buf_t));
                mpiSendBufferIndex += val_howmany;
            } else {
                const int mpiSendBufferIndexBASE = bufferId*haloSize + mpiSendBufferIndex;
                for(int i = 0; i < val_howmany; ++i)
                    sendBuffer[haloId][mpiSendBufferIndexBASE + i] = buf[val_start+i*val_stride1];
                mpiSendBufferIndex += val_howmany;
            }

        }

    }

    inline void packSendBuffer(const size_t haloId, const size_t bufferId, const buf_t *buf,
                               std::vector<size_t> overwriteHaloSendIndices, std::vector<size_t> overwriteHaloSourceIndices) const {

        size_t haloSize = localHaloIndicesSize[haloId];

        for(size_t index = 0; index < overwriteHaloSendIndices.size(); ++index)
            sendBuffer[haloId][bufferId*haloSize + overwriteHaloSendIndices[index]] = buf[overwriteHaloSourceIndices[index]];

    }

    inline MPI_Request *send(size_t haloId, const int msgtag, int remoteMpiRank = -1) {

        if(localHaloIndices[haloId].size() == 0)
            return nullptr;

        if(!setupMpiSend[haloId]) {

            if(remoteMpiRank == -1)
                remoteMpiRank = localHaloRemoteMpiRank[haloId];

            // if we stay on the same rank, we don't need to use MPI
            int myRank;
            MPI_Comm_rank(TAUSCH_COMM, &myRank);
            if(remoteMpiRank == myRank) {
                msgtagToHaloId[myRank*1000000 + msgtag] = haloId;
                return nullptr;
            }

            setupMpiSend[haloId] = true;

            MPI_Send_init(&sendBuffer[haloId][0], localHaloNumBuffers[haloId]*localHaloIndicesSize[haloId], mpiDataType, remoteMpiRank,
                      msgtag, TAUSCH_COMM, mpiSendRequests[haloId]);

        } else
            MPI_Wait(mpiSendRequests[haloId], MPI_STATUS_IGNORE);

        MPI_Start(mpiSendRequests[haloId]);

        return mpiSendRequests[haloId];

    }

    inline MPI_Request *recv(size_t haloId, const int msgtag, int remoteMpiRank = -1, const bool blocking = true) {

        if(remoteHaloIndices[haloId].size() == 0)
            return nullptr;

        if(!setupMpiRecv[haloId]) {

            if(remoteMpiRank == -1)
                remoteMpiRank = remoteHaloRemoteMpiRank[haloId];

            // if we stay on the same rank, we don't need to use MPI
            int myRank;
            MPI_Comm_rank(TAUSCH_COMM, &myRank);

            if(remoteMpiRank == myRank) {

                const int remoteHaloId = msgtagToHaloId[myRank*1000000 + msgtag];

                memcpy(recvBuffer[haloId], sendBuffer[remoteHaloId], remoteHaloNumBuffers[haloId]*remoteHaloIndicesSize[haloId]*sizeof(buf_t));

            } else {

                setupMpiRecv[haloId] = true;

                MPI_Recv_init(&recvBuffer[haloId][0], remoteHaloNumBuffers[haloId]*remoteHaloIndicesSize[haloId], mpiDataType,
                              remoteMpiRank, msgtag, TAUSCH_COMM, mpiRecvRequests[haloId]);
            }

        }

        // this will remain false if we remained on the same mpi rank
        if(setupMpiRecv[haloId]) {

            MPI_Start(mpiRecvRequests[haloId]);
            if(blocking)
                MPI_Wait(mpiRecvRequests[haloId], MPI_STATUS_IGNORE);

            return mpiRecvRequests[haloId];

        } else

            return nullptr;

    }

    inline void unpackRecvBuffer(const size_t haloId, const size_t bufferId, buf_t *buf) const {

        size_t haloSize = remoteHaloIndicesSize[haloId];

        size_t mpiRecvBufferIndex = 0;
        for(size_t region = 0; region < remoteHaloIndices[haloId].size(); ++region) {
            const std::array<int, 4> vals = remoteHaloIndices[haloId][region];

            const int val_start = vals[0];
            const int val_howmany = vals[1];
            const int val_stride1 = vals[2];
            const int val_stride2 = vals[3];

            if(val_stride1 == 1 && val_stride2 > 1) {

                for(int c = 0; c < val_howmany/2; ++c) {
                    buf[val_start+c*(val_stride1+val_stride2)    ] = recvBuffer[haloId][bufferId*haloSize + mpiRecvBufferIndex + 0];
                    buf[val_start+c*(val_stride1+val_stride2) + 1] = recvBuffer[haloId][bufferId*haloSize + mpiRecvBufferIndex + 1];
                    mpiRecvBufferIndex += 2;
                }

            } else if(val_stride1 == 1 && val_stride2 == 0) {
                memcpy(&buf[val_start], &recvBuffer[haloId][bufferId*haloSize + mpiRecvBufferIndex], val_howmany*sizeof(buf_t));
                mpiRecvBufferIndex += val_howmany;
            } else {
                const size_t mpirecvBufferIndexBASE = bufferId*haloSize + mpiRecvBufferIndex;
                for(int i = 0; i < val_howmany; ++i)
                    buf[val_start+i*val_stride1] = recvBuffer[haloId][mpirecvBufferIndexBASE + i];
                mpiRecvBufferIndex += val_howmany;
            }

        }

    }

    inline void unpackRecvBuffer(const size_t haloId, const size_t bufferId, buf_t *buf,
                                 std::vector<size_t> overwriteHaloRecvIndices, std::vector<size_t> overwriteHaloTargetIndices) const {

        size_t haloSize = remoteHaloIndicesSize[haloId];

        for(size_t index = 0; index < overwriteHaloRecvIndices.size(); ++index)
            buf[overwriteHaloTargetIndices[index]] = recvBuffer[haloId][bufferId*haloSize + overwriteHaloRecvIndices[index]];

    }

    inline MPI_Request *packAndSend(const size_t haloId, const buf_t *buf, const int msgtag, const int remoteMpiRank = -1) const {
        packSendBuffer(haloId, 0, buf);
        return send(haloId, msgtag, remoteMpiRank);
    }

    inline void recvAndUnpack(const size_t haloId, buf_t *buf, const int msgtag, const int remoteMpiRank = -1) const {
        recv(haloId, msgtag, remoteMpiRank, true);
        unpackRecvBuffer(haloId, 0, buf);
    }



#ifdef TAUSCH_OPENCL

    inline int addLocalHaloInfoOCL(std::vector<int> haloIndices, const int numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {
        return addLocalHaloInfoOCL(extractHaloIndicesWithStride(haloIndices, true),
                                   numBuffers, remoteMpiRank, hints);
    }

    inline int addLocalHaloInfoOCL(std::vector<size_t> haloIndices, const int numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {
        return addLocalHaloInfoOCL(extractHaloIndicesWithStride(std::vector<int>(haloIndices.begin(), haloIndices.end()), true),
                                   numBuffers, remoteMpiRank, hints);
    }

    inline int addLocalHaloInfoOCL(std::vector<std::array<int, 4> > haloIndices, const int numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {

        int haloSize = 0;
        for(auto tuple : haloIndices)
            haloSize += tuple[1];

        try {

            if(haloSize == 0) {

                localHaloIndices.push_back({});
                localHaloIndicesSize.push_back(haloIndices.size());
                localHaloNumBuffers.push_back(numBuffers);
                localHaloRemoteMpiRank.push_back(remoteMpiRank);

                localOptHints.push_back(hints);

                sendBuffer.push_back(new buf_t[1]{});
                mpiSendRequests.push_back(new MPI_Request());
                setupMpiSend.push_back(false);

            } else {

                localHaloIndices.push_back(haloIndices);
                localHaloIndicesSize.push_back(haloSize);
                localHaloNumBuffers.push_back(numBuffers);
                localHaloRemoteMpiRank.push_back(remoteMpiRank);

                localOptHints.push_back(hints);

                sendBuffer.push_back(new buf_t[numBuffers*haloSize]{});
                mpiSendRequests.push_back(new MPI_Request());
                setupMpiSend.push_back(false);

            }

        } catch(cl::Error &e) {
            std::cout << "Tausch::addLocalHaloInfoOCL(): OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

        return sendBuffer.size()-1;

    }

    inline int addRemoteHaloInfoOCL(std::vector<int> haloIndices, const int numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {
        return addRemoteHaloInfoOCL(extractHaloIndicesWithStride(haloIndices, true),
                                    numBuffers, remoteMpiRank, hints);
    }

    inline int addRemoteHaloInfoOCL(std::vector<size_t> haloIndices, const int numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {
        return addRemoteHaloInfoOCL(extractHaloIndicesWithStride(std::vector<int>(haloIndices.begin(), haloIndices.end()), true),
                                    numBuffers, remoteMpiRank, hints);
    }

    inline int addRemoteHaloInfoOCL(std::vector<std::array<int, 4> > haloIndices, const int numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {

        int haloSize = 0;
        for(auto tuple : haloIndices)
            haloSize += tuple[1];

        try {

            if(haloSize == 0) {

                remoteHaloIndices.push_back({});
                remoteHaloIndicesSize.push_back(0);
                remoteHaloNumBuffers.push_back(numBuffers);
                remoteHaloRemoteMpiRank.push_back(remoteMpiRank);

                remoteOptHints.push_back(hints);

                recvBuffer.push_back(new buf_t[1]{});
                mpiRecvRequests.push_back(new MPI_Request());
                setupMpiRecv.push_back(false);

            } else {

                remoteHaloIndices.push_back(haloIndices);
                remoteHaloIndicesSize.push_back(haloSize);
                remoteHaloNumBuffers.push_back(numBuffers);
                remoteHaloRemoteMpiRank.push_back(remoteMpiRank);

                remoteOptHints.push_back(hints);

                recvBuffer.push_back(new buf_t[numBuffers*haloSize]{});
                mpiRecvRequests.push_back(new MPI_Request());
                setupMpiRecv.push_back(false);

            }

        } catch(cl::Error &e) {
            std::cout << "Tausch::addRemoteHaloInfoOCL(): OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

        return recvBuffer.size()-1;

    }

    void packSendBufferOCL(const int haloId, int bufferId, cl::Buffer buf) {

        try {

            const size_t haloSize = localHaloIndicesSize[haloId];

            size_t mpiSendBufferIndex = 0;
            for(size_t iRegion = 0; iRegion < localHaloIndices[haloId].size(); ++iRegion) {
                const std::array<int, 4> vals = localHaloIndices[haloId][iRegion];

                const int val_start = vals[0];
                const int val_howmany = vals[1];
                const int val_stride = vals[2];

                cl::size_t<3> buffer_offset;
                buffer_offset[0] = val_start*sizeof(buf_t); buffer_offset[1] = 0; buffer_offset[2] = 0;
                cl::size_t<3> host_offset;
                host_offset[0] = (bufferId*haloSize + mpiSendBufferIndex)*sizeof(buf_t); host_offset[1] = 0; host_offset[2] = 0;

                cl::size_t<3> region;
                region[0] = sizeof(buf_t); region[1] = val_howmany; region[2] = 1;

                queue.enqueueReadBufferRect(buf,
                                            CL_TRUE,
                                            buffer_offset,
                                            host_offset,
                                            region,
                                            val_stride*sizeof(buf_t),
                                            0,
                                            sizeof(buf_t),
                                            0,
                                            sendBuffer[haloId]);

                mpiSendBufferIndex += val_howmany;

            }

        } catch(cl::Error &e) {
            std::cerr << "Tausch::packSendBufferOCL(): OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

    }

    void packSendBufferOCL(const int haloId, int bufferId, cl::Buffer buf,
                           const std::vector<int> overwriteHaloSendIndices, const std::vector<int> overwriteHaloSourceIndices) {

        const size_t haloSize = localHaloIndicesSize[haloId];

        try {
            auto kernel_pack = cl::make_kernel
                                    <const cl::Buffer &, cl::Buffer &, const cl::Buffer &, const int &, const int &>
                                    (programs, "packSubRegion");

            cl::Buffer clHaloIndicesIn(context, overwriteHaloSourceIndices.begin(), overwriteHaloSourceIndices.end(), true);

            int globalsize = (overwriteHaloSourceIndices.size()/clKernelLocalSize +1)*clKernelLocalSize;

            cl::Buffer tmpSendBuffer_d(context, CL_MEM_READ_WRITE, overwriteHaloSourceIndices.size()*sizeof(buf_t));

            kernel_pack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                          buf, tmpSendBuffer_d, clHaloIndicesIn, overwriteHaloSourceIndices.size(), bufferId*haloSize);

            buf_t *tmpSendBuffer_h = new buf_t[overwriteHaloSourceIndices.size()];
            cl::copy(queue, tmpSendBuffer_d, &tmpSendBuffer_h[0], &tmpSendBuffer_h[overwriteHaloSourceIndices.size()]);

            for(size_t i = 0; i < overwriteHaloSendIndices.size(); ++i)
                sendBuffer[haloId][bufferId*haloSize + overwriteHaloSendIndices[i]] = tmpSendBuffer_h[i];

        } catch(cl::Error &e) {
            std::cerr << "Tausch::packSendBufferOCL(): OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

    }

    MPI_Request *sendOCL(const int haloId, const int msgtag, int remoteMpiRank = -1) {

        if(localHaloIndicesSize[haloId] == 0)
            return nullptr;

        if(!setupMpiSend[haloId]) {

            if(remoteMpiRank == -1)
                remoteMpiRank = localHaloRemoteMpiRank[haloId];

            // if we stay on the same rank, we don't need to use MPI
            int myRank;
            MPI_Comm_rank(TAUSCH_COMM, &myRank);
            if(remoteMpiRank == myRank) {
                msgtagToHaloId[myRank*1000000 + msgtag] = haloId;
                return nullptr;
            }

            setupMpiSend[haloId] = true;

            MPI_Send_init(&sendBuffer[haloId][0], localHaloNumBuffers[haloId]*localHaloIndicesSize[haloId],
                          mpiDataType, remoteMpiRank, msgtag, TAUSCH_COMM, mpiSendRequests[haloId]);

        } else
            MPI_Wait(mpiSendRequests[haloId], MPI_STATUS_IGNORE);

        MPI_Start(mpiSendRequests[haloId]);

        return mpiSendRequests[haloId];

    }

    void recvOCL(const int haloId, const int msgtag, int remoteMpiRank = -1) {

        if(remoteHaloIndicesSize[haloId] == 0)
            return;

        if(!setupMpiRecv[haloId]) {

            if(remoteMpiRank == -1)
                remoteMpiRank = remoteHaloRemoteMpiRank[haloId];

            // if we stay on the same rank, we don't need to use MPI
            int myRank;
            MPI_Comm_rank(TAUSCH_COMM, &myRank);

            if(remoteMpiRank == myRank) {

                const int remoteHaloId = msgtagToHaloId[myRank*1000000 + msgtag];

                memcpy(recvBuffer[haloId], sendBuffer[remoteHaloId], remoteHaloNumBuffers[haloId]*remoteHaloIndicesSize[haloId]*sizeof(buf_t));

            } else {

                setupMpiRecv[haloId] = true;

                MPI_Recv_init(&recvBuffer[haloId][0], remoteHaloNumBuffers[haloId]*remoteHaloIndicesSize[haloId], mpiDataType,
                              remoteMpiRank, msgtag, TAUSCH_COMM, mpiRecvRequests[haloId]);

            }

        }

        // this will remain false if we remained on the same mpi rank
        if(setupMpiRecv[haloId]) {

            MPI_Start(mpiRecvRequests[haloId]);
            MPI_Wait(mpiRecvRequests[haloId], MPI_STATUS_IGNORE);

        }

    }

    void unpackRecvBufferOCL(const int haloId, const int bufferId, cl::Buffer buf) {

        try {

            const size_t haloSize = remoteHaloIndicesSize[haloId];

            size_t mpiRecvBufferIndex = 0;
            for(size_t iRegion = 0; iRegion < remoteHaloIndices[haloId].size(); ++iRegion) {
                const std::array<int, 4> vals = remoteHaloIndices[haloId][iRegion];

                const int val_start = vals[0];
                const int val_howmany = vals[1];
                const int val_stride = vals[2];

                cl::size_t<3> buffer_offset;
                buffer_offset[0] = val_start*sizeof(buf_t); buffer_offset[1] = 0; buffer_offset[2] = 0;
                cl::size_t<3> host_offset;
                host_offset[0] = (bufferId*haloSize + mpiRecvBufferIndex)*sizeof(buf_t); host_offset[1] = 0; host_offset[2] = 0;

                cl::size_t<3> region;
                region[0] = sizeof(buf_t); region[1] = val_howmany; region[2] = 1;

                queue.enqueueWriteBufferRect(buf,
                                             CL_TRUE,
                                             buffer_offset,
                                             host_offset,
                                             region,
                                             val_stride*sizeof(buf_t),
                                             0,
                                             sizeof(buf_t),
                                             0,
                                             recvBuffer[haloId]);

                mpiRecvBufferIndex += val_howmany;

            }

        } catch(cl::Error &e) {
            std::cerr << "Tausch::unpackRecvBufferOCL() :: OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

    }

    void unpackRecvBufferOCL(const int haloId, int bufferId, cl::Buffer buf,
                          const std::vector<int> overwriteHaloRecvIndices, const std::vector<int> overwriteHaloTargetIndices) {

        const size_t haloSize = localHaloIndicesSize[haloId];

        try {

            auto kernel_unpack = cl::make_kernel
                                    <const cl::Buffer &, cl::Buffer &, const cl::Buffer &, const int &, const int &>
                                    (programs, "unpackSubRegion");

            cl::Buffer clHaloIndicesOut(context, overwriteHaloTargetIndices.begin(), overwriteHaloTargetIndices.end(), true);

            buf_t *tmpRecvBuffer_h = new buf_t[overwriteHaloTargetIndices.size()];
            for(size_t i = 0; i < overwriteHaloTargetIndices.size(); ++i)
                tmpRecvBuffer_h[i] = recvBuffer[haloId][bufferId*haloSize + overwriteHaloRecvIndices[i]];

            cl::Buffer tmpRecvBuffer_d(context, CL_MEM_READ_WRITE, overwriteHaloRecvIndices.size()*sizeof(buf_t));
            cl::copy(queue, &tmpRecvBuffer_h[0], &tmpRecvBuffer_h[overwriteHaloRecvIndices.size()], tmpRecvBuffer_d);

            int globalsize = (overwriteHaloRecvIndices.size()/clKernelLocalSize +1)*clKernelLocalSize;

            kernel_unpack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                          tmpRecvBuffer_d, buf, clHaloIndicesOut, overwriteHaloTargetIndices.size(), bufferId*haloSize);

        } catch(cl::Error &e) {
            std::cerr << "Tausch::unpackRecvBufferOCL() :: OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }
    }

    inline MPI_Request *packAndSendOCL(const int haloId, const cl::Buffer buf, const int msgtag, const int remoteMpiRank = -1) const {
        packSendBufferOCL(haloId, 0, buf);
        return sendOCL(haloId, msgtag, remoteMpiRank);
    }

    inline void recvAndUnpackOCL(const int haloId, cl::Buffer buf, const int msgtag, const int remoteMpiRank = -1) const {
        recvOCL(haloId, msgtag, remoteMpiRank);
        unpackRecvBuffer(haloId, 0, buf);
    }

#endif

#ifdef TAUSCH_CUDA

    inline int addLocalHaloInfoCUDA(std::vector<int> haloIndices,
                                    const size_t numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {
        return addLocalHaloInfoCUDA(extractHaloIndicesWithStride(haloIndices, true), numBuffers, remoteMpiRank, hints);
    }
    inline int addLocalHaloInfoCUDA(std::vector<size_t> haloIndices,
                                    const size_t numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {
        return addLocalHaloInfoCUDA(extractHaloIndicesWithStride(std::vector<int>(haloIndices.begin(), haloIndices.end()), true), remoteMpiRank, hints);
    }
    inline int addLocalHaloInfoCUDA(std::vector<std::array<int, 4> > haloIndices,
                                    const size_t numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {

        int haloSize = 0;
        for(auto tuple : haloIndices)
            haloSize += tuple[1];

        if(haloSize == 0) {

            localHaloIndices.push_back({});
            localHaloIndicesSize.push_back(0);
            localHaloNumBuffers.push_back(numBuffers);
            localHaloRemoteMpiRank.push_back(remoteMpiRank);

            localOptHints.push_back(hints);

            sendBuffer.push_back(new buf_t[1]{});
            mpiSendRequests.push_back(new MPI_Request());
            setupMpiSend.push_back(false);

        } else {

            localHaloIndices.push_back(haloIndices);
            localHaloIndicesSize.push_back(haloSize);
            localHaloNumBuffers.push_back(numBuffers);
            localHaloRemoteMpiRank.push_back(remoteMpiRank);

            localOptHints.push_back(hints);

            sendBuffer.push_back(new buf_t[numBuffers*haloSize]{});
            mpiSendRequests.push_back(new MPI_Request());
            setupMpiSend.push_back(false);

        }

        return sendBuffer.size()-1;

    }

    inline int addRemoteHaloInfoCUDA(std::vector<int> haloIndices,
                                     const int numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {
        return addRemoteHaloInfoCUDA(extractHaloIndicesWithStride(haloIndices, true), numBuffers, remoteMpiRank, hints);
    }

    inline int addRemoteHaloInfoCUDA(std::vector<size_t> haloIndices,
                                     const int numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {
        return addRemoteHaloInfoCUDA(extractHaloIndicesWithStride(std::vector<int>(haloIndices.begin(), haloIndices.end()), true), numBuffers, remoteMpiRank, hints);
    }

    inline int addRemoteHaloInfoCUDA(std::vector<std::array<int, 4> > haloIndices,
                                     const int numBuffers = 1, const int remoteMpiRank = -1, TauschOptimizationHint hints = TauschOptimizationHint::NoHints) {

        int haloSize = 0;
        for(auto tuple : haloIndices)
            haloSize += tuple[1];

        if(haloSize == 0) {

            remoteHaloIndices.push_back({});
            remoteHaloIndicesSize.push_back(0);
            remoteHaloNumBuffers.push_back(numBuffers);
            remoteHaloRemoteMpiRank.push_back(remoteMpiRank);

            remoteOptHints.push_back(hints);

            recvBuffer.push_back(new buf_t[1]{});
            mpiRecvRequests.push_back(new MPI_Request());
            setupMpiRecv.push_back(false);

        } else {

            remoteHaloIndices.push_back(haloIndices);
            remoteHaloIndicesSize.push_back(haloSize);
            remoteHaloNumBuffers.push_back(numBuffers);
            remoteHaloRemoteMpiRank.push_back(remoteMpiRank);

            remoteOptHints.push_back(hints);

            recvBuffer.push_back(new buf_t[numBuffers*haloSize]{});
            mpiRecvRequests.push_back(new MPI_Request());
            setupMpiRecv.push_back(false);

        }

        return recvBuffer.size()-1;

    }


    void packSendBufferCUDA(const int haloId, int bufferId, buf_t *buf_d) {

        const size_t haloSize = localHaloIndicesSize[haloId];

        if(localOptHints[haloId] == TauschOptimizationHint::StaysOnDevice) {

            if(sendCommunicationBufferKeptOnCuda.find(haloId) == sendCommunicationBufferKeptOnCuda.end()) {

                buf_t *cudabuf;
                cudaMalloc(&cudabuf, localHaloNumBuffers[haloId]*haloSize*sizeof(buf_t));
                sendCommunicationBufferKeptOnCuda[haloId] = cudabuf;

            }

            size_t mpiSendBufferIndex = 0;
            for(size_t region = 0; region < localHaloIndices[haloId].size(); ++region) {
                const std::array<int, 4> vals = localHaloIndices[haloId][region];

                const int val_start = vals[0];
                const int val_howmany = vals[1];
                const int val_stride = vals[2];

                cudaError_t err = cudaMemcpy2D(&sendCommunicationBufferKeptOnCuda[haloId][bufferId*haloSize + mpiSendBufferIndex], sizeof(buf_t),
                                               &buf_d[val_start], val_stride*sizeof(buf_t),
                                               sizeof(buf_t), val_howmany, cudaMemcpyDeviceToDevice);
                if(err != cudaSuccess)
                    std::cout << "Tausch::packSendBufferCUDA() 1: CUDA error detected: " << err << std::endl;

                mpiSendBufferIndex += val_howmany;

            }

            return;

        }

        size_t mpiSendBufferIndex = 0;
        for(size_t region = 0; region < localHaloIndices[haloId].size(); ++region) {
            const std::array<int, 4> vals = localHaloIndices[haloId][region];

            const int val_start = vals[0];
            const int val_howmany = vals[1];
            const int val_stride = vals[2];

            cudaError_t err = cudaMemcpy2D(&sendBuffer[haloId][bufferId*haloSize + mpiSendBufferIndex], sizeof(buf_t),
                                           &buf_d[val_start], val_stride*sizeof(buf_t),
                                           sizeof(buf_t), val_howmany, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                std::cout << "Tausch::packSendBufferCUDA() 2: CUDA error detected: " << err << std::endl;

            mpiSendBufferIndex += val_howmany;

        }

    }

    inline MPI_Request *sendCUDA(size_t haloId, const int msgtag, int remoteMpiRank = -1) {

        if(localHaloIndicesSize[haloId] == 0)
            return nullptr;

        if(!setupMpiSend[haloId]) {

            if(remoteMpiRank == -1)
                remoteMpiRank = localHaloRemoteMpiRank[haloId];

            // if we stay on the same rank, we don't need to use MPI
            int myRank;
            MPI_Comm_rank(TAUSCH_COMM, &myRank);
            if(remoteMpiRank == myRank) {
                msgtagToHaloId[myRank*1000000 + msgtag] = haloId;
                return nullptr;
            }

            setupMpiSend[haloId] = true;

            MPI_Send_init(&sendBuffer[haloId][0], localHaloNumBuffers[haloId]*localHaloIndicesSize[haloId], mpiDataType,
                          remoteMpiRank, msgtag, TAUSCH_COMM, mpiSendRequests[haloId]);

        } else
            MPI_Wait(mpiSendRequests[haloId], MPI_STATUS_IGNORE);

        MPI_Start(mpiSendRequests[haloId]);

        return mpiSendRequests[haloId];

    }

    inline MPI_Request *recvCUDA(size_t haloId, const int msgtag, int remoteMpiRank = -1, const bool blocking = true) {

        if(remoteHaloIndicesSize[haloId] == 0)
            return nullptr;

        if(!setupMpiRecv[haloId]) {

            if(remoteMpiRank == -1)
                remoteMpiRank = remoteHaloRemoteMpiRank[haloId];

            int myRank;
            MPI_Comm_rank(TAUSCH_COMM, &myRank);

            const int remoteHaloId = msgtagToHaloId[myRank*1000000 + msgtag];

            if(remoteMpiRank == myRank && remoteOptHints[haloId] == TauschOptimizationHint::StaysOnDevice) {

                buf_t *cudabuf;
                cudaMalloc(&cudabuf, remoteHaloNumBuffers[haloId]*remoteHaloIndicesSize[haloId]*sizeof(buf_t));
                cudaMemcpy(cudabuf, sendCommunicationBufferKeptOnCuda[remoteHaloId], remoteHaloNumBuffers[haloId]*remoteHaloIndicesSize[haloId]*sizeof(buf_t), cudaMemcpyDeviceToDevice);
                recvCommunicationBufferKeptOnCuda[haloId] = cudabuf;

            } else if(remoteMpiRank == myRank) {

                memcpy(recvBuffer[haloId], sendBuffer[remoteHaloId], remoteHaloNumBuffers[haloId]*remoteHaloIndicesSize[haloId]*sizeof(buf_t));

            } else {

                setupMpiRecv[haloId] = true;

                MPI_Recv_init(&recvBuffer[haloId][0], remoteHaloNumBuffers[haloId]*remoteHaloIndicesSize[haloId], mpiDataType,
                              remoteMpiRank, msgtag, TAUSCH_COMM, mpiRecvRequests[haloId]);

            }

        }

        // this will remain false if we remained on the same mpi rank
        if(setupMpiRecv[haloId]) {

            MPI_Start(mpiRecvRequests[haloId]);
            if(blocking)
                MPI_Wait(mpiRecvRequests[haloId], MPI_STATUS_IGNORE);

            return mpiRecvRequests[haloId];

        } else
            return nullptr;

    }

    inline void unpackRecvBufferCUDA(const size_t haloId, const size_t bufferId, buf_t *buf_d) {

        size_t haloSize = remoteHaloIndicesSize[haloId];

        if(remoteOptHints[haloId] == TauschOptimizationHint::StaysOnDevice) {

            size_t mpiRecvBufferIndex = 0;
            for(size_t region = 0; region < remoteHaloIndices[haloId].size(); ++region) {
                const std::array<int, 4> vals = remoteHaloIndices[haloId][region];

                const int val_start = vals[0];
                const int val_howmany = vals[1];
                const int val_stride = vals[2];

                cudaError_t err = cudaMemcpy2D(&buf_d[val_start], val_stride*sizeof(buf_t),
                                               &recvCommunicationBufferKeptOnCuda[haloId][bufferId*haloSize + mpiRecvBufferIndex], sizeof(buf_t),
                                               sizeof(buf_t), val_howmany, cudaMemcpyDeviceToDevice);
                if(err != cudaSuccess)
                    std::cout << "Tausch::unpackRecvBufferCUDA(): CUDA error detected: " << err << std::endl;

                mpiRecvBufferIndex += val_howmany;

            }

        } else {

            size_t mpiRecvBufferIndex = 0;
            for(size_t region = 0; region < remoteHaloIndices[haloId].size(); ++region) {
                const std::array<int, 4> vals = remoteHaloIndices[haloId][region];

                const int val_start = vals[0];
                const int val_howmany = vals[1];
                const int val_stride = vals[2];

                cudaError_t err = cudaMemcpy2D(&buf_d[val_start], val_stride*sizeof(buf_t),
                                               &recvBuffer[haloId][bufferId*haloSize + mpiRecvBufferIndex], sizeof(buf_t),
                                               sizeof(buf_t), val_howmany, cudaMemcpyHostToDevice);
                if(err != cudaSuccess)
                    std::cout << "Tausch::unpackRecvBufferCUDA(): CUDA error detected: " << err << std::endl;

                mpiRecvBufferIndex += val_howmany;

            }

        }

    }

#endif

private:

    inline std::vector<std::array<int, 4> > extractHaloIndicesWithStride(std::vector<int> indices, bool gpu = false) const {

        // special cases: 0, 1, 2 entries only

        if(indices.size() == 0)

            return std::vector<std::array<int, 4> >();

        else if(indices.size() == 1)

            return {{indices[0], 1, 1, 0}};

        else if(indices.size() == 2)

            return {{indices[0], 2, indices[1]-indices[0], 0}};

        // more than 2 entries

        std::vector<std::array<int, 4> > ret;

        // these keep track of our current position in the array and the nature of the current values
        int start = -1;
        int howMany = 0;
        int stride1 = 0;
        int stride2 = 0;

        // whichCase can be 1 or 2 for the two cases (see below)
        int whichCase = -1;

        // our index in the array
        int ind = 0;

        // we stop one before end, as we need at least two values to compute stride.
        // this leaves the possibility of a trailing entry at end of array -> taken care of at the end
        while(ind < indices.size()-1) {

            // compute three strides (based on four values
            int curStride0 = -1, curStride1 = -1, curStride2 = -1;

            curStride0 = indices[ind+1]-indices[ind];
            if(ind < indices.size()-2)
                curStride1 = indices[ind+2]-indices[ind+1];
            if(ind < indices.size()-3)
                curStride2 = indices[ind+3]-indices[ind+2];

            // halo width of 2 (stride pattern: 1, x, 1, x, 1, etc.
            if(!gpu && curStride0 == 1 && curStride1 > 1 && curStride2 == 1) {

                // new halo region
                if(start == -1) {

                    start = indices[ind];
                    howMany = 4;
                    stride1 = 1;
                    stride2 = curStride1;

                    ind += 4;

                // something currently stored in temp variables
                } else {

                    // we had a similar case before
                    if(whichCase == 1) {

                        // same case as before
                        if(indices[ind]-indices[ind-1] == stride2) {

                            howMany += 2;
                            ind += 2;

                            // and the next couple also fits the bill
                            if(curStride1 == stride2) {

                                howMany += 2;
                                ind += 2;

                            }


                        }

                    // before we had case #2
                    } else {

                        // store previous setup
                        ret.push_back({start, howMany, stride1, stride2});

                        // reset values for new case
                        start = indices[ind];
                        howMany = 4;
                        stride1 = 1;
                        stride2 = curStride1;

                        ind += 4;

                    }

                }

                // end of case #1
                whichCase = 1;

            } else {

                // new halo region
                if(start == -1) {

                    if(!gpu && ((curStride1 == 1 && curStride2 > 1) || curStride1 == curStride2)) {

                        ret.push_back({indices[ind], 1, 1, 0});

                        ind += 1;
                        howMany = 0;

                        continue;

                    } else {

                        // base values reset
                        start = indices[ind];
                        howMany = 0;
                        stride1 = curStride0;
                        stride2 = 0;

                    }

                } else {

                    // same case as before
                    if(whichCase == 2) {

                        // continuation from before
                        if(indices[ind]-indices[ind-1] == stride1) {

                            // only the first value fits the bill, the rest is different
                            if(curStride0 != stride1) {

                                howMany += 1;
                                ind += 1;

                                continue;

                                // to ease the complexity of this function, we let the next iteration take care of the other three values that we ignore here

                            }

                        // the stride has changed
                        } else {

                            // store previous setup
                            ret.push_back({start, howMany, stride1, stride2});

                            // base values reset
                            start = indices[ind];
                            howMany = 0;
                            stride1 = curStride0;
                            stride2 = 0;

                        }

                    // before we had case #1
                    } else {

                        // store previous setup
                        ret.push_back({start, howMany, stride1, stride2});

                        // base values reset
                        start = indices[ind];
                        howMany = 0;
                        stride1= curStride0;
                        stride2 = 0;

                    }

                }

                // four values that have the same stride
                if(curStride0 == curStride1 && curStride1 == curStride2 && curStride1 != -1 && curStride2 != -1) {

                    howMany += 4;
                    ind += 4;

                // three values
                } else if(curStride0 == curStride1 && curStride1 != -1) {

                    howMany += 3;
                    ind += 3;

                // only two values
                } else {

                    // there might be a pattern up ahead -> add current value as single entry!
                    if(!gpu && ((curStride1 == 1 && curStride2 > 1) || curStride1 == curStride2)) {

                        // store previous setup
                        ret.push_back({start, howMany, stride1, stride2});

                        // only move pointer by 1
                        ind += 1;
                        start = -1;
                        stride1 = curStride1;
                        stride2 = 0;

                    // no pattern coming up right away (afawct)
                    } else {

                        howMany += 2;
                        ind += 2;

                    }

                }

                // end of case #2
                whichCase = 2;

            }

        }

        // if the last value fits the previous setup, add to it
        if(ind == indices.size()-1 && howMany > 0 && stride2 == 0 && stride1 == indices[ind]-indices[ind-1]) {
            ++howMany;
            ++ind;
        }

        // store final setup (if something is left)
        if(howMany > 0)
            ret.push_back({start, howMany, stride1, stride2});

        // possibly trailing entry at end -> add as single entry
        if(ind == indices.size()-1)
            ret.push_back({indices[ind], 1, 1, 0});

        return ret;

    }

    MPI_Comm TAUSCH_COMM;
    MPI_Datatype mpiDataType;

    std::vector<std::vector<std::array<int, 4> > > localHaloIndices;
    std::vector<std::vector<std::array<int, 4> > > remoteHaloIndices;

    std::vector<size_t> localHaloIndicesSize;
    std::vector<size_t> remoteHaloIndicesSize;

    std::vector<int> localHaloRemoteMpiRank;
    std::vector<int> remoteHaloRemoteMpiRank;

    std::vector<size_t> localHaloNumBuffers;
    std::vector<size_t> remoteHaloNumBuffers;

    std::vector<buf_t*> sendBuffer;
    std::vector<buf_t*> recvBuffer;

    std::vector<MPI_Request*> mpiSendRequests;
    std::vector<MPI_Request*> mpiRecvRequests;

    std::vector<bool> setupMpiSend;
    std::vector<bool> setupMpiRecv;

    std::vector<TauschOptimizationHint> localOptHints;
    std::vector<TauschOptimizationHint> remoteOptHints;


    // this is used for exchanges on same mpi rank
    std::map<int, int> msgtagToHaloId;

#ifdef TAUSCH_OPENCL

    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program programs;
    int clKernelLocalSize;
    std::string cName4BufT;

#endif

#ifdef TAUSCH_CUDA
    std::map<int, buf_t*> sendCommunicationBufferKeptOnCuda;
    std::map<int, buf_t*> recvCommunicationBufferKeptOnCuda;
#endif

};


#endif
