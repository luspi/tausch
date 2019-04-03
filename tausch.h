#ifndef TAUSCH_H
#define TAUSCH_H

//#define TAUSCH_CUDA

#include <mpi.h>
#include <vector>

#ifdef TAUSCH_OPENCL
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#endif

#ifdef TAUSCH_CUDA
#include <cuda_runtime.h>
#endif

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

kernel void pack(global const buf_t * restrict inBuf, global buf_t * restrict outBuf,
                 global const int * restrict inIndices, const int numIndices, const int bufferId) {

    int gid = get_global_id(0);

    if(gid < numIndices)
        outBuf[bufferId*numIndices + gid] = inBuf[inIndices[gid]];

}

kernel void packSubRegion(global const buf_t * restrict inBuf, global buf_t * restrict outBuf,
                          global const int * restrict inIndices, global const int * restrict outIndices,
                          const int numIndices, const int bufferId) {

    int gid = get_global_id(0);

    if(gid < numIndices)
        outBuf[bufferId*numIndices + outIndices[gid]] = inBuf[inIndices[gid]];

}

kernel void unpack(global const buf_t * restrict const inBuf,
                   global buf_t * restrict const outBuf,
                   global const int * restrict const outIndices,
                   const int numIndices, const int bufferId) {

    int gid = get_global_id(0);

    if(gid < numIndices)
        outBuf[outIndices[gid]] = inBuf[bufferId*numIndices + gid];

}

kernel void unpackSubRegion(global const buf_t * restrict inBuf, global buf_t * restrict outBuf,
                            global const int * restrict inIndices, global const int * restrict outIndices,
                            const int numIndices, const int bufferId) {

    int gid = get_global_id(0);

    if(gid < numIndices)
      outBuf[outIndices[gid]] = inBuf[bufferId*numIndices + inIndices[gid]];

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

    inline int addLocalHaloInfo(std::vector<int> haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1) {
        return addLocalHaloInfo(extractHaloIndicesWithStride(haloIndices), numBuffers, remoteMpiRank);
    }

    inline int addLocalHaloInfo(std::vector<size_t> haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1) {
        return addLocalHaloInfo(extractHaloIndicesWithStride(std::vector<int>(haloIndices.begin(), haloIndices.end())),
                                numBuffers, remoteMpiRank);
    }

    inline int addLocalHaloInfo(std::vector<std::array<int, 3> > haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1) {

        localHaloIndicesCPU.push_back(haloIndices);
        localHaloRemoteMpiRankCPU.push_back(remoteMpiRank);
        localHaloNumBuffersCPU.push_back(numBuffers);

        size_t bufsize = 0;
        for(size_t i = 0; i < haloIndices.size(); ++i)
            bufsize += static_cast<size_t>(haloIndices.at(i)[1]);

        localHaloIndicesSizeCPU.push_back(bufsize);

        void *newbuf = NULL;
        posix_memalign(&newbuf, 64, numBuffers*bufsize*sizeof(buf_t));
        buf_t *newbuf_buft = reinterpret_cast<buf_t*>(newbuf);
        double zero = 0;
        std::fill_n(newbuf_buft, numBuffers*bufsize, zero);
        sendBufferCPU.push_back(newbuf_buft);

        mpiSendRequestsCPU.push_back(new MPI_Request());

        setupMpiSendCPU.push_back(false);

        return sendBufferCPU.size()-1;

    }


    inline int addRemoteHaloInfo(std::vector<int> haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1) {
        return addRemoteHaloInfo(extractHaloIndicesWithStride(haloIndices), numBuffers, remoteMpiRank);
    }

    inline int addRemoteHaloInfo(std::vector<size_t> haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1) {
        return addRemoteHaloInfo(extractHaloIndicesWithStride(std::vector<int>(haloIndices.begin(), haloIndices.end())),
                                 numBuffers, remoteMpiRank);
    }

    inline int addRemoteHaloInfo(std::vector<std::array<int, 3> > haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1) {

        remoteHaloIndicesCPU.push_back(haloIndices);
        remoteHaloRemoteMpiRankCPU.push_back(remoteMpiRank);
        remoteHaloNumBuffersCPU.push_back(numBuffers);

        size_t bufsize = 0;
        for(size_t i = 0; i < haloIndices.size(); ++i)
            bufsize += static_cast<size_t>(haloIndices.at(i)[1]);
        remoteHaloIndicesSizeCPU.push_back(bufsize);

        void *newbuf = NULL;
        posix_memalign(&newbuf, 64, numBuffers*bufsize*sizeof(buf_t));
        buf_t *newbuf_buft = reinterpret_cast<buf_t*>(newbuf);
        double zero = 0;
        std::fill_n(newbuf_buft, numBuffers*bufsize, zero);
        recvBufferCPU.push_back(newbuf_buft);

        mpiRecvRequestsCPU.push_back(new MPI_Request());

        setupMpiRecvCPU.push_back(false);

        return recvBufferCPU.size()-1;

    }

    inline void packSendBuffer(const size_t haloId, const size_t bufferId, const buf_t *buf) const {

        const size_t haloSize = localHaloIndicesSizeCPU[haloId];

        size_t mpiSendBufferIndex = 0;
        for(size_t region = 0; region < localHaloIndicesCPU[haloId].size(); ++region) {
            const std::array<int, 3> vals = localHaloIndicesCPU[haloId][region];

            const int val_start = vals[0];
            const int val_howmany = vals[1];
            const int val_stride = vals[2];

            if(val_stride == 1) {
                memcpy(&sendBufferCPU[haloId][bufferId*haloSize + mpiSendBufferIndex], &buf[val_start], val_howmany*sizeof(buf_t));
                mpiSendBufferIndex += val_howmany;
            } else {
                const int mpiSendBufferIndexBASE = bufferId*haloSize + mpiSendBufferIndex;
                for(int i = 0; i < val_howmany; ++i)
                    sendBufferCPU[haloId][mpiSendBufferIndexBASE + i] = buf[val_start+i*val_stride];
                mpiSendBufferIndex += val_howmany;
            }

        }

    }

    inline void packSendBuffer(const size_t haloId, const size_t bufferId, const buf_t *buf,
                               std::vector<size_t> overwriteHaloSendIndices, std::vector<size_t> overwriteHaloSourceIndices) const {

        size_t haloSize = localHaloIndicesSizeCPU[haloId];

        for(size_t index = 0; index < overwriteHaloSendIndices.size(); ++index)
            sendBufferCPU[haloId][bufferId*haloSize + overwriteHaloSendIndices[index]] = buf[overwriteHaloSourceIndices[index]];

    }

    inline MPI_Request *send(size_t haloId, const int msgtag, int remoteMpiRank = -1) {

        if(localHaloIndicesCPU[haloId].size() == 0)
            return nullptr;

        if(!setupMpiSendCPU[haloId]) {

            setupMpiSendCPU[haloId] = true;

            if(remoteMpiRank == -1)
                remoteMpiRank = localHaloRemoteMpiRankCPU[haloId];

            MPI_Send_init(&sendBufferCPU[haloId][0], localHaloNumBuffersCPU[haloId]*localHaloIndicesSizeCPU[haloId], mpiDataType, remoteMpiRank,
                      msgtag, TAUSCH_COMM, mpiSendRequestsCPU[haloId]);

        } else
            MPI_Wait(mpiSendRequestsCPU[haloId], MPI_STATUS_IGNORE);

        MPI_Start(mpiSendRequestsCPU[haloId]);

        return mpiSendRequestsCPU[haloId];

    }

    inline MPI_Request *recv(size_t haloId, const int msgtag, int remoteMpiRank = -1, const bool blocking = true) {

        if(remoteHaloIndicesCPU[haloId].size() == 0)
            return nullptr;

        if(!setupMpiRecvCPU[haloId]) {

            setupMpiRecvCPU[haloId] = true;

            if(remoteMpiRank == -1)
                remoteMpiRank = remoteHaloRemoteMpiRankCPU[haloId];

            MPI_Recv_init(&recvBufferCPU[haloId][0], remoteHaloNumBuffersCPU[haloId]*remoteHaloIndicesSizeCPU[haloId], mpiDataType,
                          remoteMpiRank, msgtag, TAUSCH_COMM, mpiRecvRequestsCPU[haloId]);

        }

        MPI_Start(mpiRecvRequestsCPU[haloId]);
        if(blocking)
            MPI_Wait(mpiRecvRequestsCPU[haloId], MPI_STATUS_IGNORE);

        return mpiRecvRequestsCPU[haloId];

    }

    inline void unpackRecvBuffer(const size_t haloId, const size_t bufferId, buf_t *buf) const {

        size_t haloSize = remoteHaloIndicesSizeCPU[haloId];

        size_t mpiRecvBufferIndex = 0;
        for(size_t region = 0; region < remoteHaloIndicesCPU[haloId].size(); ++region) {
            const std::array<int, 3> vals = remoteHaloIndicesCPU[haloId][region];

            const int val_start = vals[0];
            const int val_howmany = vals[1];
            const int val_stride = vals[2];

            if(val_stride == 1) {
                memcpy(&buf[val_start], &recvBufferCPU[haloId][bufferId*haloSize + mpiRecvBufferIndex], val_howmany*sizeof(buf_t));
                mpiRecvBufferIndex += val_howmany;
            } else {
                const size_t mpirecvBufferIndexBASE = bufferId*haloSize + mpiRecvBufferIndex;
                for(int i = 0; i < val_howmany; ++i)
                    buf[val_start+i*val_stride] = recvBufferCPU[haloId][mpirecvBufferIndexBASE + i];
                mpiRecvBufferIndex += val_howmany;
            }

        }

    }

    inline void unpackRecvBuffer(const size_t haloId, const size_t bufferId, buf_t *buf,
                                 std::vector<size_t> overwriteHaloRecvIndices, std::vector<size_t> overwriteHaloTargetIndices) const {

        size_t haloSize = remoteHaloIndicesSizeCPU[haloId];

        for(size_t index = 0; index < overwriteHaloRecvIndices.size(); ++index)
            buf[overwriteHaloTargetIndices[index]] = recvBufferCPU[haloId][bufferId*haloSize + overwriteHaloRecvIndices[index]];

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

    inline int addLocalHaloInfoOCL(std::vector<std::array<int, 3> > haloIndices, const int numBuffers = 1, const int remoteMpiRank = -1) {
        std::vector<int> indices;
        for(auto tuple : haloIndices) {
            for(int i = 0; i < tuple[1]; ++i)
                indices.push_back(tuple[0]+i*tuple[2]);
        }
        return addLocalHaloInfoOCL(indices, numBuffers, remoteMpiRank);
    }

    inline int addLocalHaloInfoOCL(std::vector<size_t> haloIndices, const int numBuffers = 1, const int remoteMpiRank = -1) {
        return addLocalHaloInfoOCL(std::vector<int>(haloIndices.begin(), haloIndices.end()), numBuffers, remoteMpiRank);
    }

    inline int addLocalHaloInfoOCL(std::vector<int> haloIndices, const int numBuffers = 1, const int remoteMpiRank = -1) {

        try {

            if(haloIndices.size() == 0) {

                cl::Buffer clHaloIndices(context, CL_MEM_READ_WRITE, sizeof(buf_t));
                localHaloIndicesOCL_d.push_back(clHaloIndices);
                localHaloIndicesOCL_h.push_back({});

                localHaloIndicesSizeOCL.push_back(haloIndices.size());

                localHaloNumBuffersOCL.push_back(numBuffers);

                localHaloRemoteMpiRankOCL.push_back(remoteMpiRank);

                cl::Buffer clRecvBuffer(context, CL_MEM_READ_WRITE, sizeof(buf_t));
                sendBufferOCL_d.push_back(clRecvBuffer);
                sendBufferOCL_h.push_back(new buf_t[1]{});

                mpiSendRequestsOCL.push_back(new MPI_Request());

                setupMpiSendOCL.push_back(false);

            } else {

                cl::Buffer clHaloIndices(context, haloIndices.begin(), haloIndices.end(), true);
                localHaloIndicesOCL_d.push_back(clHaloIndices);
                localHaloIndicesOCL_h.push_back(haloIndices);

                localHaloIndicesSizeOCL.push_back(haloIndices.size());

                localHaloNumBuffersOCL.push_back(numBuffers);

                localHaloRemoteMpiRankOCL.push_back(remoteMpiRank);

                cl::Buffer clSendBuffer(context, CL_MEM_READ_WRITE, numBuffers*haloIndices.size()*sizeof(buf_t));
                sendBufferOCL_d.push_back(clSendBuffer);
                sendBufferOCL_h.push_back(new buf_t[numBuffers*haloIndices.size()]{});

                mpiSendRequestsOCL.push_back(new MPI_Request());

                setupMpiSendOCL.push_back(false);

            }

        } catch(cl::Error &e) {
            std::cout << "Tausch::addLocalHaloInfoOCL(): OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

        return sendBufferOCL_d.size()-1;

    }

    inline int addRemoteHaloInfoOCL(std::vector<std::array<int, 3> > haloIndices, const int numBuffers = 1, const int remoteMpiRank = -1) {
        std::vector<int> indices;
        for(auto tuple : haloIndices) {
            for(int i = 0; i < tuple[1]; ++i)
                indices.push_back(tuple[0]+i*tuple[2]);
        }
        return addRemoteHaloInfoOCL(indices, numBuffers, remoteMpiRank);
    }

    inline int addRemoteHaloInfoOCL(std::vector<size_t> haloIndices, const int numBuffers = 1, const int remoteMpiRank = -1) {
        return addRemoteHaloInfoOCL(std::vector<int>(haloIndices.begin(), haloIndices.end()), numBuffers, remoteMpiRank);
    }

    inline int addRemoteHaloInfoOCL(std::vector<int> haloIndices, const int numBuffers = 1, const int remoteMpiRank = -1) {

        try {

            if(haloIndices.size() == 0) {

                cl::Buffer clHaloIndices(context, CL_MEM_READ_WRITE, sizeof(buf_t));
                remoteHaloIndicesOCL_d.push_back(clHaloIndices);
                remoteHaloIndicesOCL_h.push_back({});

                remoteHaloIndicesSizeOCL.push_back(haloIndices.size());

                remoteHaloNumBuffersOCL.push_back(numBuffers);

                remoteHaloRemoteMpiRankOCL.push_back(remoteMpiRank);

                cl::Buffer clRecvBuffer(context, CL_MEM_READ_WRITE, sizeof(buf_t));
                recvBufferOCL_d.push_back(clRecvBuffer);
                recvBufferOCL_h.push_back(new buf_t[1]{});

                mpiRecvRequestsOCL.push_back(new MPI_Request());

                setupMpiRecvOCL.push_back(false);

            } else {

                cl::Buffer clHaloIndices(context, haloIndices.begin(), haloIndices.end(), true);
                remoteHaloIndicesOCL_d.push_back(clHaloIndices);
                remoteHaloIndicesOCL_h.push_back(haloIndices);

                remoteHaloIndicesSizeOCL.push_back(haloIndices.size());

                remoteHaloNumBuffersOCL.push_back(numBuffers);

                cl::Buffer clRecvBuffer(context, CL_MEM_READ_WRITE, numBuffers*haloIndices.size()*sizeof(buf_t));
                recvBufferOCL_d.push_back(clRecvBuffer);
                recvBufferOCL_h.push_back(new buf_t[numBuffers*haloIndices.size()]{});

                mpiRecvRequestsOCL.push_back(new MPI_Request());

                setupMpiRecvOCL.push_back(false);

            }

        } catch(cl::Error &e) {
            std::cout << "Tausch::addRemoteHaloInfoOCL(): OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

        return recvBufferOCL_d.size()-1;

    }

    void packSendBufferOCL(const int haloId, int bufferId, cl::Buffer buf) {

        try {
            auto kernel_pack = cl::make_kernel
                                    <const cl::Buffer &, cl::Buffer &, const cl::Buffer &, const int &, const int &>
                                    (programs, "pack");

            int globalsize = (localHaloIndicesSizeOCL[haloId]/clKernelLocalSize +1)*clKernelLocalSize;

            kernel_pack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                          buf, sendBufferOCL_d[haloId], localHaloIndicesOCL_d[haloId], localHaloIndicesSizeOCL[haloId], bufferId);

        } catch(cl::Error &e) {
            std::cerr << "Tausch::packSendBufferOCL(): OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

    }

    void packSendBufferOCL(const int haloId, int bufferId, cl::Buffer buf,
                           const std::vector<int> overwriteHaloSendIndices, const std::vector<int> overwriteHaloSourceIndices) {

        try {
            auto kernel_pack = cl::make_kernel
                                    <const cl::Buffer &, cl::Buffer &, const cl::Buffer &, const cl::Buffer &, const int &, const int &>
                                    (programs, "packSubRegion");

            cl::Buffer clHaloIndicesIn(context, overwriteHaloSourceIndices.begin(), overwriteHaloSourceIndices.end(), true);
            cl::Buffer clHaloIndicesOut(context, overwriteHaloSendIndices.begin(), overwriteHaloSendIndices.end(), true);

            int globalsize = (localHaloIndicesSizeOCL[haloId]/clKernelLocalSize +1)*clKernelLocalSize;

            kernel_pack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                          buf, sendBufferOCL_d[haloId], clHaloIndicesIn, clHaloIndicesOut, localHaloIndicesSizeOCL[haloId], bufferId);

        } catch(cl::Error &e) {
            std::cerr << "Tausch::packSendBufferOCL(): OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

    }

    MPI_Request *sendOCL(const int haloId, const int msgtag, int remoteMpiRank = -1) {

        if(localHaloIndicesSizeOCL[haloId] == 0)
            return nullptr;

        if(!setupMpiSendOCL[haloId]) {

            setupMpiSendOCL[haloId] = true;

            if(remoteMpiRank == -1)
                remoteMpiRank = localHaloRemoteMpiRankOCL[haloId];

            MPI_Send_init(&sendBufferOCL_h[haloId][0], localHaloNumBuffersOCL[haloId]*localHaloIndicesSizeOCL[haloId],
                          mpiDataType, remoteMpiRank, msgtag, TAUSCH_COMM, mpiSendRequestsOCL[haloId]);

        } else
            MPI_Wait(mpiSendRequestsOCL[haloId], MPI_STATUS_IGNORE);

        try {
            cl::copy(queue, sendBufferOCL_d[haloId], &sendBufferOCL_h[haloId][0],
                    &sendBufferOCL_h[haloId][localHaloNumBuffersOCL[haloId]*localHaloIndicesSizeOCL[haloId]]);
        } catch(cl::Error &e) {
            std::cerr << "Tausch::sendOCL(): OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

        MPI_Start(mpiSendRequestsOCL[haloId]);

        return mpiSendRequestsOCL[haloId];

    }

    void recvOCL(const int haloId, const int msgtag, int remoteMpiRank = -1) {

        if(remoteHaloIndicesSizeOCL[haloId] == 0)
            return;

        if(!setupMpiRecvOCL[haloId]) {

            setupMpiRecvOCL[haloId] = true;

            if(remoteMpiRank == -1)
                remoteMpiRank = remoteHaloRemoteMpiRankOCL[haloId];

            MPI_Recv_init(&recvBufferOCL_h[haloId][0], remoteHaloNumBuffersOCL[haloId]*remoteHaloIndicesSizeOCL[haloId], mpiDataType,
                          remoteMpiRank, msgtag, TAUSCH_COMM, mpiRecvRequestsOCL[haloId]);

        }

        MPI_Start(mpiRecvRequestsOCL[haloId]);
        MPI_Wait(mpiRecvRequestsOCL[haloId], MPI_STATUS_IGNORE);

        try {
            cl::copy(queue, &recvBufferOCL_h[haloId][0],
                     &recvBufferOCL_h[haloId][remoteHaloNumBuffersOCL[haloId]*remoteHaloIndicesSizeOCL[haloId]], recvBufferOCL_d[haloId]);
        } catch(cl::Error &e) {
            std::cerr << "Tausch::recvOCL(): OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

    }

    void unpackRecvBufferOCL(const int haloId, const int bufferId, cl::Buffer buf) {

        try {
            auto kernel_unpack = cl::make_kernel
                                    <const cl::Buffer &, cl::Buffer &, const cl::Buffer &, const int &, const int &>
                                    (programs, "unpack");

            int globalsize = (remoteHaloIndicesSizeOCL[haloId]/clKernelLocalSize +1)*clKernelLocalSize;

            kernel_unpack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                          recvBufferOCL_d[haloId], buf, remoteHaloIndicesOCL_d[haloId], remoteHaloIndicesSizeOCL[haloId], bufferId);

        } catch(cl::Error &e) {
            std::cerr << "Tausch::unpackRecvBufferOCL() :: OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

    }

    void unpackRecvBufferOCL(const int haloId, int bufferId, cl::Buffer buf,
                          const std::vector<int> overwriteHaloRecvIndices, const std::vector<int> overwriteHaloTargetIndices) {

        try {

            auto kernel_unpack = cl::make_kernel
                                    <const cl::Buffer &, cl::Buffer &, const cl::Buffer &, const cl::Buffer &, const int &, const int &>
                                    (programs, "unpackSubRegion");

            cl::Buffer clHaloIndicesIn(context, overwriteHaloRecvIndices.begin(), overwriteHaloRecvIndices.end(), true);
            cl::Buffer clHaloIndicesOut(context, overwriteHaloTargetIndices.begin(), overwriteHaloTargetIndices.end(), true);

            int globalsize = (remoteHaloIndicesSizeOCL[haloId]/clKernelLocalSize +1)*clKernelLocalSize;

            kernel_unpack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                          recvBufferOCL_d[haloId], buf, clHaloIndicesIn, clHaloIndicesOut, remoteHaloIndicesSizeOCL[haloId], bufferId);

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
                                    const size_t numBuffers = 1, const int remoteMpiRank = -1) {
        return addLocalHaloInfoCUDA(extractHaloIndicesWithStride(haloIndices), numBuffers, remoteMpiRank);
    }
    inline int addLocalHaloInfoCUDA(std::vector<size_t> haloIndices,
                                    const size_t numBuffers = 1, const int remoteMpiRank = -1) {
        return addLocalHaloInfoCUDA(extractHaloIndicesWithStride(std::vector<int>(haloIndices.begin(), haloIndices.end())), remoteMpiRank);
    }
    inline int addLocalHaloInfoCUDA(std::vector<std::array<int, 3> > haloIndices,
                                    const size_t numBuffers = 1, const int remoteMpiRank = -1) {

        if(haloIndices.size() == 0) {

            localHaloIndicesCUDA_h.push_back({});

            localHaloIndicesSizeCUDA.push_back(0);

            localHaloNumBuffersCUDA.push_back(numBuffers);

            localHaloRemoteMpiRankCUDA.push_back(remoteMpiRank);

            buf_t *recvBuffer_d;
            cudaMalloc(&recvBuffer_d, sizeof(buf_t));
            sendBufferCUDA_d.push_back(recvBuffer_d);
            sendBufferCUDA_h.push_back(new buf_t[1]{});

            mpiSendRequestsCUDA.push_back(new MPI_Request());

            setupMpiSendCUDA.push_back(false);

        } else {

            size_t bufsize = 0;
            for(size_t i = 0; i < haloIndices.size(); ++i)
                bufsize += static_cast<size_t>(haloIndices.at(i)[1]);

            localHaloIndicesCUDA_h.push_back(haloIndices);

            localHaloIndicesSizeCUDA.push_back(bufsize);

            localHaloNumBuffersCUDA.push_back(numBuffers);

            localHaloRemoteMpiRankCUDA.push_back(remoteMpiRank);

            buf_t *sendBuffer_d;
            cudaMalloc(&sendBuffer_d, numBuffers*bufsize*sizeof(buf_t));
            sendBufferCUDA_d.push_back(sendBuffer_d);
            sendBufferCUDA_h.push_back(new buf_t[numBuffers*bufsize]{});

            mpiSendRequestsCUDA.push_back(new MPI_Request());

            setupMpiSendCUDA.push_back(false);

        }

        return sendBufferCUDA_d.size()-1;

    }

    inline int addRemoteHaloInfoCUDA(std::vector<int> haloIndices,
                                     const int numBuffers = 1, const int remoteMpiRank = -1) {
        return addRemoteHaloInfoCUDA(extractHaloIndicesWithStride(haloIndices), numBuffers, remoteMpiRank);
    }

    inline int addRemoteHaloInfoCUDA(std::vector<size_t> haloIndices,
                                     const int numBuffers = 1, const int remoteMpiRank = -1) {
        return addRemoteHaloInfoCUDA(extractHaloIndicesWithStride(std::vector<int>(haloIndices.begin(), haloIndices.end())), numBuffers, remoteMpiRank);
    }

    inline int addRemoteHaloInfoCUDA(std::vector<std::array<int, 3> > haloIndices,
                                     const int numBuffers = 1, const int remoteMpiRank = -1) {

        if(haloIndices.size() == 0) {

            remoteHaloIndicesCUDA_h.push_back({});

            remoteHaloIndicesSizeCUDA.push_back(0);

            remoteHaloNumBuffersCUDA.push_back(numBuffers);

            remoteHaloRemoteMpiRankCUDA.push_back(remoteMpiRank);

            buf_t *recvBuffer_d;
            cudaMalloc(&recvBuffer_d, sizeof(buf_t));
            recvBufferCUDA_d.push_back(recvBuffer_d);
            recvBufferCUDA_h.push_back(new buf_t[1]{});

            mpiRecvRequestsCUDA.push_back(new MPI_Request());

            setupMpiRecvCUDA.push_back(false);

        } else {

            size_t bufsize = 0;
            for(size_t i = 0; i < haloIndices.size(); ++i)
                bufsize += static_cast<size_t>(haloIndices.at(i)[1]);

            remoteHaloIndicesCUDA_h.push_back(haloIndices);

            remoteHaloIndicesSizeCUDA.push_back(bufsize);

            remoteHaloNumBuffersCUDA.push_back(numBuffers);

            remoteHaloRemoteMpiRankCUDA.push_back(remoteMpiRank);

            buf_t *recvBuffer_d;
            cudaMalloc(&recvBuffer_d, numBuffers*bufsize*sizeof(buf_t));
            recvBufferCUDA_d.push_back(recvBuffer_d);
            recvBufferCUDA_h.push_back(new buf_t[numBuffers*bufsize]{});

            mpiRecvRequestsCUDA.push_back(new MPI_Request());

            setupMpiRecvCUDA.push_back(false);

        }

        return recvBufferCUDA_d.size()-1;

    }


    void packSendBufferCUDA(const int haloId, int bufferId, buf_t *buf_d) {

        const size_t haloSize = localHaloIndicesSizeCUDA[haloId];

        size_t mpiSendBufferIndex = 0;
        for(size_t region = 0; region < localHaloIndicesCUDA_h[haloId].size(); ++region) {
            const std::array<int, 3> vals = localHaloIndicesCUDA_h[haloId][region];

            const int val_start = vals[0];
            const int val_howmany = vals[1];
            const int val_stride = vals[2];

            cudaMemcpy2D(&sendBufferCUDA_h[haloId][bufferId*haloSize + mpiSendBufferIndex], sizeof(buf_t),
                         &buf_d[val_start], val_stride*sizeof(buf_t),
                         sizeof(buf_t), val_howmany, cudaMemcpyDeviceToHost);
            mpiSendBufferIndex += val_howmany;

        }

    }

    inline MPI_Request *sendCUDA(size_t haloId, const int msgtag, int remoteMpiRank = -1) {

        if(localHaloIndicesSizeCUDA[haloId] == 0)
            return nullptr;

        if(!setupMpiSendCUDA[haloId]) {

            setupMpiSendCUDA[haloId] = true;

            if(remoteMpiRank == -1)
                remoteMpiRank = localHaloRemoteMpiRankCUDA[haloId];

            MPI_Send_init(&sendBufferCUDA_h[haloId][0], localHaloNumBuffersCUDA[haloId]*localHaloIndicesSizeCUDA[haloId], mpiDataType,
                          remoteMpiRank, msgtag, TAUSCH_COMM, mpiSendRequestsCUDA[haloId]);

        } else
            MPI_Wait(mpiSendRequestsCUDA[haloId], MPI_STATUS_IGNORE);

        MPI_Start(mpiSendRequestsCUDA[haloId]);

        return mpiSendRequestsCUDA[haloId];

    }

    inline MPI_Request *recvCUDA(size_t haloId, const int msgtag, int remoteMpiRank = -1, const bool blocking = true) {

        if(remoteHaloIndicesSizeCUDA[haloId] == 0)
            return nullptr;

        if(!setupMpiRecvCUDA[haloId]) {

            setupMpiRecvCUDA[haloId] = true;

            if(remoteMpiRank == -1)
                remoteMpiRank = remoteHaloRemoteMpiRankCUDA[haloId];

            MPI_Recv_init(&recvBufferCUDA_h[haloId][0], remoteHaloNumBuffersCUDA[haloId]*remoteHaloIndicesSizeCUDA[haloId], mpiDataType,
                          remoteMpiRank, msgtag, TAUSCH_COMM, mpiRecvRequestsCUDA[haloId]);

        }

        MPI_Start(mpiRecvRequestsCUDA[haloId]);
        if(blocking)
            MPI_Wait(mpiRecvRequestsCUDA[haloId], MPI_STATUS_IGNORE);

        return mpiRecvRequestsCUDA[haloId];

    }

    inline void unpackRecvBufferCUDA(const size_t haloId, const size_t bufferId, buf_t *buf_d) const {

        size_t haloSize = remoteHaloIndicesSizeCUDA[haloId];

        size_t mpiRecvBufferIndex = 0;
        for(size_t region = 0; region < remoteHaloIndicesCUDA_h[haloId].size(); ++region) {
            const std::array<int, 3> vals = remoteHaloIndicesCUDA_h[haloId][region];

            const int val_start = vals[0];
            const int val_howmany = vals[1];
            const int val_stride = vals[2];

            cudaMemcpy2D(&buf_d[val_start], val_stride*sizeof(buf_t),
                         &recvBufferCUDA_h[haloId][bufferId*haloSize + mpiRecvBufferIndex], sizeof(buf_t),
                         sizeof(buf_t), val_howmany, cudaMemcpyHostToDevice);
            mpiRecvBufferIndex += val_howmany;

        }

    }

#endif

private:

    inline std::vector<std::array<int, 3> > extractHaloIndicesWithStride(std::vector<int> indices) const {

        std::vector<std::array<int, 3> > ret;

        // special cases: 0, 1, 2 entries only

        if(indices.size() == 0)
            return ret;
        else if(indices.size() == 1) {
            std::array<int, 3> val = {static_cast<int>(indices[0]), 1, 1};
            ret.push_back(val);
            return ret;
        } else if(indices.size() == 2) {
            std::array<int, 3> val = {static_cast<int>(indices[0]), 2, static_cast<int>(indices[1])-static_cast<int>(indices[0])};
            ret.push_back(val);
            return ret;
        }

        // compute strides (first entry assumes to have same stride as second entry)
        std::vector<int> strides;
        strides.push_back(indices[1]-indices[0]);
        for(size_t i = 1; i < indices.size(); ++i)
            strides.push_back(indices[i]-indices[i-1]);

        // the current start/size/stride
        int curStart = static_cast<int>(indices[0]);
        int curStride = static_cast<int>(indices[1])-static_cast<int>(indices[0]);
        int curNum = 1;

        for(size_t ind = 1; ind < indices.size(); ++ind) {

            // the stride has changed
            if(strides[ind] != curStride) {

                // store everything up to now as region with same stride
                std::array<int, 3> vals = {curStart, curNum, curStride};
                ret.push_back(vals);

                // one stray element at the end
                if(ind == indices.size()-1) {
                    std::array<int, 3> val = {static_cast<int>(indices[ind]), 1, 1};
                    ret.push_back(val);
                } else {
                    // update/reset start/stride/size
                    curStart = static_cast<int>(indices[ind]);
                    curStride = strides[ind+1];
                    curNum = 1;
                }

            // same stride again
            } else {
                // one more item
                ++curNum;
                // if we reached the end, save region before finishing
                if(ind == indices.size()-1) {
                    std::array<int, 3> vals = {curStart, curNum, curStride};
                    ret.push_back(vals);
                }
            }

        }

        return ret;

    }

    MPI_Comm TAUSCH_COMM;
    MPI_Datatype mpiDataType;

    std::vector<std::vector<std::array<int, 3> > > localHaloIndicesCPU;
    std::vector<std::vector<std::array<int, 3> > > remoteHaloIndicesCPU;

    std::vector<size_t> localHaloIndicesSizeCPU;
    std::vector<size_t> remoteHaloIndicesSizeCPU;

    std::vector<int> localHaloRemoteMpiRankCPU;
    std::vector<int> remoteHaloRemoteMpiRankCPU;

    std::vector<size_t> localHaloNumBuffersCPU;
    std::vector<size_t> remoteHaloNumBuffersCPU;

    std::vector<buf_t*> sendBufferCPU;
    std::vector<buf_t*> recvBufferCPU;

    std::vector<MPI_Request*> mpiSendRequestsCPU;
    std::vector<MPI_Request*> mpiRecvRequestsCPU;

    std::vector<bool> setupMpiSendCPU;
    std::vector<bool> setupMpiRecvCPU;

#ifdef TAUSCH_OPENCL

    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program programs;
    int clKernelLocalSize;
    std::string cName4BufT;

    std::vector<std::vector<int> > localHaloIndicesOCL_h;
    std::vector<std::vector<int> > remoteHaloIndicesOCL_h;
    std::vector<cl::Buffer> localHaloIndicesOCL_d;
    std::vector<cl::Buffer> remoteHaloIndicesOCL_d;

    std::vector<int> remoteHaloIndicesSizeOCL;
    std::vector<int> localHaloIndicesSizeOCL;

    std::vector<int> localHaloRemoteMpiRankOCL;
    std::vector<int> remoteHaloRemoteMpiRankOCL;

    std::vector<int> localHaloNumBuffersOCL;
    std::vector<int> remoteHaloNumBuffersOCL;

    std::vector<cl::Buffer> sendBufferOCL_d;
    std::vector<buf_t*> sendBufferOCL_h;
    std::vector<cl::Buffer> recvBufferOCL_d;
    std::vector<buf_t*> recvBufferOCL_h;

    std::vector<MPI_Request*> mpiSendRequestsOCL;
    std::vector<MPI_Request*> mpiRecvRequestsOCL;

    std::vector<bool> setupMpiSendOCL;
    std::vector<bool> setupMpiRecvOCL;

#endif

    std::vector<std::vector<std::array<int, 3> > > localHaloIndicesCUDA_h;
    std::vector<std::vector<std::array<int, 3> > > remoteHaloIndicesCUDA_h;

    std::vector<int> remoteHaloIndicesSizeCUDA;
    std::vector<int> localHaloIndicesSizeCUDA;

    std::vector<int> localHaloRemoteMpiRankCUDA;
    std::vector<int> remoteHaloRemoteMpiRankCUDA;

    std::vector<int> localHaloNumBuffersCUDA;
    std::vector<int> remoteHaloNumBuffersCUDA;

    std::vector<buf_t*> sendBufferCUDA_d;
    std::vector<buf_t*> sendBufferCUDA_h;
    std::vector<buf_t*> recvBufferCUDA_d;
    std::vector<buf_t*> recvBufferCUDA_h;

    std::vector<MPI_Request*> mpiSendRequestsCUDA;
    std::vector<MPI_Request*> mpiRecvRequestsCUDA;

    std::vector<bool> setupMpiSendCUDA;
    std::vector<bool> setupMpiRecvCUDA;


};


#endif
