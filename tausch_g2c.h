#ifndef TAUSCH_G2C_H
#define TAUSCH_G2C_H

#include <mpi.h>
#include <vector>
#include "tauschdefs.h"
#include <thread>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

template <class buf_t>
class TauschG2C {

public:
    TauschG2C(cl::Device device, cl::Context context, cl::CommandQueue queue, std::string cName4BufT) {

        this->device = device;
        this->context = context;
        this->queue = queue;
        this->cName4BufT = cName4BufT;

        clKernelLocalSize = 512;

        std::string oclstr = "typedef "+cName4BufT+" buf_t;";

        oclstr += R"d(

kernel void pack(global const buf_t * restrict inBuf, global buf_t * restrict outBuf, global const int * restrict inIndices, const int numIndices) {

    int gid = get_global_id(0);

    if(gid < numIndices)
        outBuf[gid] = inBuf[inIndices[gid]];

}

kernel void packSubRegion(global const buf_t * restrict inBuf, global buf_t * restrict outBuf, global const int * restrict inIndices, global const int * restrict outIndices, const int numIndices) {

    int gid = get_global_id(0);

    if(gid < numIndices)
        outBuf[outIndices[gid]] = inBuf[inIndices[gid]];

}
                             )d";

        try {
            programs = cl::Program(context, oclstr, false);
            programs.build("");

//            std::string log = programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
//            std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl
//                      << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** "
//                      << std::endl << std::endl;
        } catch(cl::Error &e) {
            std::cout << "Tausch:G2C: TauschG2C(): OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
            if(e.err() == -11) {
                try {
                    std::string log = programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                    std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl
                              << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** "
                              << std::endl << std::endl;
                } catch(cl::Error &e) {
                    std::cout << "Tausch:G2C: TauschG2C(): getBuildInfo :: OpenCL exception caught: " << e.what() << " (" << e.err() << ")"
                              << std::endl;
                }
            }
        }
    }

    ~TauschG2C() {

        sendBuffer.clear();
        for(int i = 0; i < recvBuffer.size(); ++i)
            delete recvBuffer[i];

    }

    int addLocalHaloInfo(const TauschHaloRegion region, const size_t numBuffer) {

        std::vector<size_t> haloIndices;

        // 1D
        if(region.haloDepth == 0 && region.haloHeight == 0) {

            for(size_t x = 0; x < region.haloWidth; ++x)
                haloIndices.push_back(region.haloX+x);

        // 2D
        } else if(region.haloDepth == 0) {

            for(size_t y = 0; y < region.haloHeight; ++y)
                for(size_t x = 0; x < region.haloWidth; ++x)
                    haloIndices.push_back((region.haloY+y)*region.bufferWidth + region.haloX+x);

        // 3D
        } else {

            for(size_t z = 0; z < region.haloDepth; ++z)
                for(size_t y = 0; y < region.haloHeight; ++y)
                    for(size_t x = 0; x < region.haloWidth; ++x)
                        haloIndices.push_back((region.haloZ+z)*region.bufferWidth*region.bufferHeight + (region.haloY+y)*region.bufferWidth + region.haloX+x);

        }

        return addLocalHaloInfo(haloIndices, numBuffer);

    }

    int addLocalHaloInfo(const std::vector<size_t> haloIndices, const size_t numBuffers) {

        try {

            cl::Buffer clHaloIndices(context, haloIndices.begin(), haloIndices.end(), true);
            localHaloIndices.push_back(clHaloIndices);
            localHaloIndicesSize.push_back(haloIndices.size());

            cl::Buffer clNumBuffers(context, &numBuffers, (&numBuffers)+1, true);
            localHaloNumBuffers.push_back(clNumBuffers);

            cl::Buffer clSendBuffer(context, CL_MEM_READ_WRITE, haloIndices.size()*sizeof(buf_t));
            sendBuffer.push_back(clSendBuffer);

        } catch(cl::Error &e) {
            std::cout << "Tausch:G2C: addLocalHaloInfo(): OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

        return sendBuffer.size()-1;

    }

    int addRemoteHaloInfo(const TauschHaloRegion region, const size_t numBuffer) {

        std::vector<size_t> haloIndices;

        // 1D
        if(region.haloDepth == 0 && region.haloHeight == 0) {

            for(size_t x = 0; x < region.haloWidth; ++x)
                haloIndices.push_back(region.haloX+x);

        // 2D
        } else if(region.haloDepth == 0) {

            for(size_t y = 0; y < region.haloHeight; ++y)
                for(size_t x = 0; x < region.haloWidth; ++x)
                    haloIndices.push_back((region.haloY+y)*region.bufferWidth + region.haloX+x);

        // 3D
        } else {

            for(size_t z = 0; z < region.haloDepth; ++z)
                for(size_t y = 0; y < region.haloHeight; ++y)
                    for(size_t x = 0; x < region.haloWidth; ++x)
                        haloIndices.push_back((region.haloZ+z)*region.bufferWidth*region.bufferHeight + (region.haloY+y)*region.bufferWidth + region.haloX+x);

        }

        return addRemoteHaloInfo(haloIndices, numBuffer);

    }

    int addRemoteHaloInfo(const std::vector<size_t> haloIndices, const size_t numBuffers) {

        remoteHaloIndices.push_back(haloIndices);
        remoteHaloNumBuffers.push_back(numBuffers);

        dataSent.push_back(0);

        recvBuffer.push_back(new buf_t[haloIndices.size()]);

        return recvBuffer.size()-1;

    }

    void packSendBuffer(const size_t haloId, const size_t bufferId, cl::Buffer buf) {

        try {
            auto kernel_pack = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                    (programs, "pack");


            size_t globalsize = (localHaloIndicesSize[haloId]/clKernelLocalSize +1)*clKernelLocalSize;

            cl::Buffer clBufferId(context, &bufferId, (&bufferId)+1, true);

            kernel_pack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                          sendBuffer[haloId], buf, localHaloIndices[haloId], localHaloNumBuffers[haloId]);

        } catch(cl::Error &e) {
            std::cerr << "Tausch:G2C: packRecvBuffer() :: OpenCL exception caught: " << e.what()
                      << " (" << e.err() << ")" << std::endl;
        }

    }

    void packRecvBuffer(const size_t haloId, const size_t bufferId, cl::Buffer *buf, const std::vector<size_t> overwriteHaloSendIndices, const std::vector<size_t> overwriteHaloSourceIndices) {

        try {
            auto kernel_pack = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                    (programs, "pack");

            cl::Buffer clHaloIndicesIn(context, overwriteHaloSourceIndices.begin(), overwriteHaloSourceIndices.end(), true);
            cl::Buffer clHaloIndicesOut(context, overwriteHaloSendIndices.begin(), overwriteHaloSendIndices.end(), true);

            size_t globalsize = (localHaloIndicesSize[haloId]/clKernelLocalSize +1)*clKernelLocalSize;

            cl::Buffer clBufferId(context, &bufferId, (&bufferId)+1, true);

            kernel_pack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                          sendBuffer[haloId], buf, clHaloIndicesIn, clHaloIndicesOut, localHaloNumBuffers[haloId]);

        } catch(cl::Error &e) {
            std::cerr << "Tausch:G2C: packRecvBuffer() :: OpenCL exception caught: " << e.what()
                      << " (" << e.err() << ")" << std::endl;
        }

    }

    void send(const size_t haloId, int msgtag) {

        msgtags_keys.push_back(std::atomic<int>(msgtag));
        msgtags_vals.push_back(std::atomic<size_t>(haloId));

        dataSent[haloId].store(1);

    }

    void recv(const size_t haloId, int msgtag) {

        while(dataSent[haloId].load() == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        dataSent[haloId].store(0);

        int pos = -1;
        for(int i = 0; i < msgtags_keys.size(); ++i) {
            if(msgtags_keys[i].load() == msgtag) {
                pos = i;
                break;
            }
        }
        if(pos == -1) {
            std::cout << "Tausch:G2C: recv(): ERROR, unable to find msgtag " << msgtag << "..." << std::endl;
            return;
        }

        size_t id = msgtags_vals[pos];

        cl::copy(sendBuffer[id], &(recvBuffer[haloId][0]), &(recvBuffer[haloId][remoteHaloIndices[haloId].size()]));

    }

    void unpackRecvBuffer(const size_t haloId, const size_t bufferId, const buf_t *buf) {

        size_t haloSize = remoteHaloIndices[haloId].size();

        for(size_t index = 0; index < haloSize; ++index)
            buf[remoteHaloIndices[haloId][index]] = recvBuffer[haloId][bufferId*haloSize + index];

    }

    void unpackRecvBuffer(const size_t haloId, const size_t bufferId, const buf_t *buf, const std::vector<size_t> overwriteHaloRecvIndices, const std::vector<size_t> overwriteHaloTargetIndices) {

        size_t haloSize = remoteHaloIndices[haloId].size();

        for(size_t index = 0; index < overwriteHaloRecvIndices.size(); ++index)
            buf[overwriteHaloTargetIndices[index]] = recvBuffer[haloId][bufferId*haloSize + overwriteHaloRecvIndices[index]];

    }

    void packAndSend(const size_t haloId, cl::Buffer buf, const int msgtag) {
        packSendBuffer(haloId, 0, buf);
        send(haloId, msgtag);
    }

    void recvAndUnpack(const size_t haloId, buf_t *buf, const int msgtag) {
        recv(haloId, msgtag);
        unpackRecvBuffer(haloId, 0, buf);
    }

    size_t getNumLocalHalo() {
        return sendBuffer.size();
    }
    size_t getNumRemoteHalo() {
        return recvBuffer.size();
    }
    size_t getSizeLocalHalo(size_t haloId) {
        return localHaloIndicesSize[haloId];
    }
    size_t getSizeRemoteHalo(size_t haloId) {
        return remoteHaloIndices[haloId].size();
    }
    cl::Buffer getNumBuffersLocal(size_t haloId) {
        return localHaloNumBuffers[haloId];
    }
    size_t getNumBuffersRemote(size_t haloId) {
        return remoteHaloNumBuffers[haloId];
    }
    cl::Buffer getSendBuffer(size_t haloId) {
        return sendBuffer[haloId];
    }
    buf_t *getRecvBuffer(size_t haloId) {
        return recvBuffer[haloId];
    }

private:
    int alignedsize;

    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program programs;
    size_t clKernelLocalSize;
    std::string cName4BufT;

    std::vector<std::atomic<int>> msgtags_keys;
    std::vector<std::atomic<size_t>> msgtags_vals;

    std::vector<std::atomic<size_t> > dataSent;

    std::vector<cl::Buffer> localHaloIndices;
    std::vector<std::vector<size_t> > remoteHaloIndices;
    std::vector<size_t> remoteHaloNumBuffers;
    std::vector<size_t> localHaloIndicesSize;
    std::vector<cl::Buffer> localHaloNumBuffers;

    std::vector<cl::Buffer> sendBuffer;
    std::vector<buf_t*> recvBuffer;

};

#endif
