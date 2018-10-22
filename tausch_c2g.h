#ifndef TAUSCH_C2G_H
#define TAUSCH_C2G_H

#include <mpi.h>
#include <vector>
#include "tauschdefs.h"
#include <thread>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

template <class buf_t>
class TauschC2G {

public:
    TauschC2G(cl::Device device, cl::Context context, cl::CommandQueue queue) {

        this->device = device;
        this->context = context;
        this->queue = queue;

        clKernelLocalSize = 512;

        std::string oclstr = R"d(

global void unpack(global const double * restrict inBuf, global double * restrict outBuf, global const double * restrict outIndices, const int numIndices) {

    int gid = get_global_id(0);

    if(gid >= numIndices)
        return;

    outBuf[outIndices[gid]] = inBuf[gid];

}

global void unpackSubRegion(global const double * restrict inBuf, global double * restrict outBuf, global const double * restrict inIndices, global const double * restrict outIndices, const int numIndices) {

    int gid = get_global_id(0);

    if(gid >= numIndices)
        return;

    outBuf[outIndices[gid]] = inBuf[inIndices[gid]];

}
                             )d";

        try {
            programs = cl::Program(context, oclstr, false);
            programs.build("");

            std::string log = programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl
                      << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** "
                      << std::endl << std::endl;
        } catch(cl::Error &e) {
            std::cout << "Tausch:CWG: TauschCWG(): OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
            if(e.err() == -11) {
                try {
                    std::string log = programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                    std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl
                              << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** "
                              << std::endl << std::endl;
                } catch(cl::Error &e) {
                    std::cout << "Tausch:CWG: TauschCWG(): getBuildInfo :: OpenCL exception caught: " << e.what() << " (" << e.err() << ")"
                              << std::endl;
                }
            }
        }
    }

    ~TauschC2G() {

        for(int i = 0; i < sendBuffer.size(); ++i)
            delete sendBuffer[i];
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

        localHaloIndices.push_back(haloIndices);
        localHaloNumBuffers.push_back(numBuffers);

        dataSent.push_back(0);

        sendBuffer.push_back(new buf_t[haloIndices.size()]);

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

        try {

            cl::Buffer clHaloIndices(context, haloIndices.begin(), haloIndices.end(), true);
            remoteHaloIndices.push_back(clHaloIndices);
            remoteHaloIndicesSize.push_back(haloIndices.size());

            cl::Buffer clNumBuffers(context, &numBuffers, (&numBuffers)+1, true);
            remoteHaloNumBuffers.push_back(clNumBuffers);

            cl::Buffer clRecvBuffer(context, CL_MEM_READ_WRITE, haloIndices.size()*sizeof(buf_t));
            recvBuffer.push_back(clRecvBuffer);

        } catch(cl::Error &e) {
            std::cout << "Tausch:C2G: addRemoteHaloInfo(): OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

        return recvBuffer.size()-1;

    }

    void packSendBuffer(const size_t haloId, const size_t bufferId, const buf_t *buf) {

        size_t haloSize = localHaloIndices[haloId].size();

        for(size_t index = 0; index < haloSize; ++index)
            sendBuffer[haloId][bufferId*haloSize + index] = buf[localHaloIndices[haloId][index]];

    }

    void packSendBuffer(const size_t haloId, const size_t bufferId, const buf_t *buf, const std::vector<size_t> overwriteHaloSendIndices, const std::vector<size_t> overwriteHaloSourceIndices) {

        size_t haloSize = localHaloIndices[haloId].size();

        for(size_t index = 0; index < overwriteHaloSendIndices.size(); ++index)
            sendBuffer[haloId][bufferId*haloSize + overwriteHaloSendIndices[index]] = buf[overwriteHaloSourceIndices[index]];

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
            std::cout << "Tausch:C2G: recv(): ERROR, unable to find msgtag " << msgtag << "..." << std::endl;
            return;
        }

        size_t id = msgtags_vals[pos];
        cl::copy(&(sendBuffer[id][0]), &(sendBuffer[id][localHaloIndices[id].size()]), recvBuffer[haloId]);

    }

    void unpackRecvBuffer(const size_t haloId, const size_t bufferId, cl::Buffer buf) {

        try {
            auto kernel_unpack = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                    (programs, "unpack");


            size_t globalsize = (remoteHaloIndicesSize[haloId]/clKernelLocalSize +1)*clKernelLocalSize;

            cl::Buffer clBufferId(context, &bufferId, (&bufferId)+1, true);

            kernel_unpack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                          recvBuffer[haloId], buf, remoteHaloIndices[haloId], remoteHaloNumBuffers[haloId]);

        } catch(cl::Error &e) {
            std::cerr << "Tausch:C2G: unpackRecvBuffer() :: OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

    }

    void unpackRecvBuffer(const size_t haloId, const size_t bufferId, cl::Buffer buf, const std::vector<size_t> overwriteHaloRecvIndices, const std::vector<size_t> overwriteHaloTargetIndices) {

        try {
            auto kernel_unpack = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                    (programs, "unpackSubRegion");

            cl::Buffer clHaloIndicesIn(context, overwriteHaloRecvIndices.begin(), overwriteHaloRecvIndices.end(), true);
            cl::Buffer clHaloIndicesOut(context, overwriteHaloTargetIndices.begin(), overwriteHaloTargetIndices.end(), true);

            size_t globalsize = (remoteHaloIndicesSize[haloId]/clKernelLocalSize +1)*clKernelLocalSize;

            cl::Buffer clBufferId(context, &bufferId, (&bufferId)+1, true);

            kernel_unpack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                          recvBuffer[haloId], buf, clHaloIndicesIn, clHaloIndicesOut, remoteHaloNumBuffers[haloId]);

        } catch(cl::Error &e) {
            std::cerr << "Tausch:C2G: unpackRecvBuffer() :: OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

    }

    void packAndSend(const size_t haloId, buf_t *buf, const int msgtag) {
        packSendBuffer(haloId, 0, buf);
        send(haloId, msgtag);
    }

    void recvAndUnpack(const size_t haloId, cl::Buffer buf, const int msgtag) {
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
        return localHaloIndices[haloId].size();
    }
    size_t getSizeRemoteHalo(size_t haloId) {
        return remoteHaloIndicesSize[haloId];
    }
    size_t getNumBuffersLocal(size_t haloId) {
        return localHaloNumBuffers[haloId];
    }
    cl::Buffer getNumBuffersRemote(size_t haloId) {
        return remoteHaloNumBuffers[haloId];
    }
    buf_t *getSendBuffer(size_t haloId) {
        return sendBuffer[haloId];
    }
    cl::Buffer getRecvBuffer(size_t haloId) {
        return recvBuffer[haloId];
    }

private:
    int alignedsize;

    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program programs;
    size_t clKernelLocalSize;

    std::vector<std::atomic<int>> msgtags_keys;
    std::vector<std::atomic<size_t>> msgtags_vals;

    std::vector<std::atomic<size_t> > dataSent;

    std::vector<std::vector<size_t> > localHaloIndices;
    std::vector<cl::Buffer> remoteHaloIndices;
    std::vector<size_t> remoteHaloIndicesSize;
    std::vector<size_t> localHaloNumBuffers;
    std::vector<cl::Buffer> remoteHaloNumBuffers;

    std::vector<cl::Buffer> recvBuffer;
    std::vector<buf_t*> sendBuffer;

};

#endif
