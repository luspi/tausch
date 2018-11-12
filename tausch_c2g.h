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

kernel void unpack(global const double * restrict inBuf, global double * restrict outBuf, global const int * restrict outIndices, global const int *numIndices) {

    int gid = get_global_id(0);

    if(gid < *numIndices)
        outBuf[outIndices[gid]] = inBuf[gid];

}

kernel void unpackSubRegion(global const double * restrict inBuf, global double * restrict outBuf, global const int * restrict inIndices, global const int * restrict outIndices, const int numIndices) {

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
        recvBuffer.clear();

    }

    int addLocalHaloInfo(const TauschHaloRegion region, const int numBuffer) {

        std::vector<int> haloIndices;

        // 1D
        if(region.haloDepth == 0 && region.haloHeight == 0) {

            for(int x = 0; x < region.haloWidth; ++x)
                haloIndices.push_back(region.haloX+x);

        // 2D
        } else if(region.haloDepth == 0) {

            for(int y = 0; y < region.haloHeight; ++y)
                for(int x = 0; x < region.haloWidth; ++x)
                    haloIndices.push_back((region.haloY+y)*region.bufferWidth + region.haloX+x);

        // 3D
        } else {

            for(int z = 0; z < region.haloDepth; ++z)
                for(int y = 0; y < region.haloHeight; ++y)
                    for(int x = 0; x < region.haloWidth; ++x)
                        haloIndices.push_back((region.haloZ+z)*region.bufferWidth*region.bufferHeight + (region.haloY+y)*region.bufferWidth + region.haloX+x);

        }

        return addLocalHaloInfo(haloIndices, numBuffer);

    }

    int addLocalHaloInfo(const std::vector<int> haloIndices, const int numBuffers) {

        localHaloIndices.push_back(haloIndices);
        localHaloNumBuffers.push_back(numBuffers);

        dataSent.push_back(0);

        sendBuffer.push_back(new buf_t[haloIndices.size()]);

        return sendBuffer.size()-1;

    }

    int addRemoteHaloInfo(const TauschHaloRegion region, const int numBuffer) {

        std::vector<int> haloIndices;

        // 1D
        if(region.haloDepth == 0 && region.haloHeight == 0) {

            for(int x = 0; x < region.haloWidth; ++x)
                haloIndices.push_back(region.haloX+x);

        // 2D
        } else if(region.haloDepth == 0) {

            for(int y = 0; y < region.haloHeight; ++y)
                for(int x = 0; x < region.haloWidth; ++x)
                    haloIndices.push_back((region.haloY+y)*region.bufferWidth + region.haloX+x);

        // 3D
        } else {

            for(int z = 0; z < region.haloDepth; ++z)
                for(int y = 0; y < region.haloHeight; ++y)
                    for(int x = 0; x < region.haloWidth; ++x)
                        haloIndices.push_back((region.haloZ+z)*region.bufferWidth*region.bufferHeight + (region.haloY+y)*region.bufferWidth + region.haloX+x);

        }

        return addRemoteHaloInfo(haloIndices, numBuffer);

    }

    int addRemoteHaloInfo(std::vector<int> haloIndices, int numBuffers) {

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

    void packSendBuffer(const int haloId, const int bufferId, const buf_t *buf) {

        int haloSize = localHaloIndices[haloId].size();

        for(int index = 0; index < haloSize; ++index)
            sendBuffer[haloId][bufferId*haloSize + index] = buf[localHaloIndices[haloId][index]];

    }

    void packSendBuffer(const int haloId, const int bufferId, const buf_t *buf, const std::vector<int> overwriteHaloSendIndices, const std::vector<int> overwriteHaloSourceIndices) {

        int haloSize = localHaloIndices[haloId].size();

        for(int index = 0; index < overwriteHaloSendIndices.size(); ++index)
            sendBuffer[haloId][bufferId*haloSize + overwriteHaloSendIndices[index]] = buf[overwriteHaloSourceIndices[index]];

    }

    void send(const int haloId, int msgtag) {

        msgtags_keys.push_back(std::atomic<int>(msgtag));
        msgtags_vals.push_back(std::atomic<int>(haloId));

        dataSent[haloId] = 1;

    }

    void recv(const int haloId, int msgtag) {

        while(dataSent[haloId] != 1)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        dataSent[haloId] = 0;

        int pos = -1;
        for(int i = 0; i < msgtags_keys.size(); ++i) {
            if(msgtags_keys[i] == msgtag) {
                pos = i;
                break;
            }
        }
        if(pos == -1) {
            std::cout << "Tausch:C2G: recv(): ERROR, unable to find msgtag " << msgtag << "..." << std::endl;
            return;
        }

        int id = msgtags_vals[pos];
        cl::copy(queue, sendBuffer[id].begin(), sendBuffer[id].end(), recvBuffer[haloId]);

    }

    void unpackRecvBuffer(const int haloId, int bufferId, cl::Buffer buf) {

        try {

            int haloSize = remoteHaloIndicesSize[haloId];
            cl::Buffer clHaloSize(context, &haloSize, (&haloSize)+1, true);

            cl::Kernel kernel_unpack(programs, "unpack");

            try {
                kernel_unpack.setArg(0, recvBuffer[haloId]);
            } catch(cl::Error &e) {
                std::cout << "Error at 0" << std::endl;
            }
            try {
                kernel_unpack.setArg(1, buf);
            } catch(cl::Error &e) {
                std::cout << "Error at 1" << std::endl;
            }
            try {
                kernel_unpack.setArg(2, remoteHaloIndices[haloId]);
            } catch(cl::Error &e) {
                std::cout << "Error at 2" << std::endl;
            }
            try {
                kernel_unpack.setArg(3, clHaloSize);
            } catch(cl::Error &e) {
                std::cout << "Error at 3" << std::endl;
            }

            int globalsize = (remoteHaloIndicesSize[haloId]/clKernelLocalSize +1)*clKernelLocalSize;

            queue.enqueueNDRangeKernel(kernel_unpack,cl::NDRange(globalsize),cl::NDRange(clKernelLocalSize));
            queue.finish();
/*
            auto kernel_unpack = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>
                                                    (programs, "unpack");


            int globalsize = (remoteHaloIndicesSize[haloId]/clKernelLocalSize +1)*clKernelLocalSize;

//            cl::Buffer clBufferId(context, CL_MEM_READ_ONLY, 1*sizeof(int));
//            cl::copy(queue, &bufferId, (&bufferId)+1, clBufferId);
//            cl::Buffer haloSize(context, &remoteHaloIndicesSize[haloId], (&remoteHaloIndicesSize[haloId])+1, true);
            int haloSize = remoteHaloIndicesSize[haloId];
            cl::Buffer clHaloSize(context, &haloSize, (&haloSize)+1, true);

            kernel_unpack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                        recvBuffer[haloId], buf, remoteHaloIndices[haloId], clHaloSize);
*/
        } catch(cl::Error &e) {
            std::cerr << "Tausch:C2G: unpackRecvBuffer() :: OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

    }

    void unpackRecvBuffer(const int haloId, int bufferId, cl::Buffer buf, const std::vector<int> overwriteHaloRecvIndices, const std::vector<int> overwriteHaloTargetIndices) {

        try {
            auto kernel_unpack = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                    (programs, "unpackSubRegion");

            cl::Buffer clHaloIndicesIn(context, overwriteHaloRecvIndices.begin(), overwriteHaloRecvIndices.end(), true);
            cl::Buffer clHaloIndicesOut(context, overwriteHaloTargetIndices.begin(), overwriteHaloTargetIndices.end(), true);

            int globalsize = (remoteHaloIndicesSize[haloId]/clKernelLocalSize +1)*clKernelLocalSize;

            cl::Buffer clBufferId(context, &bufferId, (&bufferId)+1, true);
            cl::Buffer haloSize(context, &remoteHaloIndicesSize[haloId], (&remoteHaloIndicesSize[haloId])+1, true);

            kernel_unpack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                          recvBuffer[haloId], buf, clHaloIndicesIn, clHaloIndicesOut, haloSize);

        } catch(cl::Error &e) {
            std::cerr << "Tausch:C2G: unpackRecvBuffer() :: OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

    }

    void packAndSend(const int haloId, buf_t *buf, const int msgtag) {
        packSendBuffer(haloId, 0, buf);
        send(haloId, msgtag);
    }

    void recvAndUnpack(const int haloId, cl::Buffer buf, const int msgtag) {
        recv(haloId, msgtag);
        unpackRecvBuffer(haloId, 0, buf);
    }

    int getNumLocalHalo() {
        return sendBuffer.size();
    }
    int getNumRemoteHalo() {
        return recvBuffer.size();
    }
    int getSizeLocalHalo(int haloId) {
        return localHaloIndices[haloId].size();
    }
    int getSizeRemoteHalo(int haloId) {
        return remoteHaloIndicesSize[haloId];
    }
    int getNumBuffersLocal(int haloId) {
        return localHaloNumBuffers[haloId];
    }
    cl::Buffer getNumBuffersRemote(int haloId) {
        return remoteHaloNumBuffers[haloId];
    }
    buf_t *getSendBuffer(int haloId) {
        return sendBuffer[haloId];
    }
    cl::Buffer getRecvBuffer(int haloId) {
        return recvBuffer[haloId];
    }

private:
    int alignedsize;

    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program programs;
    int clKernelLocalSize;

    std::vector<int> msgtags_keys;
    std::vector<int> msgtags_vals;

    std::vector<int > dataSent;

    std::vector<std::vector<int> > localHaloIndices;
    std::vector<cl::Buffer> remoteHaloIndices;
    std::vector<int> remoteHaloIndicesSize;
    std::vector<int> localHaloNumBuffers;
    std::vector<cl::Buffer> remoteHaloNumBuffers;

    std::vector<cl::Buffer> recvBuffer;
    std::vector<buf_t*> sendBuffer;

};

#endif
