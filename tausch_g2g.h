#ifndef TAUSCH_G2G_H
#define TAUSCH_G2G_H

#include <vector>
#include <array>
#include "tauschdefs.h"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

template <class buf_t>
class TauschG2G {

public:
    TauschG2G(cl::Device device, cl::Context context, cl::CommandQueue queue, std::string cName4BufT) {

        this->device = device;
        this->context = context;
        this->queue = queue;
        this->cName4BufT = cName4BufT;

        clKernelLocalSize = 256;

        std::string oclstr = "typedef "+cName4BufT+" buf_t;";

        oclstr += R"d(

kernel void pack(global const buf_t * restrict inBuf,
                 global buf_t * restrict outBuf,
                 global const int * restrict inIndices,
                 global const int * restrict const numIndices,
                 global const int * restrict const bufferId) {

    int gid = get_global_id(0);

    if(gid < *numIndices)
        outBuf[(*bufferId)*(*numIndices) + gid] = inBuf[inIndices[gid]];

}

kernel void packSubRegion(global const buf_t * restrict inBuf,
                          global buf_t * restrict outBuf,
                          global const int * restrict inIndices,
                          global const int * restrict outIndices,
                          global const int * restrict const numIndices,
                          global const int * restrict const bufferId) {

    int gid = get_global_id(0);

    if(gid < *numIndices)
        outBuf[(*bufferId)*(*numIndices) + outIndices[gid]] = inBuf[inIndices[gid]];

}

kernel void unpack(global const buf_t * restrict const inBuf,
                   global buf_t * restrict const outBuf,
                   global const int * restrict const outIndices,
                   global const int * restrict const numIndices,
                   global const int * restrict const bufferId) {

    int gid = get_global_id(0);

    if(gid < *numIndices)
        outBuf[outIndices[gid]] = inBuf[(*bufferId)*(*numIndices) + gid];

}

kernel void unpackSubRegion(global const buf_t * restrict inBuf,
                            global buf_t * restrict outBuf,
                            global const int * restrict inIndices,
                            global const int * restrict outIndices,
                            global const int * restrict const numIndices,
                            global const int * restrict const bufferId) {

    int gid = get_global_id(0);

    if(gid < numIndices)
        outBuf[outIndices[gid]] = inBuf[(*bufferId)*(*numIndices) + inIndices[gid]];

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

    ~TauschG2G() {

        sendBuffer.clear();
        recvBuffer.clear();

    }

    int addLocalHaloInfo(const TauschHaloRegion region, int numBuffer) {

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

    int addLocalHaloInfo(std::vector<int> haloIndices, int numBuffers) {

        try {

            if(haloIndices.size() == 0) {

                cl::Buffer clHaloIndices(context, CL_MEM_READ_ONLY, sizeof(buf_t));
                localHaloIndices.push_back(clHaloIndices);
                localHaloIndicesSize.push_back(haloIndices.size());

                cl::Buffer clNumBuffers(context, &numBuffers, (&numBuffers)+1, true);
                localHaloNumBuffers.push_back(clNumBuffers);
                localHaloNumBuffersInt.push_back(numBuffers);

                cl::Buffer clSendBuffer(context, CL_MEM_READ_WRITE, sizeof(buf_t));
                sendBuffer.push_back(clSendBuffer);

                dataSent.push_back(0);

            } else {

                cl::Buffer clHaloIndices(context, haloIndices.begin(), haloIndices.end(), true);
                localHaloIndices.push_back(clHaloIndices);
                localHaloIndicesSize.push_back(haloIndices.size());

                cl::Buffer clNumBuffers(context, &numBuffers, (&numBuffers)+1, true);
                localHaloNumBuffers.push_back(clNumBuffers);
                localHaloNumBuffersInt.push_back(numBuffers);

                cl::Buffer clSendBuffer(context, CL_MEM_READ_WRITE, numBuffers*haloIndices.size()*sizeof(buf_t));
                sendBuffer.push_back(clSendBuffer);

                dataSent.push_back(0);

            }

        } catch(cl::Error &e) {
            std::cout << "Tausch:G2C: addLocalHaloInfo(): OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

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

            if(haloIndices.size() == 0) {

                cl::Buffer clHaloIndices(context, CL_MEM_READ_ONLY, sizeof(buf_t));
                remoteHaloIndices.push_back(clHaloIndices);
                remoteHaloIndicesSize.push_back(haloIndices.size());

                cl::Buffer clNumBuffers(context, &numBuffers, (&numBuffers)+1, true);
                remoteHaloNumBuffers.push_back(clNumBuffers);
                remoteHaloNumBuffersInt.push_back(numBuffers);

                cl::Buffer clRecvBuffer(context, CL_MEM_READ_WRITE, sizeof(buf_t));
                recvBuffer.push_back(clRecvBuffer);

            } else {

                cl::Buffer clHaloIndices(context, haloIndices.begin(), haloIndices.end(), true);
                remoteHaloIndices.push_back(clHaloIndices);
                remoteHaloIndicesSize.push_back(haloIndices.size());

                cl::Buffer clNumBuffers(context, &numBuffers, (&numBuffers)+1, true);
                remoteHaloNumBuffers.push_back(clNumBuffers);
                remoteHaloNumBuffersInt.push_back(numBuffers);

                cl::Buffer clRecvBuffer(context, CL_MEM_READ_WRITE, numBuffers*haloIndices.size()*sizeof(buf_t));
                recvBuffer.push_back(clRecvBuffer);

            }

        } catch(cl::Error &e) {
            std::cout << "Tausch:C2G: addRemoteHaloInfo(): OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

        return recvBuffer.size()-1;

    }

    void packSendBuffer(const int haloId, int bufferId, cl::Buffer buf) {

        try {
            auto kernel_pack = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>
                    (programs, "pack");


            int globalsize = (localHaloIndicesSize[haloId]/clKernelLocalSize +1)*clKernelLocalSize;

            cl::Buffer clBufferId(context, &bufferId, (&bufferId)+1, true);

            int haloSize = localHaloIndicesSize[haloId];
            cl::Buffer clHaloSize(context, &haloSize, (&haloSize)+1, true);

            kernel_pack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                        buf, sendBuffer[haloId], localHaloIndices[haloId], clHaloSize, clBufferId);

        } catch(cl::Error &e) {
            std::cerr << "Tausch:G2C: packSendBuffer() [1] :: OpenCL exception caught: " << e.what()
                      << " (" << e.err() << ")" << std::endl;
        }

    }

    void packSendBuffer(const int haloId, int bufferId, cl::Buffer buf, const std::vector<int> overwriteHaloSendIndices, const std::vector<int> overwriteHaloSourceIndices) {

        try {
            auto kernel_pack = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>
                    (programs, "pack");

            cl::Buffer clHaloIndicesIn(context, overwriteHaloSourceIndices.begin(), overwriteHaloSourceIndices.end(), true);
            cl::Buffer clHaloIndicesOut(context, overwriteHaloSendIndices.begin(), overwriteHaloSendIndices.end(), true);

            int globalsize = (localHaloIndicesSize[haloId]/clKernelLocalSize +1)*clKernelLocalSize;

            cl::Buffer clBufferId(context, &bufferId, (&bufferId)+1, true);

            int haloSize = localHaloIndicesSize[haloId];
            cl::Buffer clHaloSize(context, &haloSize, (&haloSize)+1, true);

            kernel_pack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                        buf, sendBuffer[haloId], clHaloIndicesIn, clHaloIndicesOut, clHaloSize, clBufferId);

        } catch(cl::Error &e) {
            std::cerr << "Tausch:G2C: packSendBuffer() [2] :: OpenCL exception caught: " << e.what()
                      << " (" << e.err() << ")" << std::endl;
        }

    }

    void send(const int haloId, int msgtag) {

        msgtags_keys.push_back(msgtag);
        msgtags_vals.push_back(haloId);

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
            std::cout << "Tausch:G2C: recv(): ERROR, unable to find msgtag " << msgtag << "..." << std::endl;
            return;
        }

        int id = msgtags_vals[pos];

        queue.enqueueCopyBuffer(sendBuffer[id], recvBuffer[haloId], 0, 0, remoteHaloNumBuffersInt[haloId]*remoteHaloIndicesSize[haloId]*sizeof(buf_t));

    }

    void unpackRecvBuffer(const int haloId, int bufferId, cl::Buffer buf) {

        try {
            auto kernel_unpack = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>
                    (programs, "unpack");

            int globalsize = (remoteHaloIndicesSize[haloId]/clKernelLocalSize +1)*clKernelLocalSize;
            int haloSize = remoteHaloIndicesSize[haloId];
            cl::Buffer clHaloSize(context, &haloSize, (&haloSize)+1, true);
            cl::Buffer clBufferId(context, &bufferId, (&bufferId)+1, true);
            kernel_unpack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                          recvBuffer[haloId], buf, remoteHaloIndices[haloId], clHaloSize, clBufferId);

        } catch(cl::Error &e) {
            std::cerr << "Tausch:C2G: unpackRecvBuffer() :: OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

    }

    void unpackRecvBuffer(const int haloId, int bufferId, cl::Buffer buf, const std::vector<int> overwriteHaloRecvIndices, const std::vector<int> overwriteHaloTargetIndices) {

        try {
            auto kernel_unpack = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>
                    (programs, "unpackSubRegion");

            cl::Buffer clHaloIndicesIn(context, overwriteHaloRecvIndices.begin(), overwriteHaloRecvIndices.end(), true);
            cl::Buffer clHaloIndicesOut(context, overwriteHaloTargetIndices.begin(), overwriteHaloTargetIndices.end(), true);

            int globalsize = (remoteHaloIndicesSize[haloId]/clKernelLocalSize +1)*clKernelLocalSize;

            cl::Buffer clBufferId(context, &bufferId, (&bufferId)+1, true);
            cl::Buffer haloSize(context, &remoteHaloIndicesSize[haloId], (&remoteHaloIndicesSize[haloId])+1, true);

            kernel_unpack(cl::EnqueueArgs(queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                          recvBuffer[haloId], buf, clHaloIndicesIn, clHaloIndicesOut, haloSize, clBufferId);

        } catch(cl::Error e) {
            std::cerr << "Tausch:C2G: unpackRecvBuffer() :: OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

    }

    void packAndSend(const int haloId, cl::Buffer buf, const int msgtag) {
        packSendBuffer(haloId, 0, buf);
        send(haloId, msgtag);
    }

    void recvAndUnpack(const int haloId, cl::Buffer buf, const int msgtag) {
        recv(haloId, msgtag);
        unpackRecvBuffer(haloId, 0, buf);
    }

private:
    int alignedsize;

    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program programs;
    int clKernelLocalSize;
    std::string cName4BufT;

    std::vector<int> msgtags_keys;
    std::vector<int> msgtags_vals;

    std::vector<int> dataSent;

    std::vector<cl::Buffer> localHaloIndices;
    std::vector<cl::Buffer> remoteHaloIndices;
    std::vector<cl::Buffer> localHaloNumBuffers;
    std::vector<int> localHaloNumBuffersInt;
    std::vector<cl::Buffer> remoteHaloNumBuffers;
    std::vector<int> remoteHaloNumBuffersInt;
    std::vector<int> localHaloIndicesSize;
    std::vector<int> remoteHaloIndicesSize;

    std::vector<cl::Buffer> sendBuffer;
    std::vector<cl::Buffer> recvBuffer;

};

#endif
