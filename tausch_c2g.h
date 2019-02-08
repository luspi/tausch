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
    TauschC2G(cl::Device device, cl::Context context, cl::CommandQueue queue, std::string cName4BufT) {

        this->tauschcl_device = device;
        this->tauschcl_context = context;
        this->tauschcl_queue = queue;
        this->cName4BufT = cName4BufT;

        clKernelLocalSize = 256;

        std::string oclstr = "typedef "+cName4BufT+" buf_t;";

        oclstr += R"d(

kernel void unpack(global const buf_t * restrict const inBuf,
                   global buf_t * restrict const outBuf,
                   global const int * restrict const outIndices,
                   global const int * restrict const numIndices,
                   global const int * restrict const bufferId) {

    int gid = get_global_id(0);

    if(gid < *numIndices)
        outBuf[outIndices[gid]] = inBuf[(*bufferId) * (*numIndices) + gid];

}

kernel void unpackSubRegion(global const buf_t * restrict inBuf, global buf_t * restrict outBuf, global const int * restrict inIndices, global const int * restrict outIndices, const int numIndices, const int bufferId) {

    int gid = get_global_id(0);

    if(gid < numIndices)
        outBuf[outIndices[gid]] = inBuf[bufferId*numIndices + inIndices[gid]];

}
                             )d";

        try {
            programs = cl::Program(tauschcl_context, oclstr, false);
            programs.build("");

//            std::string log = programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
//            std::cout << std::endl << " ******************** " << std::endl << " ** BUILD LOG" << std::endl
//                      << " ******************** " << std::endl << log << std::endl << std::endl << " ******************** "
//                      << std::endl << std::endl;
        } catch(cl::Error &e) {
            std::cout << "Tausch:CWG: TauschCWG(): OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
            if(e.err() == -11) {
                try {
                    std::string log = programs.getBuildInfo<CL_PROGRAM_BUILD_LOG>(tauschcl_device);
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
            delete[] sendBuffer[i];
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

        return addLocalHaloInfo(extractHaloIndicesWithStride(haloIndices), numBuffer);

    }

    int addLocalHaloInfo(const std::vector<int> haloIndices, const int numBuffers) {

        return addLocalHaloInfo(extractHaloIndicesWithStride(haloIndices), numBuffers);

    }

    int addLocalHaloInfo(const std::vector<std::array<int, 3> > haloIndices, const int numBuffers) {

        localHaloIndices.push_back(haloIndices);
        localHaloNumBuffers.push_back(numBuffers);

        dataSent.push_back(0);

        size_t bufsize = 0;
        for(size_t i = 0; i < haloIndices.size(); ++i)
            bufsize += static_cast<size_t>(haloIndices.at(i)[1]);

        localHaloIndicesSize.push_back(bufsize);

        void *newbuf = NULL;
        posix_memalign(&newbuf, 64, numBuffers*bufsize*sizeof(buf_t));
        buf_t *newbuf_buft = reinterpret_cast<buf_t*>(newbuf);
        double zero = 0;
        std::fill_n(newbuf_buft, numBuffers*bufsize, zero);
        sendBuffer.push_back(newbuf_buft);

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

                cl::Buffer clHaloIndices(tauschcl_context, CL_MEM_READ_WRITE, sizeof(buf_t));
                remoteHaloIndices.push_back(clHaloIndices);
                remoteHaloIndicesSize.push_back(haloIndices.size());

                cl::Buffer clNumBuffers(tauschcl_context, &numBuffers, (&numBuffers)+1, true);
                remoteHaloNumBuffers.push_back(clNumBuffers);

                cl::Buffer clRecvBuffer(tauschcl_context, CL_MEM_READ_WRITE, sizeof(buf_t));
                recvBuffer.push_back(clRecvBuffer);

            } else {

                cl::Buffer clHaloIndices(tauschcl_context, haloIndices.begin(), haloIndices.end(), true);
                remoteHaloIndices.push_back(clHaloIndices);
                remoteHaloIndicesSize.push_back(haloIndices.size());

                cl::Buffer clNumBuffers(tauschcl_context, &numBuffers, (&numBuffers)+1, true);
                remoteHaloNumBuffers.push_back(clNumBuffers);

                cl::Buffer clRecvBuffer(tauschcl_context, CL_MEM_READ_WRITE, numBuffers*haloIndices.size()*sizeof(buf_t));
                recvBuffer.push_back(clRecvBuffer);

            }

        } catch(cl::Error &e) {
            std::cout << "Tausch:C2G: addRemoteHaloInfo(): OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

        return recvBuffer.size()-1;

    }

    void packSendBuffer(const int haloId, const int bufferId, const buf_t *buf) {

        const int haloSize = localHaloIndicesSize[haloId];

        int sendBufferIndex = 0;
        for(int region = 0; region < localHaloIndices[haloId].size(); ++region) {
            const std::array<int, 3> vals = localHaloIndices[haloId][region];

            const int val_start = vals[0];
            const int val_howmany = vals[1];
            const int val_stride = vals[2];

            if(val_stride == 1) {
                memcpy(&sendBuffer[haloId][bufferId*haloSize + sendBufferIndex], &buf[val_start], val_howmany*sizeof(buf_t));
                sendBufferIndex += val_howmany;
            } else {
                const int sendBufferIndexBASE = bufferId*haloSize + sendBufferIndex;
                for(int i = 0; i < val_howmany; ++i)
                    sendBuffer[haloId][sendBufferIndexBASE + i] = buf[val_start+i*val_stride];
                sendBufferIndex += val_howmany;
            }

        }

    }

    void packSendBuffer(const int haloId, const int bufferId, const buf_t *buf, const std::vector<int> overwriteHaloSendIndices, const std::vector<int> overwriteHaloSourceIndices) {

        int haloSize = localHaloIndices[haloId].size();

        for(int index = 0; index < overwriteHaloSendIndices.size(); ++index)
            sendBuffer[haloId][bufferId*haloSize + overwriteHaloSendIndices[index]] = buf[overwriteHaloSourceIndices[index]];

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
            std::cout << "Tausch:C2G: recv(): ERROR, unable to find msgtag " << msgtag << "..." << std::endl;
            return;
        }

        int id = msgtags_vals[pos];
        cl::copy(tauschcl_queue, &sendBuffer[id][0], &sendBuffer[id][localHaloNumBuffers[id]*remoteHaloIndicesSize[id]], recvBuffer[haloId]);

    }

    void unpackRecvBuffer(const int haloId, int bufferId, cl::Buffer buf) {

        try {
            auto kernel_unpack = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>
                                                    (programs, "unpack");

            int globalsize = (remoteHaloIndicesSize[haloId]/clKernelLocalSize +1)*clKernelLocalSize;
            int haloSize = remoteHaloIndicesSize[haloId];
            cl::Buffer clHaloSize(tauschcl_context, &haloSize, (&haloSize)+1, true);
            cl::Buffer clBufferId(tauschcl_context, &bufferId, (&bufferId)+1, true);
            kernel_unpack(cl::EnqueueArgs(tauschcl_queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                        recvBuffer[haloId], buf, remoteHaloIndices[haloId], clHaloSize, clBufferId);

        } catch(cl::Error &e) {
            std::cerr << "Tausch:C2G: unpackRecvBuffer() :: OpenCL exception caught: " << e.what() << " (" << e.err() << ")" << std::endl;
        }

    }

    void unpackRecvBuffer(const int haloId, int bufferId, cl::Buffer buf, const std::vector<int> overwriteHaloRecvIndices, const std::vector<int> overwriteHaloTargetIndices) {

        try {
            auto kernel_unpack = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                                                    (programs, "unpackSubRegion");

            cl::Buffer clHaloIndicesIn(tauschcl_context, overwriteHaloRecvIndices.begin(), overwriteHaloRecvIndices.end(), true);
            cl::Buffer clHaloIndicesOut(tauschcl_context, overwriteHaloTargetIndices.begin(), overwriteHaloTargetIndices.end(), true);

            int globalsize = (remoteHaloIndicesSize[haloId]/clKernelLocalSize +1)*clKernelLocalSize;

            cl::Buffer clBufferId(tauschcl_context, &bufferId, (&bufferId)+1, true);
            cl::Buffer haloSize(tauschcl_context, &remoteHaloIndicesSize[haloId], (&remoteHaloIndicesSize[haloId])+1, true);

            kernel_unpack(cl::EnqueueArgs(tauschcl_queue, cl::NDRange(globalsize), cl::NDRange(clKernelLocalSize)),
                          recvBuffer[haloId], buf, clHaloIndicesIn, clHaloIndicesOut, haloSize);

        } catch(cl::Error e) {
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

    std::vector<std::array<int, 3> > extractHaloIndicesWithStride(std::vector<int> indices) {

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

    cl::Device tauschcl_device;
    cl::Context tauschcl_context;
    cl::CommandQueue tauschcl_queue;
    cl::Program programs;
    int clKernelLocalSize;
    std::string cName4BufT;

    std::vector<int> msgtags_keys;
    std::vector<int> msgtags_vals;

    std::vector<int > dataSent;

    std::vector<std::vector<std::array<int, 3> > > localHaloIndices;
    std::vector<int> localHaloIndicesSize;
    std::vector<cl::Buffer> remoteHaloIndices;
    std::vector<int> remoteHaloIndicesSize;
    std::vector<int> localHaloNumBuffers;
    std::vector<cl::Buffer> remoteHaloNumBuffers;

    std::vector<cl::Buffer> recvBuffer;
    std::vector<buf_t*> sendBuffer;

};

#endif
