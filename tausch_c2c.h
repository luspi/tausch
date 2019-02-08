#ifndef TAUSCH_C2C_H
#define TAUSCH_C2C_H

#include <mpi.h>
#include <vector>
#include <array>
#include <cstring>
#include "tauschdefs.h"

template <class buf_t>
class TauschC2C {

public:
    TauschC2C(const MPI_Datatype mpiDataType, const MPI_Comm comm) {

        this->mpiDataType = mpiDataType;
        TAUSCH_COMM = comm;

    }

    ~TauschC2C() {

        for(size_t i = 0; i < mpiSendBuffer.size(); ++i)
            free(mpiSendBuffer[i]);
        for(size_t i = 0; i < mpiRecvBuffer.size(); ++i)
            free(mpiRecvBuffer[i]);
        for(size_t i = 0; i < mpiSendRequests.size(); ++i)
            delete mpiSendRequests[i];
        for(size_t i = 0; i < mpiRecvRequests.size(); ++i)
            delete mpiRecvRequests[i];

    }

    int addLocalHaloInfo(const TauschHaloRegion region, const size_t numBuffer, const int remoteMpiRank) {

        std::vector<std::array<int, 3> > haloIndices;

        // 1D
        if(region.dimensions == 1) {

            std::array<int, 3> vals = {static_cast<int>(region.haloX), static_cast<int>(region.haloWidth), 1};
            haloIndices.push_back(vals);

        // 2D
        } else if(region.dimensions == 2) {

            for(size_t y = 0; y < region.haloHeight; ++y) {
                std::array<int, 3> vals = {static_cast<int>((region.haloY+y)*region.bufferWidth + region.haloX), static_cast<int>(region.haloWidth), 1};
                haloIndices.push_back(vals);
            }

        // 3D
        } else if(region.dimensions == 3) {

            for(size_t z = 0; z < region.haloDepth; ++z) {
                for(size_t y = 0; y < region.haloHeight; ++y) {
                    int startIndex = static_cast<int>((region.haloZ+z)*region.bufferWidth*region.bufferHeight + (region.haloY+y)*region.bufferWidth + region.haloX);
                    std::array<int, 3> vals = {startIndex, static_cast<int>(region.haloWidth), 1};
                    haloIndices.push_back(vals);
                }
            }

        } else
            std::cout << "[Tausch] ERROR: Invalid dimension specified in TauschHaloRegion!" << std::endl;

        return addLocalHaloInfo(haloIndices, numBuffer, (region.remoteMpiRank==-1 ? remoteMpiRank : region.remoteMpiRank));

    }

    int addLocalHaloInfo(std::vector<int> haloIndices, const size_t numBuffers, const int remoteMpiRank) {

        return addLocalHaloInfo(extractHaloIndicesWithStride(haloIndices), numBuffers, remoteMpiRank);

    }

    int addLocalHaloInfo(const std::vector<std::array<int, 3> > haloIndices, const size_t numBuffers, const int remoteMpiRank) {

        localHaloIndices.push_back(haloIndices);
        localHaloRemoteMpiRank.push_back(remoteMpiRank);
        localHaloNumBuffers.push_back(numBuffers);

        size_t bufsize = 0;
        for(size_t i = 0; i < haloIndices.size(); ++i)
            bufsize += static_cast<size_t>(haloIndices.at(i)[1]);

        localHaloIndicesSize.push_back(bufsize);

        void *newbuf = NULL;
        posix_memalign(&newbuf, 64, numBuffers*bufsize*sizeof(buf_t));
        buf_t *newbuf_buft = reinterpret_cast<buf_t*>(newbuf);
        double zero = 0;
        std::fill_n(newbuf_buft, numBuffers*bufsize, zero);
        mpiSendBuffer.push_back(newbuf_buft);

        mpiSendRequests.push_back(new MPI_Request());

        setupMpiSend.push_back(false);

        return mpiSendBuffer.size()-1;

    }

    int addRemoteHaloInfo(const TauschHaloRegion region, const size_t numBuffer, const int remoteMpiRank) {

        std::vector<std::array<int, 3> > haloIndices;

        // 1D
        if(region.haloDepth == 0 && region.haloHeight == 0) {

            std::array<int, 3> vals = {static_cast<int>(region.haloX), static_cast<int>(region.haloWidth), 1};
            haloIndices.push_back(vals);

        // 2D
        } else if(region.haloDepth == 0) {

            for(size_t y = 0; y < region.haloHeight; ++y) {
                size_t startIndex = (region.haloY+y)*region.bufferWidth + region.haloX;
                std::array<int, 3> vals = {static_cast<int>(startIndex), static_cast<int>(region.haloWidth), 1};
                haloIndices.push_back(vals);
            }

        // 3D
        } else {

            for(size_t z = 0; z < region.haloDepth; ++z) {
                for(size_t y = 0; y < region.haloHeight; ++y) {
                    size_t startIndex = (region.haloZ+z)*region.bufferWidth*region.bufferHeight + (region.haloY+y)*region.bufferWidth + region.haloX;
                    std::array<int, 3> vals = {static_cast<int>(startIndex), static_cast<int>(region.haloWidth), 1};
                    haloIndices.push_back(vals);
                }
            }

        }

        return addRemoteHaloInfo(haloIndices, numBuffer, (region.remoteMpiRank==-1 ? remoteMpiRank : region.remoteMpiRank));

    }

    int addRemoteHaloInfo(std::vector<int> haloIndices, const size_t numBuffers, const int remoteMpiRank) {

        return addRemoteHaloInfo(extractHaloIndicesWithStride(haloIndices), numBuffers, remoteMpiRank);

    }

    int addRemoteHaloInfo(const std::vector<std::array<int, 3> > haloIndices, const size_t numBuffers, const int remoteMpiRank) {

        remoteHaloIndices.push_back(haloIndices);
        remoteHaloRemoteMpiRank.push_back(remoteMpiRank);
        remoteHaloNumBuffers.push_back(numBuffers);

        size_t bufsize = 0;
        for(size_t i = 0; i < haloIndices.size(); ++i)
            bufsize += static_cast<size_t>(haloIndices.at(i)[1]);
        remoteHaloIndicesSize.push_back(bufsize);

        void *newbuf = NULL;
        posix_memalign(&newbuf, 64, numBuffers*bufsize*sizeof(buf_t));
        buf_t *newbuf_buft = reinterpret_cast<buf_t*>(newbuf);
        double zero = 0;
        std::fill_n(newbuf_buft, numBuffers*bufsize, zero);
        mpiRecvBuffer.push_back(newbuf_buft);

        mpiRecvRequests.push_back(new MPI_Request());

        setupMpiRecv.push_back(false);

        return mpiRecvBuffer.size()-1;

    }

    void packSendBuffer(const size_t haloId, const size_t bufferId, const buf_t *buf) {

        const size_t haloSize = localHaloIndicesSize[haloId];

        size_t mpiSendBufferIndex = 0;
        for(size_t region = 0; region < localHaloIndices[haloId].size(); ++region) {
            const std::array<int, 3> vals = localHaloIndices[haloId][region];

            const int val_start = vals[0];
            const int val_howmany = vals[1];
            const int val_stride = vals[2];

            if(val_stride == 1) {
                memcpy(&mpiSendBuffer[haloId][bufferId*haloSize + mpiSendBufferIndex], &buf[val_start], val_howmany*sizeof(buf_t));
                mpiSendBufferIndex += val_howmany;
            } else {
                const int mpiSendBufferIndexBASE = bufferId*haloSize + mpiSendBufferIndex;
                for(int i = 0; i < val_howmany; ++i)
                    mpiSendBuffer[haloId][mpiSendBufferIndexBASE + i] = buf[val_start+i*val_stride];
                mpiSendBufferIndex += val_howmany;
            }

        }

    }

    void packSendBuffer(const size_t haloId, const size_t bufferId, const buf_t *buf, const std::vector<size_t> overwriteHaloSendIndices, const std::vector<size_t> overwriteHaloSourceIndices) {

        size_t haloSize = localHaloIndicesSize[haloId];

        for(size_t index = 0; index < overwriteHaloSendIndices.size(); ++index)
            mpiSendBuffer[haloId][bufferId*haloSize + overwriteHaloSendIndices[index]] = buf[overwriteHaloSourceIndices[index]];

    }

    MPI_Request *send(const size_t haloId, const int msgtag, int remoteMpiRank) {

        if(localHaloIndices[haloId].size() == 0)
            return nullptr;

        if(!setupMpiSend[haloId]) {

            setupMpiSend[haloId] = true;

            if(remoteMpiRank == -1)
                remoteMpiRank = localHaloRemoteMpiRank[haloId];

            MPI_Send_init(&mpiSendBuffer[haloId][0], localHaloNumBuffers[haloId]*localHaloIndicesSize[haloId], mpiDataType, remoteMpiRank,
                      msgtag, TAUSCH_COMM, mpiSendRequests[haloId]);

        } else
            MPI_Wait(mpiSendRequests[haloId], MPI_STATUS_IGNORE);

        MPI_Start(mpiSendRequests[haloId]);

        return mpiSendRequests[haloId];

    }

    MPI_Request *recv(const size_t haloId, const int msgtag, int remoteMpiRank, bool blocking) {

        if(remoteHaloIndices[haloId].size() == 0)
            return nullptr;

        if(!setupMpiRecv[haloId]) {

            setupMpiRecv[haloId] = true;

            if(remoteMpiRank == -1)
                remoteMpiRank = remoteHaloRemoteMpiRank[haloId];

            MPI_Recv_init(&mpiRecvBuffer[haloId][0], remoteHaloNumBuffers[haloId]*remoteHaloIndicesSize[haloId], mpiDataType,
                          remoteMpiRank, msgtag, TAUSCH_COMM, mpiRecvRequests[haloId]);

        }

        MPI_Start(mpiRecvRequests[haloId]);
        if(blocking)
            MPI_Wait(mpiRecvRequests[haloId], MPI_STATUS_IGNORE);

        return mpiRecvRequests[haloId];

    }

    void unpackRecvBuffer(const size_t haloId, const size_t bufferId, buf_t *buf) {

        size_t haloSize = remoteHaloIndicesSize[haloId];

        size_t mpiRecvBufferIndex = 0;
        for(size_t region = 0; region < remoteHaloIndices[haloId].size(); ++region) {
            const std::array<int, 3> vals = remoteHaloIndices[haloId][region];

            const int val_start = vals[0];
            const int val_howmany = vals[1];
            const int val_stride = vals[2];

            if(val_stride == 1) {
                memcpy(&buf[val_start], &mpiRecvBuffer[haloId][bufferId*haloSize + mpiRecvBufferIndex], val_howmany*sizeof(buf_t));
                mpiRecvBufferIndex += val_howmany;
            } else {
                const size_t mpirecvBufferIndexBASE = bufferId*haloSize + mpiRecvBufferIndex;
                for(int i = 0; i < val_howmany; ++i)
                    buf[val_start+i*val_stride] = mpiRecvBuffer[haloId][mpirecvBufferIndexBASE + i];
                mpiRecvBufferIndex += val_howmany;
            }

        }

    }

    void unpackRecvBuffer(const size_t haloId, const size_t bufferId, buf_t *buf, const std::vector<size_t> overwriteHaloRecvIndices, const std::vector<size_t> overwriteHaloTargetIndices) {

        size_t haloSize = remoteHaloIndicesSize[haloId];

        for(size_t index = 0; index < overwriteHaloRecvIndices.size(); ++index)
            buf[overwriteHaloTargetIndices[index]] = mpiRecvBuffer[haloId][bufferId*haloSize + overwriteHaloRecvIndices[index]];

    }

    void packAndSend(const size_t haloId, buf_t *buf, const int msgtag, int remoteMpiRank) {
        packSendBuffer(haloId, 0, buf);
        send(haloId, msgtag, remoteMpiRank);
    }

    void recvAndUnpack(const size_t haloId, buf_t *buf, const int msgtag, int remoteMpiRank) {
        recv(haloId, msgtag, remoteMpiRank);
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

    size_t getNumLocalHalo() {
        return mpiSendBuffer.size();
    }
    size_t getNumRemoteHalo() {
        return mpiRecvBuffer.size();
    }
    size_t getSizeLocalHalo(size_t haloId) {
        return localHaloIndices[haloId].size();
    }
    size_t getSizeRemoteHalo(size_t haloId) {
        return remoteHaloIndices[haloId].size();
    }
    size_t getNumBuffersLocal(size_t haloId) {
        return localHaloNumBuffers[haloId];
    }
    size_t getNumBuffersRemote(size_t haloId) {
        return remoteHaloNumBuffers[haloId];
    }
    buf_t *getSendBuffer(size_t haloId) {
        return mpiSendBuffer[haloId];
    }
    buf_t *getRecvBuffer(size_t haloId) {
        return mpiRecvBuffer[haloId];
    }

private:
    MPI_Datatype mpiDataType;
    MPI_Comm TAUSCH_COMM;

#ifdef TAUSCH_OLD
    std::vector<std::vector<int> > localHaloIndices;
    std::vector<std::vector<int> > remoteHaloIndices;
#else
    std::vector<std::vector<std::array<int, 3> > > localHaloIndices;
    std::vector<size_t> localHaloIndicesSize;
    std::vector<std::vector<std::array<int, 3> > > remoteHaloIndices;
    std::vector<size_t> remoteHaloIndicesSize;
#endif
    std::vector<int> localHaloRemoteMpiRank;
    std::vector<int> remoteHaloRemoteMpiRank;
    std::vector<size_t> localHaloNumBuffers;
    std::vector<size_t> remoteHaloNumBuffers;

    std::vector<buf_t*> mpiRecvBuffer;
    std::vector<buf_t*> mpiSendBuffer;
    std::vector<MPI_Request*> mpiRecvRequests;
    std::vector<MPI_Request*> mpiSendRequests;

    std::vector<bool> setupMpiSend;
    std::vector<bool> setupMpiRecv;

};

#endif
