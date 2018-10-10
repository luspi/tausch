#ifndef TAUSCH_CWC_H
#define TAUSCH_CWC_H

#include <mpi.h>
#include <vector>
#include "tauschdefs.h"

template <class buf_t>
class TauschCWC {

public:
    TauschCWC(const MPI_Datatype mpiDataType, const MPI_Comm comm) {

        this->mpiDataType = mpiDataType;
        TAUSCH_COMM = comm;

    }

    ~TauschCWC() {

        for(int i = 0; i < mpiSendBuffer.size(); ++i)
            delete mpiSendBuffer[i];
        for(int i = 0; i < mpiRecvBuffer.size(); ++i)
            delete mpiRecvBuffer[i];

    }

    int addLocalHaloInfo(const TauschHaloRegion region, const size_t numBuffer, const int remoteMpiRank) {

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

        return addLocalHaloInfo(haloIndices, numBuffer, remoteMpiRank);

    }

    int addLocalHaloInfo(const std::vector<size_t> haloIndices, const size_t numBuffers, const int remoteMpiRank) {

        localHaloIndices.push_back(haloIndices);
        localHaloRemoteMpiRank.push_back(remoteMpiRank);
        localHaloNumBuffers.push_back(numBuffers);

        mpiSendBuffer.push_back(new buf_t[numBuffers * haloIndices.size()]());

        mpiSendRequests.push_back(MPI_REQUEST_NULL);

        setupMpiSend.push_back(false);

        return mpiSendBuffer.size()-1;

    }

    int addRemoteHaloInfo(const TauschHaloRegion region, const size_t numBuffer, const int remoteMpiRank) {

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

        return addRemoteHaloInfo(haloIndices, numBuffer, remoteMpiRank);

    }

    int addRemoteHaloInfo(const std::vector<size_t> haloIndices, const size_t numBuffers, const int remoteMpiRank) {

        remoteHaloIndices.push_back(haloIndices);
        remoteHaloRemoteMpiRank.push_back(remoteMpiRank);
        remoteHaloNumBuffers.push_back(numBuffers);

        mpiRecvBuffer.push_back(new buf_t[numBuffers * haloIndices.size()]());

        mpiRecvRequests.push_back(MPI_REQUEST_NULL);

        setupMpiRecv.push_back(false);

        return mpiRecvBuffer.size()-1;

    }

    void packSendBuffer(const size_t haloId, const size_t bufferId, const buf_t *buf) {

        size_t haloSize = localHaloIndices[haloId].size();

        for(size_t index = 0; index < haloSize; ++index)
            mpiSendBuffer[haloId][bufferId*haloSize + index] = buf[localHaloIndices[haloId][index]];

    }

    void packSendBuffer(const size_t haloId, const size_t bufferId, const buf_t *buf, const std::vector<size_t> overwriteHaloIndices, const size_t overwriteHaloOffset) {

        size_t haloSize = localHaloIndices[haloId].size();

        for(size_t index = 0; index < overwriteHaloIndices.size(); ++index)
            mpiSendBuffer[haloId][bufferId*haloSize + overwriteHaloOffset + index] = buf[overwriteHaloIndices[index]];

    }

    void send(const size_t haloId, const int msgtag, int remoteMpiRank) {

        if(!setupMpiSend[haloId]) {

            setupMpiSend[haloId] = true;

            if(remoteMpiRank == -1)
                remoteMpiRank = localHaloRemoteMpiRank[haloId];

            MPI_Send_init(&mpiSendBuffer[haloId][0], localHaloNumBuffers[haloId]*localHaloIndices[haloId].size(), mpiDataType, remoteMpiRank,
                      msgtag, TAUSCH_COMM, &mpiSendRequests[haloId]);

        } else
            MPI_Wait(&mpiSendRequests[haloId], MPI_STATUS_IGNORE);

        MPI_Start(&mpiSendRequests[haloId]);

    }

    void recv(const size_t haloId, const int msgtag, int remoteMpiRank) {

        if(!setupMpiRecv[haloId]) {

            setupMpiRecv[haloId] = true;

            if(remoteMpiRank == -1)
                remoteMpiRank = remoteHaloRemoteMpiRank[haloId];

            MPI_Recv_init(&mpiRecvBuffer[haloId][0], remoteHaloNumBuffers[haloId]*remoteHaloIndices[haloId].size(), mpiDataType,
                          remoteMpiRank, msgtag, TAUSCH_COMM, &mpiRecvRequests[haloId]);

        }

        MPI_Start(&mpiRecvRequests[haloId]);
        MPI_Wait(&mpiRecvRequests[haloId], MPI_STATUS_IGNORE);

    }

    void unpackRecvBuffer(const size_t haloId, const size_t bufferId, buf_t *buf) {

        size_t haloSize = remoteHaloIndices[haloId].size();

        for(size_t index = 0; index < haloSize; ++index)
            buf[remoteHaloIndices[haloId][index]] = mpiRecvBuffer[haloId][bufferId*haloSize + index];

    }

    void unpackRecvBuffer(const size_t haloId, const size_t bufferId, buf_t *buf, const std::vector<size_t> overwriteHaloIndices, const size_t overwriteHaloOffset) {

        size_t haloSize = remoteHaloIndices[haloId].size();

        for(size_t index = 0; index < overwriteHaloIndices.size(); ++index)
            buf[overwriteHaloIndices[index]] = mpiRecvBuffer[haloId][bufferId*haloSize + overwriteHaloOffset + index];

    }

    void packAndSend(const size_t haloId, buf_t *buf, const int msgtag, int remoteMpiRank) {
        packSendBuffer(haloId, 0, buf);
        send(haloId, msgtag, remoteMpiRank);
    }

    void recvAndUnpack(const size_t haloId, buf_t *buf, const int msgtag, int remoteMpiRank) {
        recv(haloId, msgtag, remoteMpiRank);
        unpackRecvBuffer(haloId, 0, buf);
    }

private:
    MPI_Datatype mpiDataType;
    MPI_Comm TAUSCH_COMM;

    std::vector<std::vector<size_t> > localHaloIndices;
    std::vector<std::vector<size_t> > remoteHaloIndices;
    std::vector<int> localHaloRemoteMpiRank;
    std::vector<int> remoteHaloRemoteMpiRank;
    std::vector<unsigned int> localHaloNumBuffers;
    std::vector<unsigned int> remoteHaloNumBuffers;

    std::vector<buf_t*> mpiRecvBuffer;
    std::vector<buf_t*> mpiSendBuffer;
    std::vector<MPI_Request> mpiRecvRequests;
    std::vector<MPI_Request> mpiSendRequests;

    std::vector<bool> setupMpiSend;
    std::vector<bool> setupMpiRecv;

};

#endif
