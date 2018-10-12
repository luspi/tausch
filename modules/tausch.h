#ifndef TAUSCH_H
#define TAUSCH_H

#include "tausch_cwc.h"

template <class buf_t>
class Tausch {

public:
    Tausch(const MPI_Datatype mpiDataType, const MPI_Comm comm = MPI_COMM_WORLD, const bool useDuplicateOfCommunicator = true) {

        if(useDuplicateOfCommunicator)
            MPI_Comm_dup(comm, &TAUSCH_COMM);
        else
            TAUSCH_COMM = comm;

        tausch_cwc = new TauschCWC<buf_t>(mpiDataType, TAUSCH_COMM);

    }

    ~Tausch() {
        delete tausch_cwc;
    }

    int addLocalHaloInfo(TauschHaloRegion region, const size_t numBuffer = 1, int remoteMpiRank = -1) {
        return tausch_cwc->addLocalHaloInfo(region, numBuffer, remoteMpiRank);
    }

    int addLocalHaloInfo(std::vector<size_t> haloIndices, const size_t numBuffers = 1, int remoteMpiRank = -1) {
        return tausch_cwc->addLocalHaloInfo(haloIndices, numBuffers, remoteMpiRank);
    }

    int addRemoteHaloInfo(TauschHaloRegion region, const size_t numBuffer = 1, int remoteMpiRank = -1) {
        return tausch_cwc->addRemoteHaloInfo(region, numBuffer, remoteMpiRank);
    }

    int addRemoteHaloInfo(std::vector<size_t> haloIndices, const size_t numBuffers = 1, int remoteMpiRank = -1) {
        return tausch_cwc->addRemoteHaloInfo(haloIndices, numBuffers, remoteMpiRank);
    }

    void packSendBuffer(size_t haloId, size_t bufferId, const buf_t *buf) {
        tausch_cwc->packSendBuffer(haloId, bufferId, buf);
    }

    void packSendBuffer(const size_t haloId, const size_t bufferId, const buf_t *buf, const std::vector<size_t> overwriteHaloSendIndices, const std::vector<size_t> overwriteHaloSourceIndices) {
        tausch_cwc->packSendBuffer(haloId, bufferId, buf, overwriteHaloSendIndices, overwriteHaloSourceIndices);
    }

    void send(size_t haloId, int msgtag, int remoteMpiRank = -1) {
        tausch_cwc->send(haloId, msgtag, remoteMpiRank);
    }

    void recv(size_t haloId, int msgtag, int remoteMpiRank = -1) {
        tausch_cwc->recv(haloId, msgtag, remoteMpiRank);
    }

    void unpackRecvBuffer(size_t haloId, size_t bufferId, buf_t *buf) {
        tausch_cwc->unpackRecvBuffer(haloId, bufferId, buf);
    }

    void unpackRecvBuffer(size_t haloId, size_t bufferId, buf_t *buf, const std::vector<size_t> overwriteHaloRecvIndices, const std::vector<size_t> overwriteHaloTargetIndices) {
        tausch_cwc->unpackRecvBuffer(haloId, bufferId, buf, overwriteHaloRecvIndices, overwriteHaloTargetIndices);
    }

    void packAndSend(const size_t haloId, buf_t *buf, const int msgtag, int remoteMpiRank = -1) {
        tausch_cwc->packAndSend(haloId, buf, msgtag, remoteMpiRank);
    }

    void recvAndUnpack(const size_t haloId, buf_t *buf, const int msgtag, int remoteMpiRank = -1) {
        tausch_cwc->recvAndUnpack(haloId, buf, msgtag, remoteMpiRank);
    }

    size_t getNumLocalHalo() {
        return tausch_cwc->getNumLocalHalo();
    }
    size_t getNumRemoteHalo() {
        return tausch_cwc->getNumRemoteHalo();
    }
    size_t getSizeLocalHalo(size_t haloId) {
        return tausch_cwc->getSizeLocalHalo(haloId);
    }
    size_t getSizeRemoteHalo(size_t haloId) {
        return tausch_cwc->getSizeRemoteHalo(haloId);
    }
    size_t getNumBuffersLocal(size_t haloId) {
        return tausch_cwc->getNumBuffersLocal(haloId);
    }
    size_t getNumBuffersRemote(size_t haloId) {
        return tausch_cwc->getNumBuffersRemote(haloId);
    }
    buf_t *getSendBuffer(size_t haloId) {
        return tausch_cwc->getSendBuffer(haloId);
    }
    buf_t *getRecvBuffer(size_t haloId) {
        return tausch_cwc->getRecvBuffer(haloId);
    }

private:
    MPI_Comm TAUSCH_COMM;

    TauschCWC<buf_t> *tausch_cwc;

};


#endif
