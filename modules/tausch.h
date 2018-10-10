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

    void packSendBuffer(size_t haloId, size_t bufferId, buf_t *buf) {
        tausch_cwc->packSendBuffer(haloId, bufferId, buf);
    }

    void packSendBuffer(const size_t haloId, const size_t bufferId, const buf_t *buf, const std::vector<size_t> overwriteHaloIndices, const size_t overwriteHaloOffset) {
        tausch_cwc->packSendBuffer(haloId, bufferId, buf, overwriteHaloIndices, overwriteHaloOffset);
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

    void unpackRecvBuffer(size_t haloId, size_t bufferId, buf_t *buf, const std::vector<size_t> overwriteHaloIndices, const size_t overwriteHaloOffset) {
        tausch_cwc->unpackRecvBuffer(haloId, bufferId, buf, overwriteHaloIndices, overwriteHaloOffset);
    }

    void packAndSend(const size_t haloId, buf_t *buf, const int msgtag, int remoteMpiRank = -1) {
        tausch_cwc->packAndSend(haloId, buf, msgtag, remoteMpiRank);
    }

    void recvAndUnpack(const size_t haloId, buf_t *buf, const int msgtag, int remoteMpiRank = -1) {
        tausch_cwc->recvAndUnpack(haloId, buf, msgtag, remoteMpiRank);
    }

private:
    MPI_Comm TAUSCH_COMM;

    TauschCWC<buf_t> *tausch_cwc;

};


#endif
