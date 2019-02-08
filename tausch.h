#ifndef TAUSCH_H
#define TAUSCH_H

#include "tausch_c2c.h"
#ifdef TAUSCH_OPENCL
#include "tausch_c2g.h"
#include "tausch_g2c.h"
#include "tausch_g2g.h"
#endif

template <class buf_t>
class Tausch {

public:
#ifndef TAUSCH_OPENCL
    Tausch(const MPI_Datatype mpiDataType, const MPI_Comm comm = MPI_COMM_WORLD, const bool useDuplicateOfCommunicator = true) {
#else
    Tausch(cl::Device device, cl::Context context, cl::CommandQueue queue, std::string cName4BufT,
            const MPI_Datatype mpiDataType, const MPI_Comm comm = MPI_COMM_WORLD, const bool useDuplicateOfCommunicator = true) {
#endif

        if(useDuplicateOfCommunicator)
            MPI_Comm_dup(comm, &TAUSCH_COMM);
        else
            TAUSCH_COMM = comm;

        tausch_cwc = new TauschC2C<buf_t>(mpiDataType, TAUSCH_COMM);
#ifdef TAUSCH_OPENCL
        tausch_c2g = new TauschC2G<buf_t>(device, context, queue, cName4BufT);
        tausch_g2c = new TauschG2C<buf_t>(device, context, queue, cName4BufT);
        tausch_g2g = new TauschG2G<buf_t>(device, context, queue, cName4BufT);
#endif
    }

    ~Tausch() {
        delete tausch_cwc;
#ifdef TAUSCH_OPENCL
        delete tausch_c2g;
        delete tausch_g2c;
        delete tausch_g2g;
#endif
    }

    TauschC2C<buf_t> *tausch_cwc;

    int addLocalHaloInfo(TauschHaloRegion region, const size_t numBuffer = 1, const int remoteMpiRank = -1) {
        return tausch_cwc->addLocalHaloInfo(region, numBuffer, remoteMpiRank);
    }

    int addLocalHaloInfo(std::vector<int> haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1) {
        return tausch_cwc->addLocalHaloInfo(haloIndices, numBuffers, remoteMpiRank);
    }

    int addLocalHaloInfo(std::vector<size_t> haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1) {
        return tausch_cwc->addLocalHaloInfo(std::vector<int>(haloIndices.begin(), haloIndices.end()), numBuffers, remoteMpiRank);
    }

    int addRemoteHaloInfo(TauschHaloRegion region, const size_t numBuffer = 1, const int remoteMpiRank = -1) {
        return tausch_cwc->addRemoteHaloInfo(region, numBuffer, remoteMpiRank);
    }

    int addRemoteHaloInfo(std::vector<int> haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1) {
        return tausch_cwc->addRemoteHaloInfo(haloIndices, numBuffers, remoteMpiRank);
    }

    int addRemoteHaloInfo(std::vector<size_t> haloIndices, const size_t numBuffers = 1, const int remoteMpiRank = -1) {
        return tausch_cwc->addRemoteHaloInfo(std::vector<int>(haloIndices.begin(), haloIndices.end()), numBuffers, remoteMpiRank);
    }

    void packSendBuffer(size_t haloId, size_t bufferId, const buf_t *buf) {
        tausch_cwc->packSendBuffer(haloId, bufferId, buf);
    }

    void packSendBuffer(const size_t haloId, const size_t bufferId, const buf_t *buf, const std::vector<size_t> overwriteHaloSendIndices, const std::vector<size_t> overwriteHaloSourceIndices) {
        tausch_cwc->packSendBuffer(haloId, bufferId, buf, overwriteHaloSendIndices, overwriteHaloSourceIndices);
    }

    MPI_Request *send(size_t haloId, int msgtag, int remoteMpiRank = -1) {
        return tausch_cwc->send(haloId, msgtag, remoteMpiRank);
    }

    MPI_Request *recv(size_t haloId, int msgtag, int remoteMpiRank = -1, bool blocking = true) {
        return tausch_cwc->recv(haloId, msgtag, remoteMpiRank, blocking);
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

    std::vector<std::array<int, 3> > extractHaloIndicesWithStride(std::vector<size_t> indices) {
        return tausch_cwc->extractHaloIndicesWithStride(indices);
    }

#ifdef TAUSCH_OPENCL
    TauschC2G<buf_t> *tausch_c2g;

    int addLocalHaloInfoC2G(const TauschHaloRegion region, const int numBuffer = 1) {
        return tausch_c2g->addLocalHaloInfo(region, numBuffer);
    }
    int addLocalHaloInfoC2G(const std::vector<int> haloIndices, const int numBuffers = 1) {
        return tausch_c2g->addLocalHaloInfo(haloIndices, numBuffers);
    }
    int addRemoteHaloInfoC2G(const TauschHaloRegion region, int numBuffer = 1) {
        return tausch_c2g->addRemoteHaloInfo(region, numBuffer);
    }
    int addRemoteHaloInfoC2G(std::vector<int> haloIndices, int numBuffers = 1) {
        return tausch_c2g->addRemoteHaloInfo(haloIndices, numBuffers);
    }
    void packSendBufferC2G(const int haloId, const int bufferId, const buf_t *buf) {
        tausch_c2g->packSendBuffer(haloId, bufferId, buf);
    }
    void packSendBufferC2G(const int haloId, const int bufferId, const buf_t *buf, const std::vector<int> overwriteHaloSendIndices, const std::vector<int> overwriteHaloSourceIndices) {
        tausch_c2g->packSendBuffer(haloId, bufferId, buf, overwriteHaloSendIndices, overwriteHaloSourceIndices);
    }
    void sendC2G(const int haloId, int msgtag) {
        tausch_c2g->send(haloId, msgtag);
    }
    void recvC2G(const int haloId, int msgtag) {
        tausch_c2g->recv(haloId, msgtag);
    }
    void unpackRecvBufferC2G(const int haloId, int bufferId, cl::Buffer buf) {
        tausch_c2g->unpackRecvBuffer(haloId, bufferId, buf);
    }
    void unpackRecvBufferC2G(const int haloId, int bufferId, cl::Buffer buf, const std::vector<int> overwriteHaloRecvIndices, const std::vector<int> overwriteHaloTargetIndices) {
        tausch_c2g->unpackRecvBuffer(haloId, bufferId, buf, overwriteHaloRecvIndices, overwriteHaloTargetIndices);
    }
    void packAndSendC2G(const int haloId, buf_t *buf, const int msgtag) {
        tausch_c2g->packAndSend(haloId, buf, msgtag);
    }
    void recvAndUnpackC2G(const int haloId, cl::Buffer buf, const int msgtag) {
        tausch_c2g->recvAndUnpack(haloId, buf, msgtag);
    }

    TauschG2C<buf_t> *tausch_g2c;

    int addLocalHaloInfoG2C(const TauschHaloRegion region, int numBuffer = 1) {
        return tausch_g2c->addLocalHaloInfo(region, numBuffer);
    }
    int addLocalHaloInfoG2C(std::vector<int> haloIndices, int numBuffers = 1) {
        return tausch_g2c->addLocalHaloInfo(haloIndices, numBuffers);
    }
    int addRemoteHaloInfoG2C(const TauschHaloRegion region, const int numBuffer = 1) {
        return tausch_g2c->addRemoteHaloInfo(region, numBuffer);
    }
    int addRemoteHaloInfoG2C(const std::vector<int> haloIndices, const int numBuffers = 1) {
        return tausch_g2c->addRemoteHaloInfo(haloIndices, numBuffers);
    }
    void packSendBufferG2C(const int haloId, int bufferId, const cl::Buffer buf) {
        tausch_g2c->packSendBuffer(haloId, bufferId, buf);
    }
    void packSendBufferG2C(const int haloId, int bufferId, const cl::Buffer buf, const std::vector<int> overwriteHaloSendIndices, const std::vector<int> overwriteHaloSourceIndices) {
        tausch_g2c->packSendBuffer(haloId, bufferId, buf, overwriteHaloSendIndices, overwriteHaloSourceIndices);
    }
    void sendG2C(const int haloId, int msgtag) {
        tausch_g2c->send(haloId, msgtag);
    }
    void recvG2C(const int haloId, int msgtag) {
        tausch_g2c->recv(haloId, msgtag);
    }
    void unpackRecvBufferG2C(const int haloId, const int bufferId, buf_t *buf) {
        tausch_g2c->unpackRecvBuffer(haloId, bufferId, buf);
    }
    void unpackRecvBufferG2C(const int haloId, const int bufferId, buf_t *buf, const std::vector<int> overwriteHaloRecvIndices, const std::vector<int> overwriteHaloTargetIndices) {
        tausch_g2c->unpackRecvBuffer(haloId, bufferId, overwriteHaloRecvIndices, overwriteHaloTargetIndices);
    }
    void packAndSendG2C(const int haloId, cl::Buffer buf, const int msgtag) {
        tausch_g2c->packAndSend(haloId, buf, msgtag);
    }
    void recvAndUnpackG2C(const int haloId, buf_t *buf, const int msgtag) {
        tausch_g2c->recvAndUnpack(haloId, buf, msgtag);
    }

    TauschG2G<buf_t> *tausch_g2g;

    int addLocalHaloInfoG2G(const TauschHaloRegion region, int numBuffer = 1) {
        return tausch_g2g->addLocalHaloInfo(region, numBuffer);
    }
    int addLocalHaloInfoG2G(std::vector<int> haloIndices, int numBuffers = 1) {
        return tausch_g2g->addLocalHaloInfo(haloIndices, numBuffers);
    }
    int addRemoteHaloInfoG2G(const TauschHaloRegion region, int numBuffer = 1) {
        return tausch_g2g->addRemoteHaloInfo(region, numBuffer);
    }
    int addRemoteHaloInfoG2G(std::vector<int> haloIndices, int numBuffers = 1) {
        return tausch_g2g->addRemoteHaloInfo(haloIndices, numBuffers);
    }
    void packSendBufferG2G(const int haloId, int bufferId, const cl::Buffer buf) {
        tausch_g2g->packSendBuffer(haloId, bufferId, buf);
    }
    void packSendBufferG2G(const int haloId, int bufferId, const cl::Buffer buf, const std::vector<int> overwriteHaloSendIndices, const std::vector<int> overwriteHaloSourceIndices) {
        tausch_g2g->packSendBuffer(haloId, bufferId, buf, overwriteHaloSendIndices, overwriteHaloSourceIndices);
    }
    void sendG2G(const int haloId, int msgtag) {
        tausch_g2g->send(haloId, msgtag);
    }
    void recvG2G(const int haloId, int msgtag) {
        tausch_g2g->recv(haloId, msgtag);
    }
    void unpackRecvBufferG2G(const int haloId, int bufferId, cl::Buffer buf) {
        tausch_g2g->unpackRecvBuffer(haloId, bufferId, buf);
    }
    void unpackRecvBufferG2G(const int haloId, int bufferId, cl::Buffer buf, const std::vector<int> overwriteHaloRecvIndices, const std::vector<int> overwriteHaloTargetIndices) {
        tausch_g2g->unpackRecvBuffer(haloId, bufferId, overwriteHaloRecvIndices, overwriteHaloTargetIndices);
    }
    void packAndSendG2G(const int haloId, cl::Buffer buf, const int msgtag) {
        tausch_g2g->packAndSend(haloId, buf, msgtag);
    }
    void recvAndUnpackG2G(const int haloId, cl::Buffer buf, const int msgtag) {
        tausch_g2g->recvAndUnpack(haloId, buf, msgtag);
    }

#endif

private:
    MPI_Comm TAUSCH_COMM;

};


#endif
