#include "../tausch1d.h"
#include "../tausch2d.h"
#include "../tausch3d.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctauschbool.h"

CTauschBool *tausch_new_bool(size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm, TauschVersion version) {

    if(version != TAUSCH_1D && version != TAUSCH_2D && version != TAUSCH_3D) {
        std::cerr << "[CTauschBool] ERROR: Invalid version specified: " << version << " - Abort..." << std::endl;
        exit(1);
    }

    Tausch<bool> *t;

    if(version == TAUSCH_1D)
        t = new Tausch1D<bool>(MPI_DOUBLE, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_2D)
        t = new Tausch2D<bool>(MPI_DOUBLE, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_3D)
        t = new Tausch3D<bool>(MPI_DOUBLE, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);

    return reinterpret_cast<CTauschBool*>(t);

}

void tausch_delete_bool(CTauschBool *tC) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    delete t;
}

void tausch_setCpuLocalHaloInfo_bool(CTauschBool *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->setLocalHaloInfoCpu(numHaloParts, haloSpecs);
}

void tausch_setCpuRemoteHaloInfo_bool(CTauschBool *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->setRemoteHaloInfoCpu(numHaloParts, haloSpecs);
}

void tausch_postReceiveCpu_bool(CTauschBool *tC, size_t haloId, int mpitag) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->postReceiveCpu(haloId, mpitag);
}

void tausch_postAllReceivesCpu_bool(CTauschBool *tC, int *mpitag) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->postAllReceivesCpu(mpitag);
}

void tausch_packNextSendBuffer_bool(CTauschBool *tC, size_t haloId, size_t bufferId, bool *buf, TauschPackRegion region) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->packSendBufferCpu(haloId, bufferId, buf, region);
}

void tausch_send_bool(CTauschBool *tC, size_t haloId, int mpitag) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->sendCpu(haloId, mpitag);
}

void tausch_recv_bool(CTauschBool *tC, size_t haloId) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->recvCpu(haloId);
}

void tausch_unpackNextRecvBuffer_bool(CTauschBool *tC, size_t haloId, size_t bufferId, bool *buf) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->unpackRecvBufferCpu(haloId, bufferId, buf);
}

void tausch_packAndSend_bool(CTauschBool *tC, size_t haloId, bool *buf, TauschPackRegion region, int mpitag) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->packAndSendCpu(haloId, buf, region, mpitag);
}

void tausch_recvAndUnpack_bool(CTauschBool *tC, size_t haloId, bool *buf) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->recvAndUnpackCpu(haloId, buf);
}


#ifdef __cplusplus
}
#endif

