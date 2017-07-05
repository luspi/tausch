#include "../tausch1d.h"
#include "../tausch2d.h"
#include "../tausch3d.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctauschint.h"

CTauschInt *tausch_new_int(size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm, TauschVersion version) {

    if(version != TAUSCH_1D && version != TAUSCH_2D && version != TAUSCH_3D) {
        std::cerr << "[CTauschInt] ERROR: Invalid version specified: " << version << " - Abort..." << std::endl;
        exit(1);
    }

    Tausch<int> *t;

    if(version == TAUSCH_1D)
        t = new Tausch1D<int>(MPI_INT, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_2D)
        t = new Tausch2D<int>(MPI_INT, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_3D)
        t = new Tausch3D<int>(MPI_INT, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);

    return reinterpret_cast<CTauschInt*>(t);

}

void tausch_delete_int(CTauschInt *tC) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    delete t;
}

void tausch_setLocalHaloInfoCpu_int(CTauschInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->setLocalHaloInfoCpu(numHaloParts, haloSpecs);
}

void tausch_setRemoteHaloInfoCpu_int(CTauschInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->setRemoteHaloInfoCpu(numHaloParts, haloSpecs);
}

void tausch_postReceiveCpu_int(CTauschInt *tC, size_t haloId, int mpitag) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->postReceiveCpu(haloId, mpitag);
}

void tausch_postAllReceivesCpu_int(CTauschInt *tC, int *mpitag) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->postAllReceivesCpu(mpitag);
}

void tausch_packSendBufferCpu_int(CTauschInt *tC, size_t haloId, size_t bufferId, int *buf, TauschPackRegion region) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->packSendBufferCpu(haloId, bufferId, buf, region);
}

void tausch_sendCpu_int(CTauschInt *tC, size_t haloId, int mpitag) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->sendCpu(haloId, mpitag);
}

void tausch_recvCpu_int(CTauschInt *tC, size_t haloId) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->recvCpu(haloId);
}

void tausch_unpackRecvBufferCpu_int(CTauschInt *tC, size_t haloId, size_t bufferId, int *buf, TauschPackRegion region) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->unpackRecvBufferCpu(haloId, bufferId, buf, region);
}

void tausch_packAndSendCpu_int(CTauschInt *tC, size_t haloId, int *buf, TauschPackRegion region, int mpitag) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->packAndSendCpu(haloId, buf, region, mpitag);
}

void tausch_recvAndUnpackCpu_int(CTauschInt *tC, size_t haloId, int *buf, TauschPackRegion region) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->recvAndUnpackCpu(haloId, buf, region);
}


#ifdef __cplusplus
}
#endif

