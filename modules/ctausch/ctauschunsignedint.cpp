#include "../tausch1d.h"
#include "../tausch2d.h"
#include "../tausch3d.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctauschunsignedint.h"

CTauschUnsignedInt *tausch_new_unsignedint(size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm, TauschVersion version) {

    if(version != TAUSCH_1D && version != TAUSCH_2D && version != TAUSCH_3D) {
        std::cerr << "[CTauschUnsignedInt] ERROR: Invalid version specified: " << version << " - Abort..." << std::endl;
        exit(1);
    }

    Tausch<unsigned int> *t;

    if(version == TAUSCH_1D)
        t = new Tausch1D<unsigned int>(MPI_DOUBLE, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_2D)
        t = new Tausch2D<unsigned int>(MPI_DOUBLE, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_3D)
        t = new Tausch3D<unsigned int>(MPI_DOUBLE, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);

    return reinterpret_cast<CTauschUnsignedInt*>(t);

}

void tausch_delete_unsignedint(CTauschUnsignedInt *tC) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    delete t;
}

void tausch_setLocalHaloInfoCpu_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->setLocalHaloInfoCpu(numHaloParts, haloSpecs);
}

void tausch_setRemoteHaloInfoCpu_unsignedint(CTauschUnsignedInt *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->setRemoteHaloInfoCpu(numHaloParts, haloSpecs);
}

void tausch_postReceiveCpu_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int mpitag) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->postReceiveCpu(haloId, mpitag);
}

void tausch_postAllReceivesCpu_unsignedint(CTauschUnsignedInt *tC, int *mpitag) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->postAllReceivesCpu(mpitag);
}

void tausch_packSendBufferCpu_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf, TauschPackRegion region) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->packSendBufferCpu(haloId, bufferId, buf, region);
}

void tausch_sendCpu_unsignedint(CTauschUnsignedInt *tC, size_t haloId, int mpitag) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->sendCpu(haloId, mpitag);
}

void tausch_recvCpu_unsignedint(CTauschUnsignedInt *tC, size_t haloId) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->recvCpu(haloId);
}

void tausch_unpackRecvBufferCpu_unsignedint(CTauschUnsignedInt *tC, size_t haloId, size_t bufferId, unsigned int *buf, TauschPackRegion region) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->unpackRecvBufferCpu(haloId, bufferId, buf, region);
}

void tausch_packAndSendCpu_unsignedint(CTauschUnsignedInt *tC, size_t haloId, unsigned int *buf, TauschPackRegion region, int mpitag) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->packAndSendCpu(haloId, buf, region, mpitag);
}

void tausch_recvAndUnpackCpu_unsignedint(CTauschUnsignedInt *tC, size_t haloId, unsigned int *buf, TauschPackRegion region) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->recvAndUnpackCpu(haloId, buf, region);
}


#ifdef __cplusplus
}
#endif

