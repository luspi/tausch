#include "../tausch1d.h"
#include "../tausch2d.h"
#include "../tausch3d.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctauschdouble.h"

CTauschDouble *tausch_new_double(size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm, TauschVersion version) {

    if(version != TAUSCH_1D && version != TAUSCH_2D && version != TAUSCH_3D) {
        std::cerr << "[CTauschDouble] ERROR: Invalid version specified: " << version << " - Abort..." << std::endl;
        exit(1);
    }

    Tausch<double> *t;

    if(version == TAUSCH_1D)
        t = new Tausch1D<double>(MPI_DOUBLE, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_2D)
        t = new Tausch2D<double>(MPI_DOUBLE, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_3D)
        t = new Tausch3D<double>(MPI_DOUBLE, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);

    return reinterpret_cast<CTauschDouble*>(t);

}

void tausch_delete_double(CTauschDouble *tC) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    delete t;
}

void tausch_setCpuLocalHaloInfo_double(CTauschDouble *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->setLocalHaloInfoCpu(numHaloParts, haloSpecs);
}

void tausch_setCpuRemoteHaloInfo_double(CTauschDouble *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->setRemoteHaloInfoCpu(numHaloParts, haloSpecs);
}

void tausch_postReceiveCpu_double(CTauschDouble *tC, size_t haloId, int mpitag) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->postReceiveCpu(haloId, mpitag);
}

void tausch_postAllReceivesCpu_double(CTauschDouble *tC, int *mpitag) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->postAllReceivesCpu(mpitag);
}

void tausch_packNextSendBuffer_double(CTauschDouble *tC, size_t haloId, size_t bufferId, double *buf, TauschPackRegion region) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->packSendBufferCpu(haloId, bufferId, buf, region);
}

void tausch_send_double(CTauschDouble *tC, size_t haloId, int mpitag) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->sendCpu(haloId, mpitag);
}

void tausch_recv_double(CTauschDouble *tC, size_t haloId) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->recvCpu(haloId);
}

void tausch_unpackNextRecvBuffer_double(CTauschDouble *tC, size_t haloId, size_t bufferId, double *buf, TauschPackRegion region) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->unpackRecvBufferCpu(haloId, bufferId, buf, region);
}

void tausch_packAndSend_double(CTauschDouble *tC, size_t haloId, double *buf, TauschPackRegion region, int mpitag) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->packAndSendCpu(haloId, buf, region, mpitag);
}

void tausch_recvAndUnpack_double(CTauschDouble *tC, size_t haloId, double *buf, TauschPackRegion region) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->recvAndUnpackCpu(haloId, buf, region);
}


#ifdef __cplusplus
}
#endif

