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
        t = new Tausch1D<int>(MPI_DOUBLE, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_2D)
        t = new Tausch2D<int>(MPI_DOUBLE, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_3D)
        t = new Tausch3D<int>(MPI_DOUBLE, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);

    return reinterpret_cast<CTauschInt*>(t);

}

void tausch_delete_int(CTauschInt *tC) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    delete t;
}

void tausch_setLocalHaloInfo_int(CTauschInt *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->setLocalHaloInfo(flags, numHaloParts, haloSpecs);
}

void tausch_setRemoteHaloInfo_int(CTauschInt *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->setRemoteHaloInfo(flags, numHaloParts, haloSpecs);
}

void tausch_postReceive_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int msgtag) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->postReceive(flags, haloId, msgtag);
}

void tausch_postAllReceives_int(CTauschInt *tC, TauschDeviceDirection flags, int *msgtag) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->postAllReceives(flags, msgtag);
}

#ifdef TAUSCH_OPENCL
void tausch_packSendBuffer_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, int *buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    if(bufcl == NULL)
        t->packSendBuffer(flags, haloId, bufferId, buf, region);
    else
        t->packSendBuffer(flags, haloId, bufferId, cl::Buffer(*bufcl));
}
#else
void tausch_packSendBuffer_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, int *buf, TauschPackRegion region) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->packSendBuffer(flags, haloId, bufferId, buf, region);
}
#endif

void tausch_send_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int msgtag) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->send(flags, haloId, msgtag);
}

void tausch_recv_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->recv(flags, haloId);
}

#ifdef TAUSCH_OPENCL
void tausch_unpackNextRecvBuffer_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, int *buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    if(bufcl == NULL)
        t->unpackRecvBuffer(flags, haloId, bufferId, buf, region);
    else
        t->unpackRecvBuffer(flags, haloId, bufferId, cl::Buffer(*bufcl));
}
#else
void tausch_unpackNextRecvBuffer_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, int *buf, TauschPackRegion region) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->unpackRecvBuffer(flags, haloId, bufferId, buf, region);
}
#endif

#ifdef TAUSCH_OPENCL
void tausch_packAndSend_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int *buf, cl_mem *bufcl, TauschPackRegion region, int msgtag) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    if(bufcl == NULL)
        t->packAndSend(flags, haloId, buf, region, msgtag);
    else
        t->packAndSend(flags, haloId, cl::Buffer(*bufcl), msgtag);
}
#else
void tausch_packAndSend_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int *buf, TauschPackRegion region, int msgtag) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->packAndSend(flags, haloId, buf, region, msgtag);
}
#endif

#ifdef TAUSCH_OPENCL
void tausch_recvAndUnpack_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int *buf, TauschPackRegion region) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    if(bufcl == NULL)
        t->recvAndUnpack(flags, haloId, buf, region);
    else
        t->recvAndUnpack(flags, haloId, cl::Buffer(*bufcl));
}
#else
void tausch_recvAndUnpack_int(CTauschInt *tC, TauschDeviceDirection flags, size_t haloId, int *buf, TauschPackRegion region) {
    Tausch<int> *t = reinterpret_cast<Tausch<int>*>(tC);
    t->recvAndUnpack(flags, haloId, buf, region);
}
#endif


#ifdef __cplusplus
}
#endif

