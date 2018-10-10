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

void tausch_setLocalHaloInfo_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->setLocalHaloInfo(flags, numHaloParts, haloSpecs);
}

void tausch_setRemoteHaloInfo_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->setRemoteHaloInfo(flags, numHaloParts, haloSpecs);
}

void tausch_postReceive_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, int msgtag) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->postReceive(flags, haloId, msgtag);
}

void tausch_postAllReceives_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, int *msgtag) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->postAllReceives(flags, msgtag);
}

#ifdef TAUSCH_OPENCL
void tausch_packSendBuffer_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, unsigned int*buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    if(bufcl == NULL)
        t->packSendBuffer(flags, haloId, bufferId, buf, region);
    else
        t->packSendBuffer(flags, haloId, bufferId, cl::Buffer(*bufcl));
}
#else
void tausch_packSendBuffer_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, unsigned int*buf, TauschPackRegion region) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->packSendBuffer(flags, haloId, bufferId, buf, region);
}
#endif

void tausch_send_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, int msgtag) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->send(flags, haloId, msgtag);
}

void tausch_recv_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->recv(flags, haloId);
}

#ifdef TAUSCH_OPENCL
void tausch_unpackNextRecvBuffer_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, unsigned int*buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    if(bufcl == NULL)
        t->unpackRecvBuffer(flags, haloId, bufferId, buf, region);
    else
        t->unpackRecvBuffer(flags, haloId, bufferId, cl::Buffer(*bufcl));
}
#else
void tausch_unpackNextRecvBuffer_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, unsigned int*buf, TauschPackRegion region) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->unpackRecvBuffer(flags, haloId, bufferId, buf, region);
}
#endif

#ifdef TAUSCH_OPENCL
void tausch_packAndSend_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, unsigned int*buf, cl_mem *bufcl, TauschPackRegion region, int msgtag) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    if(bufcl == NULL)
        t->packAndSend(flags, haloId, buf, region, msgtag);
    else
        t->packAndSend(flags, haloId, cl::Buffer(*bufcl), msgtag);
}
#else
void tausch_packAndSend_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, unsigned int*buf, TauschPackRegion region, int msgtag) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->packAndSend(flags, haloId, buf, region, msgtag);
}
#endif

#ifdef TAUSCH_OPENCL
void tausch_recvAndUnpack_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, unsigned int*buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    if(bufcl == NULL)
        t->recvAndUnpack(flags, haloId, buf, region);
    else
        t->recvAndUnpack(flags, haloId, cl::Buffer(*bufcl));
}
#else
void tausch_recvAndUnpack_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, unsigned int*buf, TauschPackRegion region) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->recvAndUnpack(flags, haloId, buf, region);
}
#endif


#ifdef __cplusplus
}
#endif

