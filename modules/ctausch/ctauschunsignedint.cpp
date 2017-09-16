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
        t = new Tausch1D<unsigned int>(MPI_UNSIGNED, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_2D)
        t = new Tausch2D<unsigned int>(MPI_UNSIGNED, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_3D)
        t = new Tausch3D<unsigned int>(MPI_UNSIGNED, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);

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

void tausch_postReceive_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, int mpitag) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->postReceive(flags, haloId, mpitag);
}

void tausch_postAllReceives_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, int *mpitag) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->postAllReceives(flags, mpitag);
}

void tausch_packSendBuffer_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, unsigned int *buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    if(bufcl == NULL)
        t->packSendBuffer(flags, haloId, bufferId, buf, region);
    else
        t->packSendBuffer(flags, haloId, bufferId, cl::Buffer(*bufcl));
}

void tausch_send_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, int mpitag) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->send(flags, haloId, mpitag);
}

void tausch_recv_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->recv(flags, haloId);
}

void tausch_unpackNextRecvBuffer_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, unsigned int *buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    if(bufcl == NULL)
        t->unpackRecvBuffer(flags, haloId, bufferId, buf, region);
    else
        t->unpackRecvBuffer(flags, haloId, bufferId, cl::Buffer(*bufcl));
}

void tausch_packAndSend_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, unsigned int *buf, cl_mem *bufcl, TauschPackRegion region, int mpitag) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    if(bufcl == NULL)
        t->packAndSend(flags, haloId, buf, region, mpitag);
    else
        t->packAndSend(flags, haloId, cl::Buffer(*bufcl), mpitag);
}

void tausch_recvAndUnpack_unsignedint(CTauschUnsignedInt *tC, TauschDeviceDirection flags, size_t haloId, unsigned int *buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    if(bufcl == NULL)
        t->recvAndUnpack(flags, haloId, buf, region);
    else
        t->recvAndUnpack(flags, haloId, cl::Buffer(*bufcl));
}


#ifdef __cplusplus
}
#endif

