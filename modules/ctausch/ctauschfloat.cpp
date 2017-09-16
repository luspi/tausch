#include "../tausch1d.h"
#include "../tausch2d.h"
#include "../tausch3d.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctauschfloat.h"

CTauschFloat *tausch_new_float(size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm, TauschVersion version) {

    if(version != TAUSCH_1D && version != TAUSCH_2D && version != TAUSCH_3D) {
        std::cerr << "[CTauschFloat] ERROR: Invalid version specified: " << version << " - Abort..." << std::endl;
        exit(1);
    }

    Tausch<float> *t;

    if(version == TAUSCH_1D)
        t = new Tausch1D<float>(MPI_FLOAT, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_2D)
        t = new Tausch2D<float>(MPI_FLOAT, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_3D)
        t = new Tausch3D<float>(MPI_FLOAT, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);

    return reinterpret_cast<CTauschFloat*>(t);

}

void tausch_delete_float(CTauschFloat *tC) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    delete t;
}

void tausch_setLocalHaloInfo_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->setLocalHaloInfo(flags, numHaloParts, haloSpecs);
}

void tausch_setRemoteHaloInfo_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->setRemoteHaloInfo(flags, numHaloParts, haloSpecs);
}

void tausch_postReceive_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, int mpitag) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->postReceive(flags, haloId, mpitag);
}

void tausch_postAllReceives_float(CTauschFloat *tC, TauschDeviceDirection flags, int *mpitag) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->postAllReceives(flags, mpitag);
}

void tausch_packSendBuffer_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, float *buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    if(bufcl == NULL)
        t->packSendBuffer(flags, haloId, bufferId, buf, region);
    else
        t->packSendBuffer(flags, haloId, bufferId, cl::Buffer(*bufcl));
}

void tausch_send_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, int mpitag) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->send(flags, haloId, mpitag);
}

void tausch_recv_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->recv(flags, haloId);
}

void tausch_unpackNextRecvBuffer_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, float *buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    if(bufcl == NULL)
        t->unpackRecvBuffer(flags, haloId, bufferId, buf, region);
    else
        t->unpackRecvBuffer(flags, haloId, bufferId, cl::Buffer(*bufcl));
}

void tausch_packAndSend_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, float *buf, cl_mem *bufcl, TauschPackRegion region, int mpitag) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    if(bufcl == NULL)
        t->packAndSend(flags, haloId, buf, region, mpitag);
    else
        t->packAndSend(flags, haloId, cl::Buffer(*bufcl), mpitag);
}

void tausch_recvAndUnpack_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, float *buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    if(bufcl == NULL)
        t->recvAndUnpack(flags, haloId, buf, region);
    else
        t->recvAndUnpack(flags, haloId, cl::Buffer(*bufcl));
}


#ifdef __cplusplus
}
#endif

