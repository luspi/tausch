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
        t = new Tausch1D<float>(MPI_DOUBLE, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_2D)
        t = new Tausch2D<float>(MPI_DOUBLE, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);
    else if(version == TAUSCH_3D)
        t = new Tausch3D<float>(MPI_DOUBLE, numBuffers, (valuesPerPointPerBuffer==NULL ? nullptr : valuesPerPointPerBuffer), comm);

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

void tausch_postReceive_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, int msgtag) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->postReceive(flags, haloId, msgtag);
}

void tausch_postAllReceives_float(CTauschFloat *tC, TauschDeviceDirection flags, int *msgtag) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->postAllReceives(flags, msgtag);
}

#ifdef TAUSCH_OPENCL
void tausch_packSendBuffer_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, float *buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    if(bufcl == NULL)
        t->packSendBuffer(flags, haloId, bufferId, buf, region);
    else
        t->packSendBuffer(flags, haloId, bufferId, cl::Buffer(*bufcl));
}
#else
void tausch_packSendBuffer_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, float *buf, TauschPackRegion region) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->packSendBuffer(flags, haloId, bufferId, buf, region);
}
#endif

void tausch_send_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, int msgtag) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->send(flags, haloId, msgtag);
}

void tausch_recv_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->recv(flags, haloId);
}

#ifdef TAUSCH_OPENCL
void tausch_unpackNextRecvBuffer_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, float *buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    if(bufcl == NULL)
        t->unpackRecvBuffer(flags, haloId, bufferId, buf, region);
    else
        t->unpackRecvBuffer(flags, haloId, bufferId, cl::Buffer(*bufcl));
}
#else
void tausch_unpackNextRecvBuffer_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, float *buf, TauschPackRegion region) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->unpackRecvBuffer(flags, haloId, bufferId, buf, region);
}
#endif

#ifdef TAUSCH_OPENCL
void tausch_packAndSend_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, float *buf, cl_mem *bufcl, TauschPackRegion region, int msgtag) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    if(bufcl == NULL)
        t->packAndSend(flags, haloId, buf, region, msgtag);
    else
        t->packAndSend(flags, haloId, cl::Buffer(*bufcl), msgtag);
}
#else
void tausch_packAndSend_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, float *buf, TauschPackRegion region, int msgtag) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->packAndSend(flags, haloId, buf, region, msgtag);
}
#endif

#ifdef TAUSCH_OPENCL
void tausch_recvAndUnpack_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, float *buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    if(bufcl == NULL)
        t->recvAndUnpack(flags, haloId, buf, region);
    else
        t->recvAndUnpack(flags, haloId, cl::Buffer(*bufcl));
}
#else
void tausch_recvAndUnpack_float(CTauschFloat *tC, TauschDeviceDirection flags, size_t haloId, float *buf, TauschPackRegion region) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->recvAndUnpack(flags, haloId, buf, region);
}
#endif


#ifdef __cplusplus
}
#endif

