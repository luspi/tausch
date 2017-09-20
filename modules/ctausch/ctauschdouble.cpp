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

void tausch_setLocalHaloInfo_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->setLocalHaloInfo(flags, numHaloParts, haloSpecs);
}

void tausch_setRemoteHaloInfo_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->setRemoteHaloInfo(flags, numHaloParts, haloSpecs);
}

void tausch_postReceive_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, int msgtag) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->postReceive(flags, haloId, msgtag);
}

void tausch_postAllReceives_double(CTauschDouble *tC, TauschDeviceDirection flags, int *msgtag) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->postAllReceives(flags, msgtag);
}

#ifdef TAUSCH_OPENCL
void tausch_packSendBuffer_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, double *buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    if(bufcl == NULL)
        t->packSendBuffer(flags, haloId, bufferId, buf, region);
    else
        t->packSendBuffer(flags, haloId, bufferId, cl::Buffer(*bufcl));
}
#else
void tausch_packSendBuffer_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, double *buf, TauschPackRegion region) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->packSendBuffer(flags, haloId, bufferId, buf, region);
}
#endif

void tausch_send_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, int msgtag) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->send(flags, haloId, msgtag);
}

void tausch_recv_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->recv(flags, haloId);
}

#ifdef TAUSCH_OPENCL
void tausch_unpackNextRecvBuffer_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, double *buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    if(bufcl == NULL)
        t->unpackRecvBuffer(flags, haloId, bufferId, buf, region);
    else
        t->unpackRecvBuffer(flags, haloId, bufferId, cl::Buffer(*bufcl));
}
#else
void tausch_unpackNextRecvBuffer_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, size_t bufferId, double *buf, TauschPackRegion region) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->unpackRecvBuffer(flags, haloId, bufferId, buf, region);
}
#endif

#ifdef TAUSCH_OPENCL
void tausch_packAndSend_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, double *buf, cl_mem *bufcl, TauschPackRegion region, int msgtag) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    if(bufcl == NULL)
        t->packAndSend(flags, haloId, buf, region, msgtag);
    else
        t->packAndSend(flags, haloId, cl::Buffer(*bufcl), msgtag);
}
#else
void tausch_packAndSend_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, double *buf, TauschPackRegion region, int msgtag) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->packAndSend(flags, haloId, buf, region, msgtag);
}
#endif

#ifdef TAUSCH_OPENCL
void tausch_recvAndUnpack_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, double *buf, cl_mem *bufcl, TauschPackRegion region) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    if(bufcl == NULL)
        t->recvAndUnpack(flags, haloId, buf, region);
    else
        t->recvAndUnpack(flags, haloId, cl::Buffer(*bufcl));
}
#else
void tausch_recvAndUnpack_double(CTauschDouble *tC, TauschDeviceDirection flags, size_t haloId, double *buf, TauschPackRegion region) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->recvAndUnpack(flags, haloId, buf, region);
}
#endif


#ifdef __cplusplus
}
#endif

