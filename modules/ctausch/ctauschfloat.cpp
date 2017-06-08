#include "../tausch1d.h"
#include "../tausch2d.h"
#include "../tausch3d.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctauschfloat.h"

CTauschFloat *tausch_new_float(size_t *localDim, size_t numBuffers, size_t *valuesPerPointPerBuffer, MPI_Comm comm, TauschVersion version) {

    if(version != TAUSCH_1D && version != TAUSCH_2D && version != TAUSCH_3D) {
        std::cerr << "[CTauschFloat] ERROR: Invalid version specified: " << version << " - Abort..." << std::endl;
        exit(1);
    }

    Tausch<float> *t;

    if(version == TAUSCH_1D)
        t = new Tausch1D<float>(localDim, MPI_FLOAT, numBuffers, valuesPerPointPerBuffer, comm);
    else if(version == TAUSCH_2D)
        t = new Tausch2D<float>(localDim, MPI_FLOAT, numBuffers, valuesPerPointPerBuffer, comm);
    else if(version == TAUSCH_3D)
        t = new Tausch3D<float>(localDim, MPI_FLOAT, numBuffers, valuesPerPointPerBuffer, comm);

    return reinterpret_cast<CTauschFloat*>(t);

}

void tausch_delete_float(CTauschFloat *tC) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    delete t;
}

void tausch_setCpuLocalHaloInfo_float(CTauschFloat *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->setLocalHaloInfoCpu(numHaloParts, haloSpecs);
}

void tausch_setCpuRemoteHaloInfo_float(CTauschFloat *tC, size_t numHaloParts, TauschHaloSpec *haloSpecs) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->setRemoteHaloInfoCpu(numHaloParts, haloSpecs);
}

void tausch_postReceiveCpu_float(CTauschFloat *tC, size_t id, int mpitag) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->postReceiveCpu(id, mpitag);
}

void tausch_postAllReceivesCpu_float(CTauschFloat *tC, int *mpitag) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->postAllReceivesCpu(mpitag);
}

void tausch_packNextSendBuffer_float(CTauschFloat *tC, size_t id, float *buf) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->packNextSendBufferCpu(id, buf);
}

void tausch_send_float(CTauschFloat *tC, size_t id, int mpitag) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->sendCpu(id, mpitag);
}

void tausch_recv_float(CTauschFloat *tC, size_t id) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->recvCpu(id);
}

void tausch_unpackNextRecvBuffer_float(CTauschFloat *tC, size_t id, float *buf) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->unpackNextRecvBufferCpu(id, buf);
}

void tausch_packAndSend_float(CTauschFloat *tC, size_t id, int mpitag, float *buf) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->packAndSendCpu(id, buf, mpitag);
}

void tausch_recvAndUnpack_float(CTauschFloat *tC, size_t id, float *buf) {
    Tausch<float> *t = reinterpret_cast<Tausch<float>*>(tC);
    t->recvAndUnpackCpu(id, buf);
}


#ifdef __cplusplus
}
#endif

