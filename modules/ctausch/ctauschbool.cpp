#include "../../tausch.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctauschbool.h"

CTauschBool *tausch_new_bool(int *localDim, int *haloWidth, int numBuffers, int valuesPerPoint, MPI_Comm comm, TauschVersion version) {

    if(version != TAUSCH_1D && version != TAUSCH_2D && version != TAUSCH_3D) {
        std::cerr << "[CTauschBool] ERROR: Invalid version specified: " << version << " - Abort..." << std::endl;
        exit(1);
    }

    Tausch<bool> *t;

    if(version == TAUSCH_1D)
        t = new Tausch1D<bool>(localDim, haloWidth, MPI_DOUBLE, numBuffers, valuesPerPoint, comm);
    else if(version == TAUSCH_2D)
        t = new Tausch2D<bool>(localDim, haloWidth, MPI_DOUBLE, numBuffers, valuesPerPoint, comm);
    else if(version == TAUSCH_3D)
        t = new Tausch3D<bool>(localDim, haloWidth, MPI_DOUBLE, numBuffers, valuesPerPoint, comm);

    return reinterpret_cast<CTauschBool*>(t);

}

void tausch_delete_bool(CTauschBool *tC) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    delete t;
}

void tausch_setCpuLocalHaloInfo_bool(CTauschBool *tC, int numHaloParts, int **haloSpecs) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->setLocalHaloInfoCpu(numHaloParts, haloSpecs);
}

void tausch_setCpuRemoteHaloInfo_bool(CTauschBool *tC, int numHaloParts, int **haloSpecs) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->setRemoteHaloInfoCpu(numHaloParts, haloSpecs);
}

void tausch_postReceiveCpu_bool(CTauschBool *tC, int id, int mpitag) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->postReceiveCpu(id, mpitag);
}

void tausch_postAllReceivesCpu_bool(CTauschUnsignedInt *tC, int *mpitag) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->postAllReceivesCpu(mpitag);
}

void tausch_packNextSendBuffer_bool(CTauschBool *tC, int id, bool *buf) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->packNextSendBufferCpu(id, buf);
}

void tausch_send_bool(CTauschBool *tC, int id, int mpitag) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->sendCpu(id, mpitag);
}

void tausch_recv_bool(CTauschBool *tC, int id) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->recvCpu(id);
}

void tausch_unpackNextRecvBuffer_bool(CTauschBool *tC, int id, bool *buf) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->unpackNextRecvBufferCpu(id, buf);
}

void tausch_packAndSend_bool(CTauschBool *tC, int id, int mpitag, bool *buf) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->packAndSendCpu(id, mpitag, buf);
}

void tausch_recvAndUnpack_bool(CTauschBool *tC, int id, bool *buf) {
    Tausch<bool> *t = reinterpret_cast<Tausch<bool>*>(tC);
    t->recvAndUnpackCpu(id, buf);
}


#ifdef __cplusplus
}
#endif

