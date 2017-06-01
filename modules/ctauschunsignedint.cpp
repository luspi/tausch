#include "../tausch.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctauschunsignedint.h"

CTauschUnsignedInt *tausch_new_unsignedint(int *localDim, int *haloWidth, int numBuffers, int valuesPerPoint, MPI_Comm comm, TauschVersion version) {

    if(version != TAUSCH_1D && version != TAUSCH_2D && version != TAUSCH_3D) {
        std::cerr << "[CTauschUnsignedInt] ERROR: Invalid version specified: " << version << " - Abort..." << std::endl;
        exit(1);
    }

    Tausch<unsigned int> *t;

    if(version == TAUSCH_1D)
        t = new Tausch1D<unsigned int>(localDim, haloWidth, MPI_DOUBLE, numBuffers, valuesPerPoint, comm);
    else if(version == TAUSCH_2D)
        t = new Tausch2D<unsigned int>(localDim, haloWidth, MPI_DOUBLE, numBuffers, valuesPerPoint, comm);
    else if(version == TAUSCH_3D)
        t = new Tausch3D<unsigned int>(localDim, haloWidth, MPI_DOUBLE, numBuffers, valuesPerPoint, comm);

    return reinterpret_cast<CTauschUnsignedInt*>(t);

}

void tausch_delete_unsignedint(CTauschUnsignedInt *tC) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    delete t;
}

void tausch_setCpuLocalHaloInfo_unsignedint(CTauschUnsignedInt *tC, int numHaloParts, int **haloSpecs) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->setLocalHaloInfoCpu(numHaloParts, haloSpecs);
}

void tausch_setCpuRemoteHaloInfo_unsignedint(CTauschUnsignedInt *tC, int numHaloParts, int **haloSpecs) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->setRemoteHaloInfoCpu(numHaloParts, haloSpecs);
}

void tausch_postMpiReceives_unsignedint(CTauschUnsignedInt *tC) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->postReceivesCpu();
}

void tausch_packNextSendBuffer_unsignedint(CTauschUnsignedInt *tC, int id, unsigned int *buf) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->packNextSendBufferCpu(id, buf);
}

void tausch_send_unsignedint(CTauschUnsignedInt *tC, int id) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->sendCpu(id);
}

void tausch_recv_unsignedint(CTauschUnsignedInt *tC, int id) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->recvCpu(id);
}

void tausch_unpackNextRecvBuffer_unsignedint(CTauschUnsignedInt *tC, int id, unsigned int *buf) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->unpackNextRecvBufferCpu(id, buf);
}

void tausch_packAndSend_unsignedint(CTauschUnsignedInt *tC, int id, unsigned int *buf) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->packAndSendCpu(id, buf);
}

void tausch_recvAndUnpack_unsignedint(CTauschUnsignedInt *tC, int id, unsigned int *buf) {
    Tausch<unsigned int> *t = reinterpret_cast<Tausch<unsigned int>*>(tC);
    t->recvAndUnpackCpu(id, buf);
}


#ifdef __cplusplus
}
#endif

