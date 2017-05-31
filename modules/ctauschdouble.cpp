#include "../tausch.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctauschdouble.h"

CTauschDouble *tausch_new_double(int *localDim, int *haloWidth, int numBuffers, int valuesPerPoint, MPI_Comm comm, TauschVersion version) {

    if(version != TAUSCH_1D && version != TAUSCH_2D && version != TAUSCH_3D) {
        std::cerr << "[CTauschDouble] ERROR: Invalid version specified: " << version << " - Abort..." << std::endl;
        exit(1);
    }

    Tausch<double> *t;

    if(version == TAUSCH_1D)
        t = new Tausch1D<double>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
    else if(version == TAUSCH_2D)
        t = new Tausch2D<double>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);
    else if(version == TAUSCH_3D)
        t = new Tausch3D<double>(localDim, haloWidth, numBuffers, valuesPerPoint, comm);

    return reinterpret_cast<CTauschDouble*>(t);

}

void tausch_delete_double(CTauschDouble *tC) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    delete t;
}

void tausch_setCpuLocalHaloInfo_double(CTauschDouble *tC, int numHaloParts, int **haloSpecs) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->setCpuLocalHaloInfo(numHaloParts, haloSpecs);
}

void tausch_setCpuRemoteHaloInfo_double(CTauschDouble *tC, int numHaloParts, int **haloSpecs) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->setCpuRemoteHaloInfo(numHaloParts, haloSpecs);
}

void tausch_postMpiReceives_double(CTauschDouble *tC) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->postMpiReceives();
}

void tausch_packNextSendBuffer_double(CTauschDouble *tC, int id, double *buf) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->packNextSendBuffer(id, buf);
}

void tausch_send_double(CTauschDouble *tC, int id) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->send(id);
}

void tausch_recv_double(CTauschDouble *tC, int id) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->recv(id);
}

void tausch_unpackNextRecvBuffer_double(CTauschDouble *tC, int id, double *buf) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->unpackNextRecvBuffer(id, buf);
}

void tausch_packAndSend_double(CTauschDouble *tC, int id, double *buf) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->packAndSend(id, buf);
}

void tausch_recvAndUnpack_double(CTauschDouble *tC, int id, double *buf) {
    Tausch<double> *t = reinterpret_cast<Tausch<double>*>(tC);
    t->recvAndUnpack(id, buf);
}


#ifdef __cplusplus
}
#endif

