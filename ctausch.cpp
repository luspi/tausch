#include "tausch.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctausch.h"

CTausch *tausch_new(MPI_Comm comm, const bool useDuplicateOfCommunicator) {
    Tausch *t = new Tausch(comm, useDuplicateOfCommunicator);
    return reinterpret_cast<CTausch*>(t);
}
void tausch_delete(CTausch *tC) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    delete t;
}

/****************************************/
// addSendHaloInfo

void tausch_addSendHaloInfo(CTausch *tC, int *haloIndices, const size_t numHaloIndices, const size_t typeSize, const int remoteMpiRank) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->addSendHaloInfo(haloIndices, numHaloIndices, typeSize, remoteMpiRank);
}


/****************************************/
// addRecvHaloInfo

void tausch_addRecvHaloInfo(CTausch *tC, int *haloIndices, const size_t numHaloIndices, const size_t typeSize, const int remoteMpiRank) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->addRecvHaloInfo(haloIndices, numHaloIndices, typeSize, remoteMpiRank);
}

void tausch_setSendCommunicationStrategy(CTausch *tC, const size_t haloId, int strategy) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);

    Tausch::Communication str;
    if(strategy == Tausch::Communication::Default) str = Tausch::Communication::Default;
    else if(strategy == Tausch::Communication::TryDirectCopy) str = Tausch::Communication::TryDirectCopy;
    else if(strategy == Tausch::Communication::DerivedMpiDatatype) str = Tausch::Communication::DerivedMpiDatatype;
    else if(strategy == Tausch::Communication::CUDAAwareMPI) str = Tausch::Communication::CUDAAwareMPI;
    else if(strategy == Tausch::Communication::MPIPersistent) str = Tausch::Communication::MPIPersistent;
    else if(strategy == Tausch::Communication::GPUMultiCopy) str = Tausch::Communication::GPUMultiCopy;

    t->setSendCommunicationStrategy(haloId, str);
}

void tausch_setRecvCommunicationStrategy(CTausch *tC, const size_t haloId, int strategy) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);

    Tausch::Communication str;
    if(strategy == Tausch::Communication::Default) str = Tausch::Communication::Default;
    else if(strategy == Tausch::Communication::TryDirectCopy) str = Tausch::Communication::TryDirectCopy;
    else if(strategy == Tausch::Communication::DerivedMpiDatatype) str = Tausch::Communication::DerivedMpiDatatype;
    else if(strategy == Tausch::Communication::CUDAAwareMPI) str = Tausch::Communication::CUDAAwareMPI;
    else if(strategy == Tausch::Communication::MPIPersistent) str = Tausch::Communication::MPIPersistent;
    else if(strategy == Tausch::Communication::GPUMultiCopy) str = Tausch::Communication::GPUMultiCopy;

    t->setRecvCommunicationStrategy(haloId, str);
}
void tausch_setSendHaloBuffer(CTausch *tC, const int haloId, const int bufferId, unsigned char* buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->setSendHaloBuffer(haloId, bufferId, buf);
}
void tausch_setRecvHaloBuffer(CTausch *tC, const int haloId, const int bufferId, unsigned char* buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->setRecvHaloBuffer(haloId, bufferId, buf);
}
void tausch_packSendBuffer(CTausch *tC, const size_t haloId, const size_t bufferId, const unsigned char *buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->packSendBuffer(haloId, bufferId, buf);
}
MPI_Request tausch_send(CTausch *tC, const size_t haloId, const int msgtag, const int remoteMpiRank, const int bufferId, const bool blocking, MPI_Comm communicator) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    return t->send(haloId, msgtag, remoteMpiRank, bufferId, blocking, communicator);
}
MPI_Request tausch_recv(CTausch *tC, size_t haloId, const int msgtag, const int remoteMpiRank, const int bufferId, const bool blocking, MPI_Comm communicator) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    return t->recv(haloId, msgtag, remoteMpiRank, bufferId, blocking, communicator);
}
void tausch_unpackRecvBuffer(CTausch *tC, const size_t haloId, const size_t bufferId, unsigned char *buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->unpackRecvBuffer(haloId, bufferId, buf);
}

#ifdef __cplusplus
}
#endif
