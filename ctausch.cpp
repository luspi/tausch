#include "tausch.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ctausch.h"

/****************************************/
// constructor

CTausch *tausch_new(MPI_Comm comm, const bool useDuplicateOfCommunicator) {
    Tausch *t = new Tausch(comm, useDuplicateOfCommunicator);
    return reinterpret_cast<CTausch*>(t);
}
CTausch *tausch_new_f(MPI_Fint comm, const bool useDuplicateOfCommunicator) {
    return tausch_new(MPI_Comm_f2c(comm), useDuplicateOfCommunicator);
}

/****************************************/
// destructor

void tausch_delete(CTausch *tC) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    delete t;
}

/****************************************/
// addSendHaloInfo()

void tausch_addSendHaloInfo(CTausch *tC, int *haloIndices, const size_t numHaloIndices, const size_t typeSize, const int remoteMpiRank) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->addSendHaloInfo(haloIndices, numHaloIndices, typeSize, remoteMpiRank);
}


/****************************************/
// addRecvHaloInfo()

void tausch_addRecvHaloInfo(CTausch *tC, int *haloIndices, const size_t numHaloIndices, const size_t typeSize, const int remoteMpiRank) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->addRecvHaloInfo(haloIndices, numHaloIndices, typeSize, remoteMpiRank);
}

/****************************************/
// setSendCommunicationStrategy()

void tausch_setSendCommunicationStrategy(CTausch *tC, const size_t haloId, int strategy) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);

    Tausch::Communication str = Tausch::Communication::Default;
    if(strategy == Tausch::Communication::TryDirectCopy) str = Tausch::Communication::TryDirectCopy;
    else if(strategy == Tausch::Communication::DerivedMpiDatatype) str = Tausch::Communication::DerivedMpiDatatype;
    else if(strategy == Tausch::Communication::CUDAAwareMPI) str = Tausch::Communication::CUDAAwareMPI;
    else if(strategy == Tausch::Communication::MPIPersistent) str = Tausch::Communication::MPIPersistent;
    else if(strategy == Tausch::Communication::GPUMultiCopy) str = Tausch::Communication::GPUMultiCopy;

    t->setSendCommunicationStrategy(haloId, str);
}

/****************************************/
// setRecvCommunicationStrategy()

void tausch_setRecvCommunicationStrategy(CTausch *tC, const size_t haloId, int strategy) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);

    Tausch::Communication str = Tausch::Communication::Default;
    if(strategy == Tausch::Communication::TryDirectCopy) str = Tausch::Communication::TryDirectCopy;
    else if(strategy == Tausch::Communication::DerivedMpiDatatype) str = Tausch::Communication::DerivedMpiDatatype;
    else if(strategy == Tausch::Communication::CUDAAwareMPI) str = Tausch::Communication::CUDAAwareMPI;
    else if(strategy == Tausch::Communication::MPIPersistent) str = Tausch::Communication::MPIPersistent;
    else if(strategy == Tausch::Communication::GPUMultiCopy) str = Tausch::Communication::GPUMultiCopy;

    t->setRecvCommunicationStrategy(haloId, str);
}

/****************************************/
// setSendHaloBuffer()

void tausch_setSendHaloBuffer(CTausch *tC, const int haloId, const int bufferId, unsigned char* buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->setSendHaloBuffer(haloId, bufferId, buf);
}
void tausch_setSendHaloBuffer_double(CTausch *tC, const int haloId, const int bufferId, double* buf) {
    tausch_setSendHaloBuffer(tC, haloId, bufferId, (unsigned char*)buf);
}
void tausch_setSendHaloBuffer_float(CTausch *tC, const int haloId, const int bufferId, float* buf) {
    tausch_setSendHaloBuffer(tC, haloId, bufferId, (unsigned char*)buf);
}
void tausch_setSendHaloBuffer_int(CTausch *tC, const int haloId, const int bufferId, int* buf) {
    tausch_setSendHaloBuffer(tC, haloId, bufferId, (unsigned char*)buf);
}

/****************************************/
// setRecvHaloBuffer()

void tausch_setRecvHaloBuffer(CTausch *tC, const int haloId, const int bufferId, unsigned char* buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->setRecvHaloBuffer(haloId, bufferId, buf);
}
void tausch_setRecvHaloBuffer_double(CTausch *tC, const int haloId, const int bufferId, double* buf) {
    tausch_setRecvHaloBuffer(tC, haloId, bufferId, (unsigned char*)buf);
}
void tausch_setRecvHaloBuffer_float(CTausch *tC, const int haloId, const int bufferId, float* buf) {
    tausch_setRecvHaloBuffer(tC, haloId, bufferId, (unsigned char*)buf);
}
void tausch_setRecvHaloBuffer_int(CTausch *tC, const int haloId, const int bufferId, int* buf) {
    tausch_setRecvHaloBuffer(tC, haloId, bufferId, (unsigned char*)buf);
}

/****************************************/
// packSendBuffer()

void tausch_packSendBuffer(CTausch *tC, const size_t haloId, const size_t bufferId, const unsigned char *buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->packSendBuffer(haloId, bufferId, buf);
}
void tausch_packSendBuffer_double(CTausch *tC, const size_t haloId, const size_t bufferId, const double *buf) {
    tausch_packSendBuffer(tC, haloId, bufferId, (unsigned char*)buf);
}
void tausch_packSendBuffer_float(CTausch *tC, const size_t haloId, const size_t bufferId, const float *buf) {
    tausch_packSendBuffer(tC, haloId, bufferId, (unsigned char*)buf);
}
void tausch_packSendBuffer_int(CTausch *tC, const size_t haloId, const size_t bufferId, const int *buf) {
    tausch_packSendBuffer(tC, haloId, bufferId, (unsigned char*)buf);
}

/****************************************/
// send()

MPI_Request tausch_send(CTausch *tC, const size_t haloId, const int msgtag, const int remoteMpiRank, const int bufferId, const bool blocking, MPI_Comm communicator) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    return t->send(haloId, msgtag, remoteMpiRank, bufferId, blocking, communicator);
}
MPI_Request tausch_send_f(CTausch *tC, const size_t haloId, const int msgtag, const int remoteMpiRank, const int bufferId, const bool blocking, MPI_Fint communicator) {
    return tausch_send(tC, haloId, msgtag, remoteMpiRank, bufferId, blocking, MPI_Comm_f2c(communicator));
}

/****************************************/
// recv()

MPI_Request tausch_recv(CTausch *tC, size_t haloId, const int msgtag, const int remoteMpiRank, const int bufferId, const bool blocking, MPI_Comm communicator) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    return t->recv(haloId, msgtag, remoteMpiRank, bufferId, blocking, communicator);
}
MPI_Request tausch_recv_f(CTausch *tC, size_t haloId, const int msgtag, const int remoteMpiRank, const int bufferId, const bool blocking, MPI_Fint communicator) {
    return tausch_recv(tC, haloId, msgtag, remoteMpiRank, bufferId, blocking, MPI_Comm_f2c(communicator));
}

/****************************************/
// unpackRecvBuffer()

void tausch_unpackRecvBuffer(CTausch *tC, const size_t haloId, const size_t bufferId, unsigned char *buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->unpackRecvBuffer(haloId, bufferId, buf);
}
void tausch_unpackRecvBuffer_double(CTausch *tC, const size_t haloId, const size_t bufferId, double *buf) {
    tausch_unpackRecvBuffer(tC, haloId, bufferId, (unsigned char*)buf);
}
void tausch_unpackRecvBuffer_float(CTausch *tC, const size_t haloId, const size_t bufferId, float *buf) {
    tausch_unpackRecvBuffer(tC, haloId, bufferId, (unsigned char*)buf);
}
void tausch_unpackRecvBuffer_int(CTausch *tC, const size_t haloId, const size_t bufferId, int *buf) {
    tausch_unpackRecvBuffer(tC, haloId, bufferId, (unsigned char*)buf);
}

#ifdef TAUSCH_OPENCL

/****************************************/
// setOpenCL()

void tausch_setOpenCL(CTausch *tC, cl_device_id device, cl_context context, cl_command_queue queue) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->setOpenCL(cl::Device(device), cl::Context(context), cl::CommandQueue(queue));
}

/****************************************/
// enableOpenCL()

void tausch_enableOpenCL(CTausch *tC, size_t deviceNumber) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->enableOpenCL(deviceNumber);
}

/****************************************/
// getOclDevice()

cl_device_id tausch_getOclDevice(CTausch *tC) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    return t->getOclDevice()();
}

/****************************************/
// getOclContext()

cl_context tausch_getOclContext(CTausch *tC) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    return t->getOclContext()();
}

/****************************************/
// getOclQueue()

cl_command_queue tausch_getOclQueue(CTausch *tC) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    return t->getOclQueue()();
}

/****************************************/
// packSendBufferOCL()

void tausch_packSendBufferOCL(CTausch *tC, const size_t haloId, const size_t bufferId, cl_mem buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->packSendBufferOCL(haloId, bufferId, *(new cl::Buffer(buf)));
}

/****************************************/
// unpackRecvBufferOCL()

void tausch_unpackRecvBufferOCL(CTausch *tC, const size_t haloId, const size_t bufferId, cl_mem buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->unpackRecvBufferOCL(haloId, bufferId, *(new cl::Buffer(buf)));
}

#endif

#ifdef __cplusplus
}
#endif
