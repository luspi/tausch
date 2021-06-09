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
void f_tausch_new_(CTausch **tC, MPI_Fint *comm, const bool *useDuplicateOfCommunicator) {
    Tausch *t = new Tausch(MPI_Comm_f2c(*comm), *useDuplicateOfCommunicator);
    *tC = reinterpret_cast<CTausch*>(t);
}

/****************************************/
// destructor

void tausch_delete(CTausch *tC) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    delete t;
}
void f_tausch_delete_(CTausch **tC) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    delete t;
}

/****************************************/
// addSendHaloInfo()

void tausch_addSendHaloInfo(CTausch *tC, int *haloIndices, const size_t numHaloIndices, const size_t typeSize, const int remoteMpiRank) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->addSendHaloInfo(haloIndices, numHaloIndices, typeSize, remoteMpiRank);
}
void f_tausch_addsendhaloinfo_(CTausch **tC, int *haloIndices, const int *numHaloIndices, const int *typeSize, const int *remoteMpiRank) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    t->addSendHaloInfo(haloIndices, *numHaloIndices, *typeSize, *remoteMpiRank);
}


/****************************************/
// addRecvHaloInfo()

void tausch_addRecvHaloInfo(CTausch *tC, int *haloIndices, const size_t numHaloIndices, const size_t typeSize, const int remoteMpiRank) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->addRecvHaloInfo(haloIndices, numHaloIndices, typeSize, remoteMpiRank);
}
void f_tausch_addrecvhaloinfo_(CTausch **tC, int *haloIndices, const int *numHaloIndices, const int *typeSize, const int *remoteMpiRank) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    t->addRecvHaloInfo(haloIndices, *numHaloIndices, *typeSize, *remoteMpiRank);
}

/****************************************/
// setSendCommunicationStrategy()

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
void f_tausch_setsendcommunicationstrategy_(CTausch **tC, const size_t *haloId, int *strategy) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);

    Tausch::Communication str;
    if(*strategy == Tausch::Communication::Default) str = Tausch::Communication::Default;
    else if(*strategy == Tausch::Communication::TryDirectCopy) str = Tausch::Communication::TryDirectCopy;
    else if(*strategy == Tausch::Communication::DerivedMpiDatatype) str = Tausch::Communication::DerivedMpiDatatype;
    else if(*strategy == Tausch::Communication::CUDAAwareMPI) str = Tausch::Communication::CUDAAwareMPI;
    else if(*strategy == Tausch::Communication::MPIPersistent) str = Tausch::Communication::MPIPersistent;
    else if(*strategy == Tausch::Communication::GPUMultiCopy) str = Tausch::Communication::GPUMultiCopy;

    t->setSendCommunicationStrategy(*haloId, str);
}

/****************************************/
// setRecvCommunicationStrategy()

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
void f_tausch_setrecvcommunicationstrategy_(CTausch **tC, const size_t *haloId, int *strategy) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);

    Tausch::Communication str;
    if(*strategy == Tausch::Communication::Default) str = Tausch::Communication::Default;
    else if(*strategy == Tausch::Communication::TryDirectCopy) str = Tausch::Communication::TryDirectCopy;
    else if(*strategy == Tausch::Communication::DerivedMpiDatatype) str = Tausch::Communication::DerivedMpiDatatype;
    else if(*strategy == Tausch::Communication::CUDAAwareMPI) str = Tausch::Communication::CUDAAwareMPI;
    else if(*strategy == Tausch::Communication::MPIPersistent) str = Tausch::Communication::MPIPersistent;
    else if(*strategy == Tausch::Communication::GPUMultiCopy) str = Tausch::Communication::GPUMultiCopy;

    t->setRecvCommunicationStrategy(*haloId, str);
}

/****************************************/
// setSendHaloBuffer()

void tausch_setSendHaloBuffer(CTausch *tC, const int haloId, const int bufferId, unsigned char* buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->setSendHaloBuffer(haloId, bufferId, buf);
}
void f_tausch_setsendhalobuffer_(CTausch **tC, const int *haloId, const int *bufferId, unsigned char* buf) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    t->setSendHaloBuffer(*haloId, *bufferId, buf);
}

/****************************************/
// setRecvHaloBuffer()

void tausch_setRecvHaloBuffer(CTausch *tC, const int haloId, const int bufferId, unsigned char* buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->setRecvHaloBuffer(haloId, bufferId, buf);
}
void f_tausch_setrecvhalobuffer_(CTausch **tC, const int *haloId, const int *bufferId, unsigned char* buf) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    t->setRecvHaloBuffer(*haloId, *bufferId, buf);
}

/****************************************/
// packSendBuffer()

void tausch_packSendBuffer(CTausch *tC, const size_t haloId, const size_t bufferId, const unsigned char *buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->packSendBuffer(haloId, bufferId, buf);
}
void f_tausch_packsendbuffer_(CTausch **tC, const int *haloId, const int *bufferId, const unsigned char *buf) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    t->packSendBuffer(*haloId, *bufferId, buf);
}

/****************************************/
// send()

MPI_Request tausch_send(CTausch *tC, const size_t haloId, const int msgtag, const int remoteMpiRank, const int bufferId, const bool blocking, MPI_Comm communicator) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    return t->send(haloId, msgtag, remoteMpiRank, bufferId, blocking, communicator);
}
MPI_Request f_tausch_send_(CTausch **tC, const int *haloId, const int *msgtag, const int *remoteMpiRank, const int *bufferId, const bool *blocking, MPI_Fint *communicator) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    return t->send(*haloId, *msgtag, *remoteMpiRank, *bufferId, *blocking, MPI_Comm_f2c(*communicator));
}

/****************************************/
// recv()

MPI_Request tausch_recv(CTausch *tC, size_t haloId, const int msgtag, const int remoteMpiRank, const int bufferId, const bool blocking, MPI_Comm communicator) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    return t->recv(haloId, msgtag, remoteMpiRank, bufferId, blocking, communicator);
}
MPI_Request f_tausch_recv_(CTausch **tC, int *haloId, const int *msgtag, const int *remoteMpiRank, const int *bufferId, const bool *blocking, MPI_Fint *communicator) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    return t->recv(*haloId, *msgtag, *remoteMpiRank, *bufferId, *blocking, MPI_Comm_f2c(*communicator));
}

/****************************************/
// unpackRecvBuffer()

void tausch_unpackRecvBuffer(CTausch *tC, const size_t haloId, const size_t bufferId, unsigned char *buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->unpackRecvBuffer(haloId, bufferId, buf);
}
void f_tausch_unpackrecvbuffer_(CTausch **tC, const int *haloId, const int *bufferId, unsigned char *buf) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    t->unpackRecvBuffer(*haloId, *bufferId, buf);
}

#ifdef TAUSCH_OPENCL

/****************************************/
// setOpenCL()

void tausch_setOpenCL(CTausch *tC, cl_device_id device, cl_context context, cl_command_queue queue) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->setOpenCL(cl::Device(device), cl::Context(context), cl::CommandQueue(queue));
}
void f_tausch_setopencl_(CTausch **tC, cl_device_id *device, cl_context *context, cl_command_queue *queue) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    t->setOpenCL(cl::Device(*device), cl::Context(*context), cl::CommandQueue(*queue));
}

/****************************************/
// enableOpenCL()

void tausch_enableOpenCL(CTausch *tC, size_t deviceNumber) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->enableOpenCL(deviceNumber);
}
void f_tausch_enableopencl_(CTausch **tC, size_t *deviceNumber) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    t->enableOpenCL(*deviceNumber);
}

/****************************************/
// getOclDevice()

cl_device_id tausch_getOclDevice(CTausch *tC) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    return t->getOclDevice()();
}
cl_device_id f_tausch_getocldevice_(CTausch **tC) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    return t->getOclDevice()();
}

/****************************************/
// getOclContext()

cl_context tausch_getOclContext(CTausch *tC) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    return t->getOclContext()();
}
cl_context f_tausch_getoclcontext_(CTausch **tC) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    return t->getOclContext()();
}

/****************************************/
// getOclQueue()

cl_command_queue tausch_getOclQueue(CTausch *tC) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    return t->getOclQueue()();
}
cl_command_queue f_tausch_getoclqueue_(CTausch **tC) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    return t->getOclQueue()();
}

/****************************************/
// packSendBufferOCL()

void tausch_packSendBufferOCL(CTausch *tC, const size_t haloId, const size_t bufferId, cl_mem buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->packSendBufferOCL(haloId, bufferId, *(new cl::Buffer(buf)));
}
void f_tausch_packsendbufferocl_(CTausch **tC, const int *haloId, const int *bufferId, cl_mem *buf) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    t->packSendBufferOCL(*haloId, *bufferId, *(new cl::Buffer(*buf)));
}

/****************************************/
// unpackRecvBufferOCL()

void tausch_unpackRecvBufferOCL(CTausch *tC, const size_t haloId, const size_t bufferId, cl_mem buf) {
    Tausch *t = reinterpret_cast<Tausch*>(tC);
    t->unpackRecvBufferOCL(haloId, bufferId, *(new cl::Buffer(buf)));
}
void f_tausch_unpackrecvbufferocl_(CTausch **tC, const int *haloId, const int *bufferId, cl_mem *buf) {
    Tausch *t = reinterpret_cast<Tausch*>(*tC);
    t->unpackRecvBufferOCL(*haloId, *bufferId, *(new cl::Buffer(*buf)));
}

#endif

#ifdef __cplusplus
}
#endif
