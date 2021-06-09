/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 */

#ifndef CTAUSCH_H
#define CTAUSCH_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef TAUSCH_OPENCL
#   ifdef __has_include
#       if __has_include("CL/opencl.h")
#           define CL_TARGET_OPENCL_VERSION 120
#           include <CL/opencl.h>
#       elif __has_include("CL/cl.h")
#           define CL_TARGET_OPENCL_VERSION 120
#           include <CL/cl.h>
#       endif
#   else
#      define CL_TARGET_OPENCL_VERSION 120
#      include <CL/cl.h>
#   endif
#endif

/*!
 *
 * The object that is created by the C API is called CTausch. After its creation it needs to be passed as parameter to any call to the API.
 *
 */
typedef void* CTausch;

/*!
 * The available communication strategies.
 */
enum Communication {
    TauschCommunicationDefault = 1,
    TauschCommunicationTryDirectCopy = 2,
    TauschCommunicationDerivedMpiDatatype = 4,
    TauschCommunicationCUDAAwareMPI = 8,
    TauschCommunicationMPIPersistent = 16,
    TauschCommunicationGPUMultiCopy = 32
};

/*!
 * Create and return a new CTausch object.
 */
CTausch *tausch_new(MPI_Comm comm, const bool useDuplicateOfCommunicator);
void f_tausch_new_(CTausch **tc, MPI_Fint *comm, const bool *useDuplicateOfCommunicator);

/*!
 * Delete the given CTausch object.
 */
void tausch_delete(CTausch *tC);
void f_tausch_delete_(CTausch **tC);

/*!
 * Add sending halo information.
 */
void tausch_addSendHaloInfo(CTausch *tC, int *haloIndices, const size_t numHaloIndices, const size_t typeSize, const int remoteMpiRank);
void f_tausch_addsendhaloinfo_(CTausch **tC, int *haloIndices, const int *numHaloIndices, const int *typeSize, const int *remoteMpiRank);

/*!
 * Add receiving halo information.
 */
void tausch_addRecvHaloInfo(CTausch *tC, int *haloIndices, const size_t numHaloIndices, const size_t typeSize, const int remoteMpiRank);
void f_tausch_addrecvhaloinfo_(CTausch **tC, int *haloIndices, const int *numHaloIndices, const int *typeSize, const int *remoteMpiRank);

/*!
 * Set communication strategy for sending data.
 */
void tausch_setSendCommunicationStrategy(CTausch *tC, const size_t haloId, int strategy);
void f_tausch_setsendcommunicationstrategy_(CTausch **tC, const size_t *haloId, int *strategy);

/*!
 * Set communication strategy for receiving data.
 */
void tausch_setRecvCommunicationStrategy(CTausch *tC, const size_t haloId, int strategy);
void f_tausch_setrecvcommunicationstrategy_(CTausch **tC, const size_t *haloId, int *strategy);

/*!
 * Set halo buffer for sending data to be used by certain communication strategies.
 */
void tausch_setSendHaloBuffer(CTausch *tC, const int haloId, const int bufferId, unsigned char* buf);
void f_tausch_setsendhalobuffer_(CTausch **tC, const int *haloId, const int *bufferId, unsigned char* buf);

/*!
 * Set halo buffer for receiving data to be used by certain communication strategies.
 */
void tausch_setRecvHaloBuffer(CTausch *tC, const int haloId, const int bufferId, unsigned char* buf);
void f_tausch_setrecvhalobuffer_(CTausch **tC, const int *haloId, const int *bufferId, unsigned char* buf);

/*!
 * Pack the halo data from the provided buffer.
 */
void tausch_packSendBuffer(CTausch *tC, const size_t haloId, const size_t bufferId, const unsigned char *buf);
void f_tausch_packsendbuffer_(CTausch **tC, const int *haloId, const int *bufferId, const unsigned char *buf);

/*!
 * Send off the data.
 */
MPI_Request tausch_send(CTausch *tC, const size_t haloId, const int msgtag, const int remoteMpiRank, const int bufferId, const bool blocking, MPI_Comm communicator);
MPI_Request f_tausch_send_(CTausch **tC, const int *haloId, const int *msgtag, const int *remoteMpiRank, const int *bufferId, const bool *blocking, MPI_Fint *communicator);

/*!
 * Receive data.
 */
MPI_Request tausch_recv(CTausch *tC, size_t haloId, const int msgtag, const int remoteMpiRank, const int bufferId, const bool blocking, MPI_Comm communicator);
MPI_Request f_tausch_recv_(CTausch **tC, int *haloId, const int *msgtag, const int *remoteMpiRank, const int *bufferId, const bool *blocking, MPI_Fint *communicator);

/*!
 * Unpack the halo data into the provided buffer.
 */
void tausch_unpackRecvBuffer(CTausch *tC, const size_t haloId, const size_t bufferId, unsigned char *buf);
void f_tausch_unpackrecvbuffer_(CTausch **tC, const int *haloId, const int *bufferId, unsigned char *buf);


#ifdef TAUSCH_OPENCL

/*!
 * Set OpenCL environment.
 */
void tausch_setOpenCL(CTausch *tC, cl_device_id device, cl_context context, cl_command_queue queue);
void f_tausch_setopencl_(CTausch **tC, cl_device_id *device, cl_context *context, cl_command_queue *queue);

/*!
 * Enable OpenCL environment (Tausch will set it up).
 */
void tausch_enableOpenCL(CTausch *tC, size_t deviceNumber);
void f_tausch_enableopencl_(CTausch **tC, size_t *deviceNumber);

/*!
 * Get handle to OpenCL device.
 */
cl_device_id tausch_getOclDevice(CTausch *tC);
cl_device_id f_tausch_getocldevice_(CTausch **tC);

/*!
 * Get handle to OpenCL context.
 */
cl_context tausch_getOclContext(CTausch *tC);
cl_context f_tausch_getoclcontext_(CTausch **tC);

/*!
 * Get handle to OpenCL command queue.
 */
cl_command_queue tausch_getOclQueue(CTausch *tC);
cl_command_queue f_tausch_getoclqueue_(CTausch **tC);

/*!
 * Pack an OpenCL buffer.
 */
void tausch_packSendBufferOCL(CTausch *tC, const size_t haloId, const size_t bufferId, cl_mem buf);
void f_tausch_packsendbufferocl_(CTausch **tC, const int *haloId, const int *bufferId, cl_mem *buf);

/*!
 * Unpack an OpenCL buffer.
 */
void tausch_unpackRecvBufferOCL(CTausch *tC, const size_t haloId, const size_t bufferId, cl_mem buf);
void f_tausch_unpackrecvbufferocl_(CTausch **tC, const int *haloId, const int *bufferId, cl_mem *buf);

#endif

#ifdef __cplusplus
}
#endif


#endif // CTAUSCHDOUBLE_H
