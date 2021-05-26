/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 *
 * \brief
 *  C wrapper to C++ API.
 */

#ifndef CTAUSCH_H
#define CTAUSCH_H

#ifdef __cplusplus
extern "C" {
#endif


#ifdef TAUSCH_OPENCL
#include <CL/cl.h>
#endif

/*!
 *
 * The object that is created by the C API is called CTausch. After its creation it needs to be passed as parameter to any call to the API.
 *
 */
typedef void* CTausch;



enum Communication {
    TauschCommunicationDefault = 1,
    TauschCommunicationTryDirectCopy = 2,
    TauschCommunicationDerivedMpiDatatype = 4,
    TauschCommunicationCUDAAwareMPI = 8,
    TauschCommunicationMPIPersistent = 16,
    TauschCommunicationGPUMultiCopy = 32
};

/*!
 *
 * Create and return a new CTausch object.
 *
 * \param numBuffers
 *  The number of buffers that will be used. If more than one, they are all combined into one message. All buffers will have to use the same
 *  discretisation! Typical value: 1.
 * \param valuesPerPointPerBuffer
 *  How many values are stored consecutively per point in the same buffer. Each buffer can have different number of values stored per point. This
 *  is expected to be an array of the same size as the number of buffers. If set to NULL, all buffers are assumed to store 1 value per point.
 * \param comm
 *  The MPI Communictor to be used. %CTauschDouble will duplicate the communicator, thus it is safe to have multiple instances of %CTauschDouble working
 *  with the same communicator. By default, MPI_COMM_WORLD will be used.
 * \param version
 *  Which version of CTauschDouble to create. This depends on the dimensionality of the problem and can be any one of the enum TAUSCH_VERSION: TAUSCH_1D,
 *  TAUSCH_2D, or TAUSCH_3D.
 *
 * \return
 *  Return the CTauschDouble object created with the specified configuration.
 *
 */
CTausch *tausch_new(MPI_Comm comm, const bool useDuplicateOfCommunicator);
void tausch_delete(CTausch *tC);

/****************************************/
// addSendHaloInfo

void tausch_addSendHaloInfo(CTausch *tC, int *haloIndices, const size_t numHaloIndices, const size_t typeSize, const int remoteMpiRank);

/****************************************/
// addRecvHaloInfo

void tausch_addRecvHaloInfo(CTausch *tC, int *haloIndices, const size_t numHaloIndices, const size_t typeSize, const int remoteMpiRank);

void tausch_setSendCommunicationStrategy(CTausch *tC, const size_t haloId, int strategy);

void tausch_setRecvCommunicationStrategy(CTausch *tC, const size_t haloId, int strategy);
void tausch_setSendHaloBuffer(CTausch *tC, const int haloId, const int bufferId, unsigned char* buf);
void tausch_setRecvHaloBuffer(CTausch *tC, const int haloId, const int bufferId, unsigned char* buf);
void tausch_packSendBuffer(CTausch *tC, const size_t haloId, const size_t bufferId, const unsigned char *buf);
MPI_Request tausch_send(CTausch *tC, const size_t haloId, const int msgtag, const int remoteMpiRank, const int bufferId, const bool blocking, MPI_Comm communicator);
MPI_Request tausch_recv(CTausch *tC, size_t haloId, const int msgtag, const int remoteMpiRank, const int bufferId, const bool blocking, MPI_Comm communicator);
void tausch_unpackRecvBuffer(CTausch *tC, const size_t haloId, const size_t bufferId, unsigned char *buf);

#ifdef __cplusplus
}
#endif


#endif // CTAUSCHDOUBLE_H
