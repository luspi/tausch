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
 * Create and return a new CTausch object.
 */
CTausch *tausch_new(MPI_Comm comm, const bool useDuplicateOfCommunicator);

/*!
 * Delete the given CTausch object.
 */
void tausch_delete(CTausch *tC);

/*!
 * Add sending halo information.
 */
void tausch_addSendHaloInfo(CTausch *tC, int *haloIndices, const size_t numHaloIndices, const size_t typeSize, const int remoteMpiRank);

/*!
 * Add receiving halo information.
 */
void tausch_addRecvHaloInfo(CTausch *tC, int *haloIndices, const size_t numHaloIndices, const size_t typeSize, const int remoteMpiRank);

/*!
 * Set communication strategy for sending data.
 */
void tausch_setSendCommunicationStrategy(CTausch *tC, const size_t haloId, int strategy);

/*!
 * Set communication strategy for receiving data.
 */
void tausch_setRecvCommunicationStrategy(CTausch *tC, const size_t haloId, int strategy);

/*!
 * Set halo buffer for sending data to be used by certain communication strategies.
 */
void tausch_setSendHaloBuffer(CTausch *tC, const int haloId, const int bufferId, unsigned char* buf);

/*!
 * Set halo buffer for receiving data to be used by certain communication strategies.
 */
void tausch_setRecvHaloBuffer(CTausch *tC, const int haloId, const int bufferId, unsigned char* buf);

/*!
 * Pack the halo data from the provided buffer.
 */
void tausch_packSendBuffer(CTausch *tC, const size_t haloId, const size_t bufferId, const unsigned char *buf);

/*!
 * Send off the data.
 */
MPI_Request tausch_send(CTausch *tC, const size_t haloId, const int msgtag, const int remoteMpiRank, const int bufferId, const bool blocking, MPI_Comm communicator);

/*!
 * Receive data.
 */
MPI_Request tausch_recv(CTausch *tC, size_t haloId, const int msgtag, const int remoteMpiRank, const int bufferId, const bool blocking, MPI_Comm communicator);

/*!
 * Unpack the halo data into the provided buffer.
 */
void tausch_unpackRecvBuffer(CTausch *tC, const size_t haloId, const size_t bufferId, unsigned char *buf);

#ifdef __cplusplus
}
#endif


#endif // CTAUSCHDOUBLE_H
