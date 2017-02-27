#include <stdbool.h>
#include <CL/cl.h>

typedef void* CTausch;

#ifdef __cplusplus
extern "C" {
#endif

CTausch *tausch_new(int localDimX, int localDimY, int mpiNumX, int mpiNumY, bool withOpenCL, bool setupOpenCL, int clLocalWorkgroupSize, bool giveOpenCLDeviceName);
void tausch_delete(CTausch *tC);

void tausch_setOpenCLInfo(CTausch *tC, const cl_device_id &clDefaultDevice, const cl_context &clContext, const cl_command_queue &clQueue);

void tausch_postCpuReceives(CTausch *tC);

void tausch_performCpuToCpuTausch(CTausch *tC);

void tausch_performCpuToCpuAndCpuToGpuTausch(CTausch *tC);

void tausch_performGpuToCpuTausch(CTausch *tC);

void tausch_startCpuToGpuTausch(CTausch *tC);
void tausch_startGpuToCpuTausch(CTausch *tC);

void tausch_completeCpuToGpuTausch(CTausch *tC);
void tausch_completeGpuToCpuTausch(CTausch *tC);

void tausch_startCpuTauschLeft(CTausch *tC);
void tausch_startCpuTauschRight(CTausch *tC);
void tausch_startCpuTauschTop(CTausch *tC);
void tausch_startCpuTauschBottom(CTausch *tC);

void tausch_completeCpuTauschLeft(CTausch *tC);
void tausch_completeCpuTauschRight(CTausch *tC);
void tausch_completeCpuTauschTop(CTausch *tC);
void tausch_completeCpuTauschBottom(CTausch *tC);

void tausch_syncCpuAndGpu(CTausch *tC, bool iAmTheCPU);


void tausch_setHaloWidth(CTausch *tC, int haloWidth);
void tausch_setCPUData(CTausch *tC, double *dat);
void tausch_setGPUData(CTausch *tC, cl_mem &dat, int gpuWidth, int gpuHeight);
bool tausch_isGpuEnabled(CTausch *tC);

#ifdef __cplusplus
}
#endif
