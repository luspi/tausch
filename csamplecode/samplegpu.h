#include "paramstruct.h"

void launchGPU(void *args) {

    struct param_struct *param = args;

    if(param->cpuonly) return;

    for(int run = 0; run < param->loops; ++run) {

        tausch_startGpuToCpuTausch(param->tau);

        tausch_completeGpuToCpuTausch(param->tau);

    }

    cl_int err = clEnqueueReadBuffer(tausch_getQueue(param->tau), param->clGpu, true, 0, (param->gpuDimX+2)*(param->gpuDimY+2)*sizeof(double), param->gpu, 0, NULL, NULL);
    tausch_checkOpenCLError(param->tau, err, "download final solution");

}
