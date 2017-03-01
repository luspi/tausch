#include "paramstruct.h"

void launchGPU(void *args) {

    struct param_struct *param = args;

    if(param->cpuonly) return;

    for(int run = 0; run < param->loops; ++run)
        tausch_performGpuToCpu(param->tausch);

    cl_int err = clEnqueueReadBuffer(tausch_getQueue(param->tausch), param->clGpu, true, 0, (param->gpuDimX+2)*(param->gpuDimY+2)*sizeof(double), param->gpu, 0, NULL, NULL);
    tausch_checkOpenCLError(param->tausch, err, "download final solution");

}
