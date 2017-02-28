#include "paramstruct.h"

void launchCPU(void *args) {

    struct param_struct *param = args;

    for(int run = 0; run < param->loops; ++run) {

        // post the receives
        tausch_postCpuReceives(param->tau);

        tausch_performCpuToCpuTausch(param->tau);

        if(!param->cpuonly) {

            tausch_startCpuToGpuTausch(param->tau);
            tausch_completeCpuToGpuTausch(param->tau);

        }

    }

}
