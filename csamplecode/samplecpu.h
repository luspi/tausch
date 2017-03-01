#include "paramstruct.h"

void launchCPU(void *args) {

    struct param_struct *param = args;

    int mpiRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    for(int run = 0; run < param->loops; ++run) {

        if(mpiRank == 0 && (run+1)%10 == 0)
            printf("Loop %i/%i\n", run+1, param->loops);

        // post the receives
        tausch_postCpuReceives(param->tausch);

        tausch_performCpuToCpu(param->tausch);

        if(!param->cpuonly)
            tausch_performCpuToGpu(param->tausch);

    }

}
