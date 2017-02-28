#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <stdbool.h>
#include <ctausch.h>
#include <pthread.h>
#include <CL/cl.h>
#include <time.h>

#include "samplecpu.h"
#include "samplegpu.h"
#include "paramstruct.h"

void printCPU(struct param_struct *param);
void printGPU(struct param_struct *param);

int main(int argc, char** argv) {

    int provided;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_SERIALIZED,&provided);

    // If this feature is not available -> abort
    if(provided != MPI_THREAD_SERIALIZED){
        printf("ERROR: The MPI library does not have full thread support at level MPI_THREAD_SERIALIZED... Abort!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    struct param_struct param;

    param.localDimX = 25, param.localDimY = 15;
    param.portionGPU = 0.5;
    param.mpiNumX = sqrt(mpiSize);
    param.mpiNumY = param.mpiNumX;
    param.loops = 1;
    param.cpuonly = false;
    param.workgroupsize = 64;
    param.giveOpenClDeviceName = false;
    param.printMpiRank = -1;

    if(argc > 1) {
        for(int i = 1; i < argc; ++i) {

            if(strcmp(argv[i], "-x") == 0 && i < argc-1)
                param.localDimX = atoi(argv[++i]);
            else if(strcmp(argv[i], "-y") == 0 && i < argc-1)
                param.localDimY = atoi(argv[++i]);
            else if(strcmp(argv[i], "-gpu") == 0 && i < argc-1)
                param.portionGPU = atof(argv[++i]);
            else if(strcmp(argv[i], "-xy") == 0 && i < argc-1) {
                param.localDimX = atoi(argv[++i]);
                param.localDimY = param.localDimX;
            } else if(strcmp(argv[i], "-mpix") == 0 && i < argc-1)
                param.mpiNumX = atoi(argv[++i]);
            else if(strcmp(argv[i], "-mpiy") == 0 && i < argc-1)
                param.mpiNumY = atoi(argv[++i]);
            else if(strcmp(argv[i], "-num") == 0 && i < argc-1)
                param.loops = atoi(argv[++i]);
            else if(strcmp(argv[i], "-cpu") == 0)
                param.cpuonly = true;
            else if(strcmp(argv[i], "-wgs") == 0 && i < argc-1)
                param.workgroupsize = atof(argv[++i]);
            else if(strcmp(argv[i], "-gpuinfo") == 0)
                param.giveOpenClDeviceName = true;
            else if(strcmp(argv[i], "-print") == 0 && i < argc-1)
                param.printMpiRank = atoi(argv[++i]);
        }
    }

    if(param.mpiNumX == 0 || param.mpiNumY == 0) {
        param.mpiNumX = sqrt(mpiSize);
        param.mpiNumY = param.mpiNumX;
    }

    if(param.mpiNumX*param.mpiNumY != mpiSize) {
        printf("ERROR: Total number of MPI ranks requested (%ix%i) doesn't match global MPI size of %i... Abort!", param.mpiNumX, param.mpiNumY, mpiSize);
        exit(1);
    }

    if(mpiRank == 0) {

        printf("\n"
               "localDimX     = %i\n"
               "localDimY     = %i\n"
               "portionGPU    = %f\n"
               "mpiNumX       = %i\n"
               "mpiNumY       = %i\n"
               "loops         = %i\n"
               "version       = %s\n"
               "workgroupsize = %i\n\n",
               param.localDimX, param.localDimY, param.portionGPU, param.mpiNumX, param.mpiNumY, param.loops, (param.cpuonly ? "CPU-only" : "CPU/GPU"), param.workgroupsize);

    }

    param.gpuDimX = param.localDimX*sqrt(param.portionGPU), param.gpuDimY = param.localDimY*sqrt(param.portionGPU);

    int num = (param.localDimX+2)*(param.localDimY+2);
    param.cpu = malloc(num*sizeof(double));

    if(param.cpuonly) {
        for(int j = 0; j < param.localDimY; ++j)
            for(int i = 0; i < param.localDimX; ++i)
                param.cpu[(j+1)*(param.localDimX+2) + i+1] = (double)j*param.localDimX+i+1;
    } else {
        for(int j = 0; j < param.localDimY; ++j)
            for(int i = 0; i < param.localDimX; ++i)
                if(!(i >= (param.localDimX-param.gpuDimX)/2 && i < (param.localDimX-param.gpuDimX)/2+param.gpuDimX
                   && j >= (param.localDimY-param.gpuDimY)/2 && j < (param.localDimY-param.gpuDimY)/2+param.gpuDimY))
                    param.cpu[(j+1)*(param.localDimX+2) + i+1] = (double)j*param.localDimX+i+1;
    }

    param.tau = tausch_newCpuAndGpu(param.localDimX, param.localDimY, param.mpiNumX, param.mpiNumY, true, !param.cpuonly, true, param.workgroupsize, param.giveOpenClDeviceName);

    if(!param.cpuonly) {

        // how many points only on the device and an OpenCL buffer for them
        param.gpu = malloc((param.gpuDimX+2)*(param.gpuDimY+2)*sizeof(double));

        for(int j = 0; j < param.gpuDimY+2; ++j) {
            for(int i = 0; i < param.gpuDimX+2; ++i) {
                double val = (double)((j-1)*param.gpuDimX+i);
                if(j == 0 || i == 0 || j == param.gpuDimY+1 || i == param.gpuDimX+1)
                    val = 0;
                param.gpu[j*(param.gpuDimX+2) + i] = val;
            }
        }

        cl_int err;
        param.clGpu = clCreateBuffer(tausch_getContext(param.tau), CL_MEM_READ_WRITE, (param.gpuDimX+2)*(param.gpuDimY+2)*sizeof(double), NULL, &err);
        tausch_checkOpenCLError(param.tau, err, "create GPU buffer");
        err = clEnqueueWriteBuffer(tausch_getQueue(param.tau), param.clGpu, true, 0, (param.gpuDimX+2)*(param.gpuDimY+2)*sizeof(double), param.gpu, 0, NULL, NULL);
        tausch_checkOpenCLError(param.tau, err, "fill GPU buffer");

    }



    tausch_setCPUData(param.tau, param.cpu);
    if(!param.cpuonly)
        tausch_setGPUData(param.tau, param.clGpu, param.gpuDimX, param.gpuDimY);

    if(mpiRank == param.printMpiRank) {
        if(!param.cpuonly) {
            printf("-------------------------------\n");
            printf("-------------------------------\n");
            printf("GPU region BEFORE\n");
            printf("-------------------------------\n");
            printGPU(&param);
        }
        printf("-------------------------------\n");
        printf("-------------------------------\n");
        printf("CPU region BEFORE\n");
        printf("-------------------------------\n");
        printCPU(&param);
        printf("-------------------------------\n");
    }

    pthread_t thrdCPU, thrdGPU;

    MPI_Barrier(MPI_COMM_WORLD);
    clock_t begin = clock();

    pthread_create(&thrdCPU, NULL, launchCPU, &param);
    if(!param.cpuonly)
        pthread_create(&thrdGPU, NULL, launchGPU, &param);

    pthread_join(thrdCPU, NULL);
    if(!param.cpuonly)
        pthread_join(thrdGPU, NULL);

    MPI_Barrier(MPI_COMM_WORLD);
    clock_t end = clock();
    double time_spent = 1000.0 * (double)(end - begin) / CLOCKS_PER_SEC;

    if(mpiRank == 0)
        printf("Time required: %f ms\n", time_spent);

    if(mpiRank == param.printMpiRank) {
        if(!param.cpuonly) {
            printf("-------------------------------\n");
            printf("-------------------------------\n");
            printf("GPU region AFTER\n");
            printf("-------------------------------\n");
            printGPU(&param);
        }
        printf("-------------------------------\n");
        printf("-------------------------------\n");
        printf("CPU region AFTER\n");
        printf("-------------------------------\n");
        printCPU(&param);
        printf("-------------------------------\n");
    }

    tausch_delete(param.tau);

    MPI_Finalize();

    return 0;

}

void printCPU(struct param_struct *param) {

    if(param->cpuonly) {
        for(int j = param->localDimY+2 -1; j >= 0; --j) {
            for(int i = 0; i < param->localDimX+2; ++i)
                printf("%3.0f ",param->cpu[j*(param->localDimX+2) + i]);
            printf("\n");
        }
    } else {
        for(int j = param->localDimY+2 -1; j >= 0; --j) {
            for(int i = 0; i < param->localDimX+2; ++i) {
                if(i-1 > (param->localDimX-param->gpuDimX)/2 && i < (param->localDimX-param->gpuDimX)/2+param->gpuDimX
                   && j-1 > (param->localDimY-param->gpuDimY)/2 && j < (param->localDimY-param->gpuDimY)/2+param->gpuDimY)
                    printf("    ");
                else
                    printf("%3.0f ",param->cpu[j*(param->localDimX+2) + i]);
            }
            printf("\n");
        }
    }

}
void printGPU(struct param_struct *param) {

    for(int i = param->gpuDimY+2 -1; i >= 0; --i) {
        for(int j = 0; j < param->gpuDimX+2; ++j)
            printf("%3.0f ", param->gpu[i*(param->gpuDimX+2) + j]);
        printf("\n");
    }

}
