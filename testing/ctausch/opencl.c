#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <stdbool.h>
#include <string.h>
#define TAUSCH_OPENCL
#include "../../ctausch.h"

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    CTausch *t = tausch_new(MPI_COMM_WORLD, false);
    tausch_enableOpenCL(t, 0);

    size_t size = 10*10;

    double *buf1 = (double*)calloc(size, sizeof(double));
    double *buf2 = (double*)calloc(size, sizeof(double));

    for(int i = 0; i < 10; ++i)
        for(int j = 0; j < 10; ++j) {
            buf1[i*10+j] = i*10+j+1;
            buf2[i*10+j] = 1;
        }

    cl_mem clbuf1 = clCreateBuffer(tausch_getOclContext(t), CL_MEM_READ_WRITE, size*sizeof(double), NULL, NULL);
    cl_mem clbuf2 = clCreateBuffer(tausch_getOclContext(t), CL_MEM_READ_WRITE, size*sizeof(double), NULL, NULL);

    clEnqueueWriteBuffer(tausch_getOclQueue(t), clbuf1, true, 0, size*sizeof(double), buf1, 0, NULL, NULL);

    double tmp = 0;
    clEnqueueFillBuffer(tausch_getOclQueue(t), clbuf2, &tmp, sizeof(double), 0, size*sizeof(double), 0, NULL, NULL);

    int *indices = (int*)malloc(10*sizeof(int));
    for(int i = 0; i < 10; ++i)
        indices[i] = 25+2*i;

    tausch_addSendHaloInfo(t, indices, 10, sizeof(double), 0);
    tausch_addRecvHaloInfo(t, indices, 10, sizeof(double), 0);

    tausch_packSendBufferOCL(t, 0, 0, clbuf1);

    tausch_send(t, 0, 0, 0, 0, false, MPI_COMM_WORLD);
    tausch_recv(t, 0, 0, 0, 0, true, MPI_COMM_WORLD);

    tausch_unpackRecvBufferOCL(t, 0, 0, clbuf2);

    clEnqueueReadBuffer(tausch_getOclQueue(t), clbuf2, true, 0, size*sizeof(double), buf2, 0, NULL, NULL);

    double res[10] = {26,28,30,32,34,36,38,40,42,44};

    for(int i = 0; i < 10; ++i) {

        if(buf2[25+2*i] != res[i]) {
            printf("C API (OpenCL) FAILED! %f != %f\n", buf2[25+2*i], res[i]);
            exit(1);
        }

    }

    printf("C API (OpenCL) test passed successfully.\n");

    tausch_delete(t);

    MPI_Finalize();

    return 0;

}
