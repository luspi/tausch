#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <stdbool.h>
#include <string.h>
#include "../../ctausch.h"

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    size_t size = 10*10;

    double *buf1 = (double*)calloc(size, sizeof(double));
    double *buf2 = (double*)calloc(size, sizeof(double));

    for(int i = 0; i < 10; ++i)
        for(int j = 0; j < 10; ++j) {
            buf1[i*10+j] = i*10+j+1;
            buf2[i*10+j] = 0;
        }

    unsigned char *buf1_c = (unsigned char*)calloc(size*sizeof(double), sizeof(unsigned char*));
    unsigned char *buf2_c = (unsigned char*)calloc(size*sizeof(double), sizeof(unsigned char*));
    memcpy(buf1_c, buf1, size*sizeof(double));

    CTausch *t = tausch_new(MPI_COMM_WORLD, false);

    int *indices = (int*)malloc(10*sizeof(int));
    for(int i = 0; i < 10; ++i)
        indices[i] = 25+2*i;

    tausch_addSendHaloInfo(t, indices, 10, sizeof(double), 0);
    tausch_addRecvHaloInfo(t, indices, 10, sizeof(double), 0);

    tausch_packSendBuffer(t, 0, 0, buf1_c);

    tausch_send(t, 0, 0, 0, 0, false, MPI_COMM_WORLD);
    tausch_recv(t, 0, 0, 0, 0, true, MPI_COMM_WORLD);

    tausch_unpackRecvBuffer(t, 0, 0, buf2_c);

    memcpy(buf2, buf2_c, size*sizeof(double));


    double res[10] = {26,28,30,32,34,36,38,40,42,44};

    for(int i = 0; i < 10; ++i) {

        if(buf2[25+2*i] != res[i]) {
            printf("C API (CPU) FAILED! %f != %f\n", buf2[25+2*i], res[i]);
            exit(1);
        }

    }

    printf("C API (CPU) test passed successfully.\n");

    MPI_Finalize();

    return 0;

}
