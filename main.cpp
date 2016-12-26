#include "sample.h"

int main(int argc, char** argv) {

    MPI_Init(&argc,&argv);

    Sample sample;

    MPI_Finalize();

    return 0;

}
