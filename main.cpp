#include "sample.h"

int main(int argc, char** argv) {

    int provided;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);

    // If this feature is not available -> abort
    if(provided != MPI_THREAD_MULTIPLE){
        std::cout << "ERROR: The MPI library does not have full thread support" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Sample sample;

    MPI_Finalize();

    return 0;

}
