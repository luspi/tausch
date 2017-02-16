#include "sample.h"

int main(int argc, char** argv) {

    int provided;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);

    // If this feature is not available -> abort
    if(provided != MPI_THREAD_MULTIPLE){
        std::cout << "ERROR: The MPI library does not have full thread support" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    int localDimX = 20, localDimY = 20;
    double portionGPU = std::sqrt(0.5);
    int mpiNumX = std::sqrt(mpiSize);
    int mpiNumY = mpiNumX;
    int loops = 1;

    if(argc > 1) {
        for(int i = 1; i < argc; ++i) {

            if(argv[i] == std::string("-x") && i < argc-1)
                localDimX = atoi(argv[++i]);
            else if(argv[i] == std::string("-y") && i < argc-1)
                localDimY = atoi(argv[++i]);
            else if(argv[i] == std::string("-gpu") && i < argc-1)
                portionGPU = atof(argv[++i]);
            else if(argv[i] == std::string("-xy") && i < argc-1) {
                localDimX = atoi(argv[++i]);
                localDimY = localDimX;
            } else if(argv[i] == std::string("-mpix") && i < argc-1)
                mpiNumX = atoi(argv[++i]);
            else if(argv[i] == std::string("-mpiy") && i < argc-1)
                mpiNumY = atoi(argv[++i]);
            else if(argv[i] == std::string("-num") && i < argc-1)
                loops = atoi(argv[++i]);
        }
    }

    if(mpiRank == 0) {

        std::cout << std::endl
                  << "localDimX  = " << localDimX << std::endl
                  << "localDimY  = " << localDimY << std::endl
                  << "portionGPU = " << portionGPU << std::endl
                  << "mpiNumX    = " << mpiNumX << std::endl
                  << "mpiNumY    = " << mpiNumY << std::endl
                  << std::endl;

    }

    Sample sample(localDimX, localDimY, portionGPU, loops, mpiNumX, mpiNumY);

    MPI_Finalize();

    return 0;

}
