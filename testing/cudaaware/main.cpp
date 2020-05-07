#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
#include <mpi.h>

#define OMPI_SKIP_MPICXX

int main(int argc, char** argv) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);

    int result = Catch::Session().run(argc, argv);

    MPI_Finalize();

    return result;
}
