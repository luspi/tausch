#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <mpi.h>
#include "../tausch.h"

int main(int argc, char** argv) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);

    int result = Catch::Session().run(argc, argv);

    MPI_Finalize();

    return result;

}

TEST_CASE("extract compressed halo") {

    const size_t maxnum = 5000;

    std::vector<int> vec(maxnum);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<int32_t> dist;

    for(size_t i = 0; i < maxnum; ++i) {

        while(true) {
            int r = dist(rng);
            if(std::find(vec.begin(), vec.end(), r) == vec.end()) {
                vec[i] = r;
                break;
            }
        }

        if((i+1)%(maxnum/10) == 0)
            std::cout << "generated " << i+1 << " of " << maxnum << " random values" << std::endl;

    }

    std::vector<int> insertthese = {12485, 12487, 12489, 12491,
                                    1010104, 1010105, 1010108, 1010109, 1010112, 1010113, 1010116, 1010117,
                                    565, 568, 571, 574, 577};
    for(size_t i = 0; i < insertthese.size(); ++i)
        if(std::find(vec.begin(), vec.end(), insertthese[i]) == vec.end())
            vec[maxnum/2+i] = insertthese[i];

    std::sort(vec.begin(), vec.end());

    Tausch<double> *tausch = new Tausch<double>(MPI_DOUBLE, MPI_COMM_WORLD, false);

    std::vector<std::array<int, 4> > extract = tausch->extractHaloIndicesWithStride(vec);

    std::vector<int> check(maxnum);

    size_t index = 0;
    for(size_t region = 0; region < extract.size(); ++region) {
        const std::array<int, 4> vals = extract[region];

        const int val_start = vals[0];
        const int val_howmanycols = vals[1];
        const int val_howmanyrows = vals[2];
        const int val_striderows = vals[3];

        for(int rows = 0; rows < val_howmanyrows; ++rows)
            for(int cols = 0; cols < val_howmanycols; ++cols)
                check[index++] = val_start+rows*val_striderows+cols;

    }

    for(size_t i = 0; i < maxnum; ++i)
        REQUIRE(vec[i]==check[i]);

}
