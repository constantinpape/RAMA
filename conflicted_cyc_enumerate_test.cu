#include "conflicted_cyc_enumerate.h"
#include <thrust/device_vector.h>
#include "utils.h"
#include "dCOO.h"

int main(int argc, char** argv)
{
    const std::vector<int> i = {0, 1, 0, 2, 3, 0, 2, 0, 3, 4, 5, 4};
    const std::vector<int> j = {1, 2, 2, 3, 4, 3, 4, 4, 5, 5, 6, 6};
    const std::vector<float> costs = {2., 3., -1., 4., 1.5, 5., 2., -2., -3., 2., -1.5, 0.5};

    dCOO A(i.begin(), i.end(), j.begin(), j.end(), costs.begin(), costs.end());
    thrust::device_vector<int> triangles_v1, triangles_v2, triangles_v3;
    std::tie(triangles_v1, triangles_v2, triangles_v3) = enumerate_conflicted_cycles(A, 3);

    std::cout<<"Found triangles: \n";

    for (int t = 0; t < triangles_v1.size(); t++)
    {
        std::cout<<triangles_v1[t]<<" "<<triangles_v2[t]<<" "<<triangles_v3[t]<<"\n";
    }
}
