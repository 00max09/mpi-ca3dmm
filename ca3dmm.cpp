#include <iostream>
#include <cmath>
#include <vector>
#include <cblas.h>
#include <mpi.h>
#include "densematgen.h"

struct Proc_grid
{
    int p_m;
    int p_n;
    int p_k;
};

Proc_grid get_grid_size(int num_processes, int n, int m, int k)
{
    Proc_grid grid;
    long long int best_res = num_processes * k * n + m * k + m * n;
    for (int p_m = 1; p_m <= num_processes; p_m++)
    {
        for (int p_n = 1; p_n <= num_processes; p_n++)
        {
            for (int p_k = 1; p_k <= num_processes; p_k++)
            {
                if (p_m * p_n * p_k > num_processes)
                {
                    break;
                }
                if (p_m * p_n * p_k >= num_processes * 19 / 20 && std::max(p_m, p_n) % std::min(p_m, p_n) == 0)
                {
                    long long conf_cost = p_m * k * n + p_n * m * k + p_k * m * n;
                    if (conf_cost <= best_res)
                    {
                        best_res = conf_cost;
                        grid.p_k = p_k;
                        grid.p_m = p_m;
                        grid.p_n = p_n;
                    }
                }
            }
            if (p_m * p_n > num_processes)
            {
                break;
            }
        }
    }
    return grid;
}

class ParMatrix
{
public:
    int mat[][];
    int x_size, y_size, x_start, x_end, y_start, y_end;
    ParMatrix(int x_size, int y_size, int x_start, int x_end, int y_start, int y_end)
    {
    }
    int getValueRel(int x, int y)
    {
        return mat[x][y];
    }
    int getValueAbs(int x, int y)
    { // unsafe
        return mat[x - x_start][y - y_start];
    }
    void allGather()
    {
    }
};

class Matrix
{
public:
    int mat[][];
    int x_size, y_size;
    Matrix(int x_size_, int y_size_)
    {
        this.x_size = x_size_;
        this.y_size = y_size_;
        mat = calloc(x_size * y_size, sizeof(int));
    }

    ~Matrix()
    {
        free(mat);
    }
} void performCannon(ParMatrix a_, ParMatrix b_, int A_group_y_length, int B_group_x_length, MPI_Comm *cannon_group_comm)
{
    Matrix c{b.x_size, a.y_size};

    // MPI_Comm cannon_group_comm_cart{};
    int cannonGroupAxisLength = (int)sqrt(cannonGroupSize);
    int myGroupRank, myCannonGroupRank;
    MPI_Comm_rank(cannon_group_comm, &myCannonGroupRank);
    int me_in_group_y = myCannonGroupRank / cannonGroupAxisLength;
    int me_in_group_y_act = myCannonGroupRank / cannonGroupAxisLength;

    int me_in_group_x = myCannonGroupRank % cannonGroupAxisLength;
    int me_in_group_x_act = myCannonGroupRank % cannonGroupAxisLength;

    // MPI_Cart_create(cannon_group_comm, 2, dim, period, false, &cannon_group_comm_cart);
    for (int i = 0; i < cannonGroupAxisLength; i++)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    a.y_size, b.x_size, a.x_size,
                    1.0, a.mat, a.y_size, B, b.y_size, 1.0, c.mat, c.y_size);
        int dest_a = me_in_group_y * cannonGroupAxisLength + (me_in_group_x_act + cannonGroupAxisLength - 1) % cannonGroupAxisLength;
        int recv_a = me_in_group_y * cannonGroupAxisLength + (me_in_group_x_act + cannonGroupAxisLength + 1) % cannonGroupAxisLength;
        int dest_b = (me_in_group_y_act + cannonGroupAxisLength - 1) / cannonGroupAxisLength * cannonGroupAxisLength + me_in_group_x;
        int recv_b = (me_in_group_y_act + cannonGroupAxisLength + 1) / cannonGroupAxisLength * cannonGroupAxisLength + me_in_group_x;
        int me_in_group_y_act = (me_in_group_y_act - 1 + cannonGroupAxisLength) % cannonGroupAxisLength;
        int A_my_y_start = A_group_y_length / cannonGroupAxisLength * me_in_group_y_act;
        int A_my_y_end = A_group_y_length / cannonGroupAxisLength * (me_in_group_y_act + 1);
        Matrix a_new{a.x_size, A_my_y_end - A_my_y_start};

        int MPI_Sendrecv(a.mat, a.x_size * a.y_size, MPI_DOUBLE,
                         dest_a, i, a_new.mat, a_new.x_size * a_new.y_size,
                         MPI_DOUBLE, recv_a, i,
                         *cannon_group_comm, MPI_STATUS_IGNORE);

        int me_in_group_x_act = (me_in_group_x_act - 1 + cannonGroupAxisLength) % cannonGroupAxisLength;
        int B_my_x_start = B_group_x_length / cannonGroupAxisLength * me_in_group_x_act;
        int B_my_x_end = B_group_x_length / cannonGroupAxisLength * (me_in_group_x_act + 1);
        Matrix b_new{a.x_size, B_my_x_end - B_my_x_start};
        int MPI_Sendrecv(b.mat, b.x_size * b.y_size, MPI_DOUBLE,
                         dest_b, i, b_new.mat, b_new.x_size * b_new.y_size,
                         MPI_DOUBLE, recv_b, i,
                         *cannon_group_comm, MPI_STATUS_IGNORE);

        ~a();      // that cant work
        ~b();      // that cant work
        a = a_new; // xD
        b = b_new; // XD
    }
}

void prepare_cannon(int group, int n, int m, int k, Proc_grid g, std::pair<int, int> seeds, bool verbose, int count_greater, MPI_Comm *group_comm)
{
    int myGroupRank;
    MPI_Comm_rank(group_comm, &myGroupRank);

    int cannonGroupCount = (std::max(g.p_m, g.p_n) / std::min(g.p_m, g.p_n));
    int cannonGroupSize = g.p_m * g.p_n / cannonGroupCount; // square of some number
    int cannonGroupAxisLength = (int)sqrt(cannonGroupSize);
    int cannonGroup = myGroupRank / cannonGroupCount;

    MPI_Comm cannon_group_comm{};
    MPI_Comm_split(group_comm, cannonGroup, myGroupRank, &cannon_group_comm);
    int myCannonGroupRank;
    MPI_Comm_rank(cannon_group_comm, &myCannonGroupRank);

    MPI_Comm cross_cannon_group_comm{};
    int crossCannonGroup = myGroupRank % cannonGroupSize;

    MPI_Comm_split(group_comm, crossCannonGroup, myGroupRank, &cross_cannon_group_comm);

    int A_group_x_start, A_group_y_start, A_group_x_end, A_group_y_end, B_group_x_start, B_group_y_start, B_group_x_end, B_group_y_end;

    //these are independent of p_m and p_n values
    A_group_x_start = ((k + g.p_k - 1) / g.p_k) * group;
    A_group_x_end = std::min(((k + g.p_k - 1) / g.p_k) * (group + 1), k);
    B_group_y_start = ((k + g.p_k - 1) / g.p_k) * group;
    B_group_y_end = std::min(((k + g.p_k - 1) / g.p_k) * (group + 1), k);
        
    if (g.p_m <= g.p_n)
    { // we are not replicating A
        A_group_y_start = ((m + g.p_m - 1) / g.p_m) * cannonGroup;
        A_group_y_end = std::min(((m + g.p_m - 1) / g.p_m) * (cannonGroup + 1), m);

        B_group_x_start = 0;
        B_group_x_end = m;
    }else
    { // we are not replicating B
        A_group_y_start = 0;
        A_group_y_end = n;

        B_group_x_start = ((n + g.p_n - 1) / g.p_n) * cannonGroup;
        B_group_x_end = std::min(((n + g.p_n - 1) / g.p_n) * (cannonGroup + 1), n);
    }

    int A_group_x_length = A_group_x_end - A_group_x_start;
    int A_group_y_length = A_group_y_end - A_group_y_start;
    int B_group_y_length = B_group_y_end - B_group_y_start;
    int B_group_x_length = B_group_x_end - B_group_x_start;



    //need to start to think :(())
    int A_me_in_group_x = myCannonGroupRank / cannonGroupAxisLength;
    int A_me_in_group_y = myCannonGroupRank % cannonGroupAxisLength;

    int A_my_x_start = A_group_x_start + A_group_x_length / cannonGroupAxisLength * A_me_in_group_x;
    int A_my_x_end = A_group_x_start + A_group_x_length / cannonGroupAxisLength * (A_me_in_group_x + 1);
    int A_my_y_start = A_group_y_start + A_group_y_length / cannonGroupAxisLength * A_me_in_group_y;
    int A_my_y_end = A_group_y_start + A_group_y_length / cannonGroupAxisLength * (A_me_in_group_y + 1);

    ParMatrix a{k, n, A_my_x_start, A_my_x_end, A_my_y_start, A_my_y_end};
    a.genWhole();
    int B_me_in_group_x = myCannonGroupRank / cannonGroupAxisLength;
    int B_me_in_group_y = myCannonGroupRank % cannonGroupAxisLength;

    int B_my_x_start = B_group_x_start + B_group_x_length / cannonGroupAxisLength * B_me_in_group_x;
    int B_my_x_end = B_group_x_start + B_group_x_length / cannonGroupAxisLength * (B_me_in_group_x + 1);
    int B_my_y_start = B_group_y_start + B_group_y_length / cannonGroupAxisLength * B_me_in_group_y;
    int B_my_y_end = B_group_y_start + B_group_y_length / cannonGroupAxisLength * (B_me_in_group_y + 1);

    int B_my_get_x_start = B_my_x_start + (B_my_x_end - B_group_x_start) * cannonGroup / cannonGroupCount;
    int B_my_get_x_end = B_my_x_start + (B_my_x_end - B_group_x_start) * (cannonGroup + 1) / cannonGroupCount;
    int B_my_get_y_start = B_my_y_start;
    int B_my_get_y_end = B_my_y_end;

    ParMatrix b{m, k, A_my_x_start, A_my_x_end, A_my_y_start, A_my_y_end};
    b.genPart(B_my_get_x_start, B_my_get_x_end, B_my_get_y_start, B_my_get_y_end);
    // and get whole B then cannon
    b.allGather();
    performCannon(a, b, A_group_y_length, B_group_x_length);
}
}

int main(int argc, char *argv[])
{
    if (argc < 6)
    {
        std::cerr << "Not enough arguments\n";
        return 1;
    }
    int n = std::stoi(std::string(argv[1]));
    int m = std::stoi(std::string(argv[2]));
    int k = std::stoi(std::string(argv[3]));
    std::vector<std::pair<int, int>> seeds;
    char *seed = strtok(argv[5], ",");
    while (seed != NULL)
    {
        char *seed2 = strtok(NULL, ",");
        seeds.push_back(std::make_pair(std::stoi(std::string(seed)), std::stoi(std::string(seed2))));
        seed = strtok(NULL, ",");
    }
    bool verbose = false;
    int print_greater = 0;
    if (argv[6] == "-v")
    {
        verbose = true;
    }
    else
    {
        print_greater = std::stoi(std::string(argv[7]));
    }
    int numProcesses, myRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    Proc_grid grid = get_grid_size(numProcesses, n, m, k);
    MPI_Comm group_comm{};
    MPI_Comm_split(MPI_COMM_WORLD, myRank / grid.p_k, myRank, &group_comm);
    int my_group = myRank / grid.p_k;
    for (int i = 0; i < seeds.size; i++)
    {
        prepare_cannon(my_group, n, m, k, grid, seeds[i], verbose, print_greater, &group_comm);
    }

    MPI_Finalize();
}