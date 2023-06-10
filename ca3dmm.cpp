// @TODO : valgrind?

#include <iostream>
#include <cmath>
#include <vector>
#include <cblas.h>
#include <mpi.h>
#include "densematgen.h"

struct Proc_grid
{
    int64_t p_m;
    int64_t p_n;
    int64_t p_k;
    int64_t n;
    int64_t m;
};

Proc_grid get_grid_size(int64_t num_processes, int64_t n, int64_t m, int64_t k)
{
    Proc_grid grid;
    int64_t best_res = num_processes * k * n + m * k + m * n;
    for (int64_t p_m = 1; p_m <= num_processes; p_m++)
    {
        for (int64_t p_n = 1; p_n <= num_processes; p_n++)
        {
            for (int64_t p_k = 1; p_k <= num_processes; p_k++)
            {
                if (p_m * p_n * p_k > num_processes)
                {
                    break;
                }
                if (20 * p_m * p_n * p_k >= num_processes * 19 && std::max(p_m, p_n) % std::min(p_m, p_n) == 0)
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

class Matrix
{
public:
    int64_t x_size, y_size, start_x, start_y;
    double *mat;
    Matrix(int64_t x_size_, int64_t y_size_, int64_t start_x_, int64_t start_y_)
    {
        this->x_size = x_size_;
        this->y_size = y_size_;
        start_x = start_x_;
        start_y = start_y_;
        mat = (double *)calloc(x_size * y_size, sizeof(double));
    }
    void getWhole(int64_t seed)
    {
        for (int64_t i = 0; i < y_size; i++)
        {
            for (int64_t z = 0; z < x_size; z++)
            {
                mat[i * x_size + z] = generate_double(seed, i + start_y, z + start_x);
            }
        }
    }

    void getPart(int64_t get_x_start, int64_t get_x_end, int64_t get_y_start, int64_t get_y_end, int64_t seed, MPI_Comm *cross_cannon_group)
    {
        for (int64_t i = get_y_start; i < get_y_end; i++)
        {
            for (int64_t z = get_x_start; z < get_x_end; z++)
            {
                mat[(i - start_y) * x_size + z - start_x] = generate_double(seed, i, z);
            }
        }
        int64_t part_size = (get_y_end - get_y_start) * (get_x_end - get_x_start);
        int processes;
        MPI_Comm_size(*cross_cannon_group, &processes);
        int recvcounts[processes], displs[processes];
        int64_t count_y_start = 0;
        for (int64_t i = 0; i < processes; i++)
        {
            int64_t count_y_end = y_size * (i + 1) / processes;
            recvcounts[i] = (count_y_end - count_y_start) * x_size;
            displs[i] = count_y_start * x_size;
            count_y_start = count_y_end;
        }
        MPI_Allgatherv(mat + (get_y_start - start_y) * x_size, part_size,
                       MPI_DOUBLE, mat, recvcounts,
                       displs, MPI_DOUBLE, *cross_cannon_group);
    }
    ~Matrix()
    {
        // free(mat);
    }
};
void performCannon(Matrix a, Matrix b, int64_t A_group_x_length, int64_t B_group_y_length, MPI_Comm *cannon_group_comm, int64_t my_group, int64_t myGroupRank, bool verbose, double count_greater, Proc_grid g, MPI_Comm *world_comm)
{

    Matrix c{b.x_size, a.y_size, b.start_x, a.start_y};

    int cannonGroupSize;
    MPI_Comm_size(*cannon_group_comm, &cannonGroupSize);
    int64_t cannonGroupAxisLength = (int64_t)sqrt(cannonGroupSize);
    int myCannonGroupRank;
    MPI_Comm_rank(*cannon_group_comm, &myCannonGroupRank);
    int64_t A_me_in_group_y = myCannonGroupRank / cannonGroupAxisLength;
    int64_t A_me_in_group_x = myCannonGroupRank % cannonGroupAxisLength;

    int64_t B_me_in_group_x = myCannonGroupRank % cannonGroupAxisLength;
    int64_t B_me_in_group_y = myCannonGroupRank / cannonGroupAxisLength;

    for (int64_t i = 0; i < cannonGroupAxisLength; i++)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    a.y_size, b.x_size, a.x_size,
                    1.0, a.mat, a.x_size, b.mat, b.x_size, 1.0, c.mat, c.x_size);
        int64_t dest_a = A_me_in_group_y * cannonGroupAxisLength + (A_me_in_group_x + cannonGroupAxisLength - 1) % cannonGroupAxisLength;
        int64_t recv_a = A_me_in_group_y * cannonGroupAxisLength + (A_me_in_group_x + cannonGroupAxisLength + 1) % cannonGroupAxisLength;
        int64_t dest_b = (B_me_in_group_y - 1 + cannonGroupAxisLength) % cannonGroupAxisLength * cannonGroupAxisLength + B_me_in_group_x;
        int64_t recv_b = (B_me_in_group_y + 1) % cannonGroupAxisLength * cannonGroupAxisLength + B_me_in_group_x;

        int64_t A_my_x_start = A_group_x_length * ((A_me_in_group_x + i + 1 + A_me_in_group_y + 2 * cannonGroupAxisLength) % cannonGroupAxisLength) / cannonGroupAxisLength;
        int64_t A_my_x_end = A_group_x_length * (((A_me_in_group_x + i + 1 + A_me_in_group_y + 2 * cannonGroupAxisLength) % cannonGroupAxisLength) + 1) / cannonGroupAxisLength;
        Matrix a_new{A_my_x_end - A_my_x_start, a.y_size, 0, 0};

        MPI_Sendrecv(a.mat, a.x_size * a.y_size, MPI_DOUBLE,
                     dest_a, i, a_new.mat, a_new.x_size * a_new.y_size,
                     MPI_DOUBLE, recv_a, i,
                     *cannon_group_comm, MPI_STATUS_IGNORE);

        Matrix b_new{b.x_size, A_my_x_end - A_my_x_start, 0, 0};
        MPI_Sendrecv(b.mat, b.x_size * b.y_size, MPI_DOUBLE,
                     dest_b, i, b_new.mat, b_new.x_size * b_new.y_size,
                     MPI_DOUBLE, recv_b, i,
                     *cannon_group_comm, MPI_STATUS_IGNORE);

        a = a_new;
        b = b_new;
    }
    int myRank, cross_group_comm_rank;
    MPI_Comm_rank(*world_comm, &myRank);

    MPI_Comm cross_group_comm{};
    MPI_Comm_split(*world_comm, myGroupRank, myRank, &cross_group_comm);
    MPI_Comm_rank(cross_group_comm, &cross_group_comm_rank);
    Matrix c2{c.x_size, c.y_size, c.start_x, c.start_y};
    MPI_Reduce(c.mat, c2.mat, c.x_size * c.y_size, MPI_DOUBLE, MPI_SUM, 0, cross_group_comm);
    MPI_Comm print64_ting_g_comm{};
    if (cross_group_comm_rank == 0)
    {
        MPI_Comm_split(*world_comm, 0, myRank, &print64_ting_g_comm);
        int print64_t_rank;
        MPI_Comm_rank(print64_ting_g_comm, &print64_t_rank);
        if (verbose)
        {

            if (print64_t_rank == 0)
            {
                std::cout << g.m << " " << g.n << std::endl;
                for (int64_t i = 0; i < g.m; i++)
                {
                    for (int64_t z = 0; z < g.n; z++)
                    {
                        if (c2.start_x <= z && c2.start_x + c2.x_size > z && c2.start_y <= i && c2.start_y + c2.y_size > i)
                        {
                            std::cout << c2.mat[(i - c2.start_y) * c2.x_size + z - c2.start_x] << " ";
                        }
                        else
                        {
                            double val;
                            MPI_Recv(&val, 1, MPI_DOUBLE, MPI_ANY_SOURCE, i * g.n + z,
                                     print64_ting_g_comm, MPI_STATUS_IGNORE);
                            std::cout << val << " ";
                        }
                    }
                    std::cout << std::endl;
                }
            }
            else
            {
                for (int64_t i = 0; i < c2.y_size; i++)
                {
                    for (int64_t z = 0; z < c2.x_size; z++)
                    {
                        MPI_Send(&c2.mat[i * c2.x_size + z], 1, MPI_DOUBLE, 0, (c2.start_y + i) * g.n + c2.start_x + z, print64_ting_g_comm);
                    }
                }
            }
        }
        else
        {
            int64_t count = 0, count2 = 0;
            for (int64_t i = 0; i < c2.y_size; i++)
            {
                for (int64_t z = 0; z < c2.x_size; z++)
                {
                    if (c2.mat[i * c2.x_size + z] > count_greater)
                    {
                        count++;
                    }
                }
            }
            MPI_Reduce(&count, &count2, 1, MPI_LONG_LONG, MPI_SUM, 0, print64_ting_g_comm);
            if (print64_t_rank == 0)
            {
                std::cout << count2 << std::endl;
            }
        }
    }
    else
    {
        MPI_Comm_split(*world_comm, 1, myRank, &print64_ting_g_comm);
    }
}

void prepare_cannon(int64_t group, int64_t n, int64_t m, int64_t k, Proc_grid g, std::pair<int64_t, int64_t> seeds, bool verbose, double count_greater, MPI_Comm *group_comm, MPI_Comm *world_comm)
{
    int myGroupRank;
    MPI_Comm_rank(*group_comm, &myGroupRank);

    int64_t cannonGroupCount = (std::max(g.p_m, g.p_n) / std::min(g.p_m, g.p_n));
    int64_t cannonGroupSize = g.p_m * g.p_n / cannonGroupCount; // square of some number
    int64_t cannonGroupAxisLength = (int64_t)sqrt(cannonGroupSize);
    int64_t cannonGroup = myGroupRank / cannonGroupSize;

    MPI_Comm cannon_group_comm{};
    MPI_Comm_split(*group_comm, cannonGroup, myGroupRank, &cannon_group_comm);
    int myCannonGroupRank;
    MPI_Comm_rank(cannon_group_comm, &myCannonGroupRank);

    MPI_Comm cross_cannon_group_comm{};
    int64_t crossCannonGroup = myGroupRank % cannonGroupSize;

    MPI_Comm_split(*group_comm, crossCannonGroup, myGroupRank, &cross_cannon_group_comm);

    int64_t A_group_x_start, A_group_y_start, A_group_x_end, A_group_y_end, B_group_x_start, B_group_y_start, B_group_x_end, B_group_y_end;

    A_group_x_start = k * group / g.p_k;
    A_group_x_end = (group + 1) * k / g.p_k;
    B_group_y_start = k * group / g.p_k;
    B_group_y_end = (group + 1) * k / g.p_k;

    if (g.p_m >= g.p_n)
    { // we are not replicating A
        A_group_y_start = (m * cannonGroup / cannonGroupCount);
        A_group_y_end = m * (cannonGroup + 1) / cannonGroupCount;

        B_group_x_start = 0;
        B_group_x_end = n;
    }
    else
    { // we are not replicating B
        A_group_y_start = 0;
        A_group_y_end = m;

        B_group_x_start = n * cannonGroup / cannonGroupCount;
        B_group_x_end = n * (cannonGroup + 1) / cannonGroupCount;
    }

    int64_t A_group_x_length = A_group_x_end - A_group_x_start;
    int64_t A_group_y_length = A_group_y_end - A_group_y_start;
    int64_t B_group_y_length = B_group_y_end - B_group_y_start;
    int64_t B_group_x_length = B_group_x_end - B_group_x_start;

    int64_t A_me_in_group_x = ((myCannonGroupRank / cannonGroupAxisLength) + (myCannonGroupRank % cannonGroupAxisLength)) % cannonGroupAxisLength;
    int64_t A_me_in_group_y = myCannonGroupRank / cannonGroupAxisLength;

    int64_t A_my_x_start = A_group_x_start + A_group_x_length * A_me_in_group_x / cannonGroupAxisLength;
    int64_t A_my_x_end = A_group_x_start + A_group_x_length * (A_me_in_group_x + 1) / cannonGroupAxisLength;
    int64_t A_my_y_start = A_group_y_start + A_group_y_length * A_me_in_group_y / cannonGroupAxisLength;
    int64_t A_my_y_end = A_group_y_start + A_group_y_length * (A_me_in_group_y + 1) / cannonGroupAxisLength;

    Matrix a{A_my_x_end - A_my_x_start, A_my_y_end - A_my_y_start, A_my_x_start, A_my_y_start};

    if (g.p_m >= g.p_n)
    { // we are not replicating A

        a.getWhole(seeds.first);
    }
    else
    { // we are not replicating B;
        int64_t A_my_get_x_start = A_my_x_start;
        int64_t A_my_get_x_end = A_my_x_end;
        int64_t A_my_get_y_start = A_my_y_start + (A_my_y_end - A_my_y_start) * cannonGroup / cannonGroupCount;
        int64_t A_my_get_y_end = A_my_y_start + (A_my_y_end - A_my_y_start) * (cannonGroup + 1) / cannonGroupCount;
        a.getPart(A_my_get_x_start, A_my_get_x_end, A_my_get_y_start, A_my_get_y_end, seeds.first, &cross_cannon_group_comm);
    }

    int64_t B_me_in_group_x = myCannonGroupRank % cannonGroupAxisLength;
    int64_t B_me_in_group_y = (myCannonGroupRank % cannonGroupAxisLength + myCannonGroupRank / cannonGroupAxisLength) % cannonGroupAxisLength;

    int64_t B_my_x_start = B_group_x_start + B_me_in_group_x * B_group_x_length / cannonGroupAxisLength;
    int64_t B_my_x_end = B_group_x_start + (B_me_in_group_x + 1) * B_group_x_length / cannonGroupAxisLength;
    int64_t B_my_y_start = B_group_y_start + B_me_in_group_y * B_group_y_length / cannonGroupAxisLength;
    int64_t B_my_y_end = B_group_y_start + (B_me_in_group_y + 1) * B_group_y_length / cannonGroupAxisLength;

    Matrix b{B_my_x_end - B_my_x_start, B_my_y_end - B_my_y_start, B_my_x_start, B_my_y_start};

    if (g.p_m >= g.p_n)
    { // we are not replicating A

        int64_t B_my_get_x_start = B_my_x_start;
        int64_t B_my_get_x_end = B_my_x_end;
        int64_t B_my_get_y_start = B_my_y_start + (B_my_y_end - B_my_y_start) * cannonGroup / cannonGroupCount;
        int64_t B_my_get_y_end = B_my_y_start + (B_my_y_end - B_my_y_start) * (cannonGroup + 1) / cannonGroupCount;

        b.getPart(B_my_get_x_start, B_my_get_x_end, B_my_get_y_start, B_my_get_y_end, seeds.second, &cross_cannon_group_comm);
    }
    else
    {
        // we are not replicating B
        b.getWhole(seeds.second);
    }
    performCannon(a, b, A_group_x_length, B_group_y_length, &cannon_group_comm, group, myGroupRank, verbose, count_greater, g, world_comm);
}

int main(int argc, char *argv[])
{
    if (argc < 6)
    {
        std::cerr << "Not enough arguments\n";
        return 1;
    }
    int64_t n = std::stoi(std::string(argv[1]));
    int64_t m = std::stoi(std::string(argv[2]));
    int64_t k = std::stoi(std::string(argv[3]));
    std::vector<std::pair<int64_t, int64_t>> seeds;
    char *seed = strtok(argv[5], ",");
    while (seed != NULL)
    {
        char *seed2 = strtok(NULL, ",");
        seeds.push_back(std::make_pair(std::stoi(std::string(seed)), std::stoi(std::string(seed2))));
        seed = strtok(NULL, ",");
    }
    bool verbose = false;
    double print64_t_greater = 0;
    if (std::string(argv[6]) == "-v")
    {
        verbose = true;
    }
    else
    {
        print64_t_greater = std::stod(std::string(argv[7]));
    }
    int numProcesses, myRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    Proc_grid grid = get_grid_size(numProcesses, n, m, k);
    grid.m = m;
    grid.n = n;

    MPI_Comm world_comm{};
    MPI_Comm_split(MPI_COMM_WORLD, (myRank < grid.p_m * grid.p_n * grid.p_k) ? 0 : 1, myRank, &world_comm);
    if (myRank < grid.p_m * grid.p_n * grid.p_k)
    {
        MPI_Comm group_comm{};
        MPI_Comm_split(world_comm, myRank / (grid.p_m * grid.p_n), myRank % (grid.p_m * grid.p_n), &group_comm);
        int64_t my_group = myRank / (grid.p_m * grid.p_n);

        for (size_t i = 0; i < seeds.size(); i++)
        {
            prepare_cannon(my_group, n, m, k, grid, seeds[i], verbose, print64_t_greater, &group_comm, &world_comm);
        }
    }

    MPI_Finalize();
}