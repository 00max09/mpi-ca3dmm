// TODO : long longi, usunięcie nieużywanych wątków tak aby nie odpierdalały, wypisywanie

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
    int n;
    int m;
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
    int x_size, y_size, start_x, start_y;
    double *mat;
    Matrix(int x_size_, int y_size_, int start_x_, int start_y_)
    {
        this->x_size = x_size_;
        this->y_size = y_size_;
        start_x = start_x_;
        start_y = start_y_;
        mat = (double *)calloc(x_size * y_size, sizeof(double));
    }
    void getWhole(int seed)
    {
        for (int i = 0; i < y_size; i++)
        {
            for (int z = 0; z < x_size; z++)
            {
                mat[i * x_size + z] = generate_double(seed, i + start_y, z + start_x);
            }
        }
    }

    void getPart(int get_x_start, int get_x_end, int get_y_start, int get_y_end, int seed, MPI_Comm *cross_cannon_group)
    {
        for (int i = get_y_start; i < get_y_end; i++)
        {
            for (int z = get_x_start; z < get_x_end; z++)
            {
                mat[(i - start_y) * x_size + z - start_x] = generate_double(seed, i, z);
            }
        }
        int part_size = (get_y_end - get_y_start) * (get_x_end - get_x_start);
        int processes;
        MPI_Comm_size(*cross_cannon_group, &processes);
        int recvcounts[processes], displs[processes];
        int count_y_start = 0;
        for (int i = 0; i < processes; i++)
        {
            int count_y_end = y_size * (i + 1) / processes;
            recvcounts[i] = (count_y_end - count_y_start) * x_size;
            displs[i] = count_y_start * x_size;
            count_y_start = count_y_end;
        }
        // std::cout << part_size << " " << get_x_start << " " << get_x_end << " " << get_y_start << " " << get_y_end << std::endl;
        MPI_Allgatherv(mat + (get_y_start - start_y) * x_size, part_size,
                       MPI_DOUBLE, mat, recvcounts,
                       displs, MPI_DOUBLE, *cross_cannon_group);
        // for (int i = 0; i < y_size; i++)
        //  {
        //         for (int z = 0; z < x_size; z++)
        //         {
        //             std::cout << mat[i * x_size + z] << " ";
        //         }
        //         std::cout << std::endl;
        //     }
    }
    ~Matrix()
    {
        // free(mat);
    }
};
void performCannon(Matrix a, Matrix b, int A_group_x_length, int B_group_y_length, MPI_Comm *cannon_group_comm, int my_group, int myGroupRank, bool verbose, double count_greater, Proc_grid g)
{

    Matrix c{b.x_size, a.y_size, b.start_x, a.start_y};

    int cannonGroupCount = (std::max(g.p_m, g.p_n) / std::min(g.p_m, g.p_n));
    // MPI_Comm cannon_group_comm_cart{};
    int cannonGroupSize;
    MPI_Comm_size(*cannon_group_comm, &cannonGroupSize);
    int cannonGroupAxisLength = (int)sqrt(cannonGroupSize);
    int myCannonGroupRank;
    MPI_Comm_rank(*cannon_group_comm, &myCannonGroupRank);
    // axis for a
    int A_me_in_group_y = myCannonGroupRank / cannonGroupAxisLength;
    int A_me_in_group_x = myCannonGroupRank % cannonGroupAxisLength;

    int B_me_in_group_x = myCannonGroupRank % cannonGroupAxisLength;
    int B_me_in_group_y = myCannonGroupRank / cannonGroupAxisLength;

    // MPI_Cart_create(cannon_group_comm, 2, dim, period, false, &cannon_group_comm_cart);
    for (int i = 0; i < cannonGroupAxisLength; i++)
    {
        // std::cout << i << " " << cannonGroupAxisLength << " " << myCannonGroupRank << " " << a.x_size << " " << a.y_size << " " << b.x_size << " " << b.y_size << std::endl;
        //  std::cout.flush();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    a.y_size, b.x_size, a.x_size,
                    1.0, a.mat, a.x_size, b.mat, b.x_size, 1.0, c.mat, c.x_size);
        int dest_a = A_me_in_group_y * cannonGroupAxisLength + (A_me_in_group_x + cannonGroupAxisLength - 1) % cannonGroupAxisLength;
        int recv_a = A_me_in_group_y * cannonGroupAxisLength + (A_me_in_group_x + cannonGroupAxisLength + 1) % cannonGroupAxisLength;
        int dest_b = (B_me_in_group_y - 1 + cannonGroupAxisLength) % cannonGroupAxisLength * cannonGroupAxisLength + B_me_in_group_x;
        int recv_b = (B_me_in_group_y + 1) % cannonGroupAxisLength * cannonGroupAxisLength + B_me_in_group_x;
        // std::cout << dest_a << " " << recv_a << " " << dest_b << " " << recv_b << "G GGG " << myCannonGroupRank << " XXXX " << cannonGroupAxisLength << std::endl;
        // B_me_in_group_y = (B_me_in_group_y + 1 + cannonGroupAxisLength) % cannonGroupAxisLength;
        // A_me_in_group_x = (A_me_in_group_x + 1 + cannonGroupAxisLength) % cannonGroupAxisLength;

        int A_my_x_start = A_group_x_length * ((A_me_in_group_x + i + 1 + A_me_in_group_y + 2 * cannonGroupAxisLength) % cannonGroupAxisLength) / cannonGroupAxisLength;     //??
        int A_my_x_end = A_group_x_length * (((A_me_in_group_x + i + 1 + A_me_in_group_y + 2 * cannonGroupAxisLength) % cannonGroupAxisLength) + 1) / cannonGroupAxisLength; //??
        Matrix a_new{A_my_x_end - A_my_x_start, a.y_size, 0, 0};

        MPI_Sendrecv(a.mat, a.x_size * a.y_size, MPI_DOUBLE,
                     dest_a, i, a_new.mat, a_new.x_size * a_new.y_size,
                     MPI_DOUBLE, recv_a, i,
                     *cannon_group_comm, MPI_STATUS_IGNORE);
        // std::cout << "ÓWNo" << std::endl;
        int B_my_y_start = B_group_y_length * B_me_in_group_y / cannonGroupAxisLength;     //??
        int B_my_y_end = B_group_y_length * (B_me_in_group_y + 1) / cannonGroupAxisLength; //??
        Matrix b_new{b.x_size, A_my_x_end - A_my_x_start, 0, 0};
        MPI_Sendrecv(b.mat, b.x_size * b.y_size, MPI_DOUBLE,
                     dest_b, i, b_new.mat, b_new.x_size * b_new.y_size,
                     MPI_DOUBLE, recv_b, i,
                     *cannon_group_comm, MPI_STATUS_IGNORE);

        // delete &a; // that cant work
        // delete &b; // that cant work
        a = a_new; // xD
        b = b_new; // XD
    }
    int myRank, cross_group_comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    MPI_Comm cross_group_comm{};
    MPI_Comm_split(MPI_COMM_WORLD, myGroupRank, myRank, &cross_group_comm);
    MPI_Comm_rank(cross_group_comm, &cross_group_comm_rank);
    Matrix c2{c.x_size, c.y_size, c.start_x, c.start_y};
    // std::cout << c.x_size << " " << c.y_size << std::endl;
    MPI_Reduce(
        c.mat,
        c2.mat,
        c.x_size * c.y_size,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        cross_group_comm);
    // std::cout << "CISEM" << std::endl;
    MPI_Comm printing_g_comm{};
    if (cross_group_comm_rank == 0)
    {
        MPI_Comm_split(MPI_COMM_WORLD, 0, myRank, &printing_g_comm);
        int print_rank;
        MPI_Comm_rank(printing_g_comm, &print_rank);
        if (verbose)
        {

            if (print_rank == 0)
            {
                std::cout<<g.m<<" "<<g.n<<std::endl; 
                for (int i = 0; i < g.m; i++)
                {
                    for (int z = 0; z < g.n; z++)
                    {
                        if (c2.start_x <= z && c2.start_x + c2.x_size > z && c2.start_y <= i && c2.start_y + c2.y_size > i)
                        {
                            std::cout << c2.mat[(i - c2.start_y) * c2.x_size + z - c2.start_x] << " ";
                        }
                        else
                        {
                            double val;
                            MPI_Recv(&val, 1, MPI_DOUBLE, MPI_ANY_SOURCE, i * g.n + z,
                                     printing_g_comm, MPI_STATUS_IGNORE);
                            std::cout << val << " ";
                        }
                    }
                    std::cout << std::endl;
                }
            }
            else
            {
                for (int i = 0; i < c2.y_size; i++)
                {
                    for (int z = 0; z < c2.x_size; z++)
                    {
                        MPI_Send(&c2.mat[i * c2.x_size + z], 1, MPI_DOUBLE, 0, (c2.start_y + i) * g.n + c2.start_x + z, printing_g_comm);
                    }
                }
            }
        }
        else
        {
            long long int count = 0, count2 = 0;
            for (int i = 0; i < c2.y_size; i++)
            {
                for (int z = 0; z < c2.x_size; z++)
                {
                    // std::cout<<c2.mat[i * c2.x_size + z]<< " "<<count_greater<<std::endl;
                    if (c2.mat[i * c2.x_size + z] > count_greater)
                    {
                        // std::cout<<"STRDAM"<<std::endl;
                        count++;
                    }
                }
            }
            MPI_Reduce(
                &count,
                &count2,
                1,
                MPI_LONG_LONG_INT,
                MPI_SUM,
                0,
                printing_g_comm);

            if (print_rank == 0)
            {
                std::cout << count2 << std::endl;
            }
        }
    }
    else
    {
        MPI_Comm_split(MPI_COMM_WORLD, 1, myRank, &printing_g_comm);
    }
}

void prepare_cannon(int group, int n, int m, int k, Proc_grid g, std::pair<int, int> seeds, bool verbose, double count_greater, MPI_Comm *group_comm)
{
    int myGroupRank;
    MPI_Comm_rank(*group_comm, &myGroupRank);

    int cannonGroupCount = (std::max(g.p_m, g.p_n) / std::min(g.p_m, g.p_n));
    int cannonGroupSize = g.p_m * g.p_n / cannonGroupCount; // square of some number
    int cannonGroupAxisLength = (int)sqrt(cannonGroupSize);
    int cannonGroup = myGroupRank / cannonGroupSize;

    MPI_Comm cannon_group_comm{};
    MPI_Comm_split(*group_comm, cannonGroup, myGroupRank, &cannon_group_comm);
    int myCannonGroupRank;
    MPI_Comm_rank(cannon_group_comm, &myCannonGroupRank);

    MPI_Comm cross_cannon_group_comm{};
    int crossCannonGroup = myGroupRank % cannonGroupSize;

    MPI_Comm_split(*group_comm, crossCannonGroup, myGroupRank, &cross_cannon_group_comm);

    int A_group_x_start, A_group_y_start, A_group_x_end, A_group_y_end, B_group_x_start, B_group_y_start, B_group_x_end, B_group_y_end;

    // these are independent of p_m and p_n values
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
      //  std::cout<<"SIEMSON JEBSON"<<std::endl;
        A_group_y_start = 0;
        A_group_y_end = m;

        B_group_x_start = n * cannonGroup / cannonGroupCount;
        B_group_x_end = n * (cannonGroup + 1) / cannonGroupCount;
    }

    int A_group_x_length = A_group_x_end - A_group_x_start;
    int A_group_y_length = A_group_y_end - A_group_y_start;
    int B_group_y_length = B_group_y_end - B_group_y_start;
    int B_group_x_length = B_group_x_end - B_group_x_start;

    // need to start to think :(())
    int A_me_in_group_x = ((myCannonGroupRank / cannonGroupAxisLength) + (myCannonGroupRank % cannonGroupAxisLength)) % cannonGroupAxisLength;
    int A_me_in_group_y = myCannonGroupRank / cannonGroupAxisLength;

    int A_my_x_start = A_group_x_start + A_group_x_length * A_me_in_group_x / cannonGroupAxisLength;
    int A_my_x_end = A_group_x_start + A_group_x_length * (A_me_in_group_x + 1) / cannonGroupAxisLength;
    int A_my_y_start = A_group_y_start + A_group_y_length * A_me_in_group_y / cannonGroupAxisLength;
    int A_my_y_end = A_group_y_start + A_group_y_length * (A_me_in_group_y + 1) / cannonGroupAxisLength;

    Matrix a{A_my_x_end - A_my_x_start, A_my_y_end - A_my_y_start, A_my_x_start, A_my_y_start};

    int A_my_get_x_start = A_my_x_start;
    int A_my_get_x_end = A_my_x_end;
    if (g.p_m >= g.p_n)
    { // we are not replicating A

        int A_my_get_y_start = A_my_y_start;
        int A_my_get_y_end = A_my_y_end;

        //  std::cout << A_my_get_x_start << " " << A_my_get_x_end << " " << A_my_get_y_start << " " << A_my_get_y_end << " A " << myCannonGroupRank << std::endl;
        a.getWhole(seeds.first);
    }
    else
    { // we are not replicating B;
        int A_my_get_y_start = A_my_y_start + (A_my_y_end - A_my_y_start) * cannonGroup / cannonGroupCount;
        int A_my_get_y_end = A_my_y_start + (A_my_y_end - A_my_y_start) * (cannonGroup + 1) / cannonGroupCount;
        a.getPart(A_my_get_x_start, A_my_get_x_end, A_my_get_y_start, A_my_get_y_end, seeds.first, &cross_cannon_group_comm);
    }

    int B_me_in_group_x = myCannonGroupRank % cannonGroupAxisLength;
    int B_me_in_group_y = (myCannonGroupRank % cannonGroupAxisLength + myCannonGroupRank / cannonGroupAxisLength) % cannonGroupAxisLength;

    int B_my_x_start = B_group_x_start + B_me_in_group_x * B_group_x_length / cannonGroupAxisLength;
    int B_my_x_end = B_group_x_start + (B_me_in_group_x + 1) * B_group_x_length / cannonGroupAxisLength;
    int B_my_y_start = B_group_y_start + B_me_in_group_y * B_group_y_length / cannonGroupAxisLength;
    int B_my_y_end = B_group_y_start + (B_me_in_group_y + 1) * B_group_y_length / cannonGroupAxisLength;

    Matrix b{B_my_x_end - B_my_x_start, B_my_y_end - B_my_y_start, B_my_x_start, B_my_y_start};

    int B_my_get_x_start = B_my_x_start;
    int B_my_get_x_end = B_my_x_end;

    if (g.p_m >= g.p_n)
    { // we are not replicating A

        //  std::cout << "FSDFD " << myGroupRank << " " << cannonGroup << " " << cannonGroupCount << std::endl;
        int B_my_get_y_start = B_my_y_start + (B_my_y_end - B_my_y_start) * cannonGroup / cannonGroupCount;
        int B_my_get_y_end = B_my_y_start + (B_my_y_end - B_my_y_start) * (cannonGroup + 1) / cannonGroupCount;
        //  std::cout << B_my_get_x_start << " " << B_my_get_x_end << " " << B_my_get_y_start << " " << B_my_get_y_end << " G " << myCannonGroupRank << std::endl;
        b.getPart(B_my_get_x_start, B_my_get_x_end, B_my_get_y_start, B_my_get_y_end, seeds.second, &cross_cannon_group_comm);
    }
    else
    {
        // we are not replicating B
        int B_my_get_y_start = B_my_y_start;
        int B_my_get_y_end = B_my_y_end;
        b.getWhole(seeds.second);
    }
    performCannon(a, b, A_group_x_length, B_group_y_length, &cannon_group_comm, group, myGroupRank, verbose, count_greater, g);
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
    double print_greater = 0;
    // std::cout << argv[6] << std::endl;
    if (std::string(argv[6]) == "-v")
    {
        verbose = true;
    }
    else
    {
        print_greater = std::stod(std::string(argv[7]));
    }
    int numProcesses, myRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    Proc_grid grid = get_grid_size(numProcesses, n, m, k);
    grid.m = m;
    grid.n = n;
    std::cout << grid.p_n << " " << grid.p_k << " " << grid.p_m << std::endl;
    MPI_Comm group_comm{};
    MPI_Comm_split(MPI_COMM_WORLD, myRank / (grid.p_m * grid.p_n), myRank % (grid.p_m * grid.p_n), &group_comm);
    int my_group = myRank / (grid.p_m * grid.p_n); //,myRank2;
    // MPI_Comm_rank(group_comm, &myRank2);
    // std::cout<<"RANK IN GROUP "<<my_group<<" "<<std::endl;

    for (int i = 0; i < seeds.size(); i++)
    {
        prepare_cannon(my_group, n, m, k, grid, seeds[i], verbose, print_greater, &group_comm);
    }

    MPI_Finalize();
}