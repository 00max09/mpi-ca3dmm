#include <iostream>
#include <mpi.h>

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
        for(int p_n = 1; p_n <= num_processes; p_n++)
        {
            for(int p_k = 1; p_k <= num_processes; p_k++)
            {
                if(p_m * p_n * p_k > num_processes)
                {
                    break;
                }
                if(p_m * p_n * p_k >= num_processes * 19 / 20 && std::max(p_m, p_n) % std::min(p_m, p_n) == 0)
                {
                    long long conf_cost = p_m * k * n + p_n * m * k + p_k * m * n;
                    if(conf_cost <= best_res)
                    {
                        best_res = conf_cost;
                        grid.p_k = p_k;
                        grid.p_m = p_m;
                        grid.p_n = p_n;
                    }
                    
                }
            }
            if(p_m * p_n >num_processes){
                break;
            }
        }
    }
    return grid;
}

void run_cannon(int group, int n, int m, int k, Proc_grid g, MPI_Comm* group_comm){
    int myGroupRank, myCannonGroupRank;
    MPI_Comm_rank(group_comm, &myGroupRank);
    MPI_Comm cannon_group_comm{};
    MPI_Comm_split(group_comm, myGroupRank / (std::max(g.p_m, g.p_n) / std::min(g.p_m, g.p_n)) , myGroupRank, &cannon_group_comm);
    MPI_Comm_rank(group_comm, &cannon_group_comm);
    
    
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
    int numProcesses, myRank;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    Proc_grid grid = get_grid_size(numProcesses, n, m, k);
    MPI_Comm group_comm{};
    MPI_Comm_split(MPI_COMM_WORLD, myRank/grid.p_k, myRank, &group_comm);
    int my_group = myRank/grid.p_k;
    run_cannon(my_group, n, m, k, grid, &group_comm);
    MPI_Finalize();

}