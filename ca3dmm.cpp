#include <iostream>
#include <cmath>
#include <vector>
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

class ParMatrix {
public:
    std::vector<std::vector>> mat;
    int x_size, y_size, x_start, x_end, y_start, y_end;
    ParMatrix(int x_size, int y_size, int x_start, int x_end, int y_start, int y_end){
        
    }
    int getValueRel(int x, int y){
        return mat[x][y];
    }
    int getValueAbs(int x, int y){ //unsafe
        return mat[x-x_start][y-y_start];
    } 
};


void run_cannon(int group, int n, int m, int k, Proc_grid g, MPI_Comm* group_comm){
    int myGroupRank, myCannonGroupRank;
    MPI_Comm_rank(group_comm, &myGroupRank);
    MPI_Comm cannon_group_comm{};
    int cannonGroupCount = (std::max(g.p_m, g.p_n) / std::min(g.p_m, g.p_n));
    int cannonGroupSize = g.p_m * g.p_n / cannonGroupCount; //square of some number
    int cannonGroupAxisLength = (int) sqrt(cannonGroupSize);
    int cannonGroup =  myGroupRank / cannonGroupCount;
    MPI_Comm_split(group_comm, cannonGroup , myGroupRank, &cannon_group_comm);
    MPI_Comm_rank(group_comm, &myCannonGroupRank);
    if(g.p_m <= g.p_n){ // we are not replicating A
        int A_group_x_start = ((k+g.p_k-1)/g.p_k)*group;
        int A_group_x_end = std::min(((k+g.p_k-1)/g.p_k)*(group+1), k);
        int A_group_x_length = A_group_x_end - A_group_x_start;
        int A_group_y_start = ((m+g.p_m-1)/g.p_m)*cannonGroup;
        int A_group_y_end = std::min(((m+g.p_m-1)/g.p_m)*(cannonGroup+1), m);
        int A_group_y_length = A_group_y_end - A_group_y_start;
        
        int B_group_y_start = ((k+g.p_k-1)/g.p_k)*group;
        int B_group_y_end = std::min(((k+g.p_k-1)/g.p_k)*(group+1), k);
        int B_group_y_length = B_group_y_end - B_group_y_start;
        int B_group_x_start = 0;
        int B_group_x_end = m;
        int B_group_x_length = B_group_x_end - B_group_x_start;
        
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
        int B_my_get_x_end = B_my_x_start + (B_my_x_end - B_group_x_start) * (cannonGroup+1) / cannonGroupCount;
        int B_my_get_y_start = B_my_y_start;
        int B_my_get_y_end = B_my_y_end;

        
        ParMatrix b{m, k, A_my_x_start, A_my_x_end, A_my_y_start, A_my_y_end};
        b.genPart(B_my_get_x_start, B_my_get_x_end, B_my_get_y_start, B_my_get_y_end);
        //create cross cannon communicators and get whole B then cannon
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
    int numProcesses, myRank;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    Proc_grid grid = get_grid_size(numProcesses, n, m, k);
    MPI_Comm group_comm{};
    MPI_Comm_split(MPI_COMM_WORLD, myRank/grid.p_k, myRank, &group_comm);
    int my_group = myRank/grid.p_k;
    run_cannon(my_group, n, m, k, groid, &group_comm);
    MPI_Finalize();

}