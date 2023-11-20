void compute(int total_count, int my_count, float my_points[][3]) {
  // total_count is the total number of points
  // my_count is the number of points for this process
  // my_points is a float table of shape [my_count][3]

  // 1- Sum over all the points in local_sum
  float local_sum[3] = {0.0f, 0.0f, 0.0f};

  for (int i = 0; i < my_count; ++i) {
        local_sum[0] += my_points[i][0];
        local_sum[1] += my_points[i][1];
        local_sum[2] += my_points[i][2];
    }

  // 2- Reduce the sum of all the points on the variable barycentre 
  float barycentre[3];
  MPI_Allreduce(local_sum, barycentre, 3, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  
  // 3- Divide every component of the barycentre by the number of points
  barycentre[0] /= total_count;
  barycentre[1] /= total_count;
  barycentre[2] /= total_count;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  // For every point
  for (int i = 0; i < my_count; ++i) {
    // 4- Compute the distance for every point
    float dx = my_points[i][0] - barycentre[0];
    float dy = my_points[i][1] - barycentre[1];
    float dz = my_points[i][2] - barycentre[2];
    float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

    
    // And printing the result
    std::cout << rank << " " << dist << std::endl;
  }
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    min_x = std::stod(argv[1]);
    max_x = std::stod(argv[2]);
    min_y = std::stod(argv[3]);
    max_y = std::stod(argv[4]);
    dx = max_x - min_x;
    dy = max_y - min_y;
    cutoff = std::stoi(argv[5]);

    int total_points = p_count * p_count;
    int points_per_process = total_points / size;
    int remaining = total_points % size;

    double *points = NULL;
    double *local_points = new double[points_per_process * 2];
    int *local_mset = new int[points_per_process];

    if (rank == 0) {
        points = new double[p_count * p_count * 2];
        for (int yp = 0; yp < p_count; ++yp) {
            double py = min_y + dy * yp / p_count;
            for (int xp = 0; xp < p_count; ++xp) {
                int lid = yp * p_count * 2 + xp * 2;
                points[lid] = min_x + dx * xp / p_count;
                points[lid + 1] = py;
            }
        }
    }

    MPI_Scatter(points, points_per_process * 2, MPI_DOUBLE,
                local_points, points_per_process * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    compute_mandelbrot(local_points, points_per_process, local_mset);

    int *mset = NULL;
    if (rank == 0) {
        mset = new int[total_points];
    }

    MPI_Gather(local_mset, points_per_process, MPI_INT, mset, 
               points_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < total_points; ++i) {
            std::cout << mset[i] << (i % p_count == p_count - 1 ? "\n" : " ");
        }
        delete[] points;
        delete[] mset;
    }

    delete[] local_points;
    delete[] local_mset;

    MPI_Finalize();
    return 0;
}
