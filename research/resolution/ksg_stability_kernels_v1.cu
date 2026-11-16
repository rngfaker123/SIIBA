// 双精度切比雪夫距离
// 精确邻域查询
// 使用设备内建函数
// 双精度正无穷
#define EXACT_INFINITY __longlong_as_double(0x7ff0000000000000LL)
#define STACK_CAPACITY 128
#define NEIGHBORS 22

__device__ double box_lower(const double *query, const double *lower,
                           const double *upper, const int node, const int dim) {
    double bound = 0.0;
    for (int axis = 0; axis < dim; ++axis) {
        const double q = query[axis];
        const double lo = lower[(long long)node * dim + axis];
        const double hi = upper[(long long)node * dim + axis];
        const double delta = q < lo ? lo - q : (q > hi ? q - hi : 0.0);
        if (delta > bound) bound = delta;
    }
    return bound;
}

__device__ double box_upper(const double *query, const double *lower,
                           const double *upper, const int node, const int dim) {
    double bound = 0.0;
    for (int axis = 0; axis < dim; ++axis) {
        const double a = fabs(query[axis] - lower[(long long)node * dim + axis]);
        const double b = fabs(query[axis] - upper[(long long)node * dim + axis]);
        if (a > bound) bound = a;
        if (b > bound) bound = b;
    }
    return bound;
}

__device__ double point_distance(const double *query, const double *point,
                                const int dim) {
    double distance = 0.0;
    for (int axis = 0; axis < dim; ++axis) {
        const double delta = fabs(query[axis] - point[axis]);
        if (delta > distance) distance = delta;
    }
    return distance;
}

extern "C" __global__ void exact_knn7(
    const double *data, const int dim, const int *left, const int *right,
    const long long *start, const long long *end, const double *lower,
    const double *upper, const double *queries, const long long query_count,
    double *output, int *error) {
    const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= query_count) return;
    const double *query = queries + row * dim;
    double best[NEIGHBORS];
    for (int rank = 0; rank < NEIGHBORS; ++rank) best[rank] = EXACT_INFINITY;
    int stack[STACK_CAPACITY];
    int size = 1;
    stack[0] = 0;
    while (size > 0) {
        const int node = stack[--size];
        // 保留等距分支
        if (box_lower(query, lower, upper, node, dim) > best[NEIGHBORS-1]) continue;
        if (left[node] < 0) {
            for (long long point = start[node]; point < end[node]; ++point) {
                const double distance = point_distance(query, data + point * dim, dim);
                if (isnan(distance)) { atomicOr(error, 2); return; }
                if (distance < best[NEIGHBORS-1]) {
                    int rank = NEIGHBORS-1;
                    while (rank > 0 && distance < best[rank-1]) {
                        best[rank] = best[rank-1];
                        --rank;
                    }
                    best[rank] = distance;
                }
            }
        } else {
            const int a = left[node], b = right[node];
            const double da = box_lower(query, lower, upper, a, dim);
            const double db = box_lower(query, lower, upper, b, dim);
            const int near_node = da <= db ? a : b;
            const int far_node = da <= db ? b : a;
            if (size + 2 > STACK_CAPACITY) { atomicOr(error, 1); return; }
            stack[size++] = far_node;
            stack[size++] = near_node;
        }
    }
    for (int rank = 0; rank < NEIGHBORS; ++rank) output[row * NEIGHBORS + rank] = best[rank];
}

extern "C" __global__ void exact_ball_counts(
    const double *data, const int dim, const int *left, const int *right,
    const long long *start, const long long *end, const double *lower,
    const double *upper, const double *queries, const double *radii,
    const long long query_count, long long *output, int *error) {
    const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= query_count) return;
    const double *query = queries + row * dim;
    const double radius = radii[row];
    long long count = 0;
    int stack[STACK_CAPACITY];
    int size = 1;
    stack[0] = 0;
    while (size > 0) {
        const int node = stack[--size];
        if (box_lower(query, lower, upper, node, dim) > radius) continue;
        if (box_upper(query, lower, upper, node, dim) <= radius) {
            count += end[node] - start[node];
        } else if (left[node] < 0) {
            for (long long point = start[node]; point < end[node]; ++point) {
                const double distance = point_distance(query, data + point * dim, dim);
                if (isnan(distance)) { atomicOr(error, 2); return; }
                if (distance <= radius) ++count;
            }
        } else {
            if (size + 2 > STACK_CAPACITY) { atomicOr(error, 1); return; }
            stack[size++] = left[node];
            stack[size++] = right[node];
        }
    }
    output[row] = count;
}
