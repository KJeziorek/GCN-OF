#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>

using namespace torch;

// ----------------------------------------------------------------------
// Generate graph edges with optional Spatio-Temporal Contrast filtering
// Input:  events [N,4]  (x,y,t,p)
// Output: features[N], positions[N,3], edges[E,2]
// ----------------------------------------------------------------------
std::tuple<Tensor, Tensor, Tensor>
generate_edges_torch(const Tensor& events,
                     int radius,
                     int width,
                     int height,
                     bool stc_enable = false,
                     int64_t dt_confirm = 100)   // e.g. if t normalized 0-128 → 200 acceptable
{
    TORCH_CHECK(events.dim() == 2 && events.size(1) == 4,
                "events tensor must be [N,4]");

    Tensor events_cpu = events.cpu().to(torch::kInt64);
    auto accessor = events_cpu.accessor<int64_t, 2>();
    const int64_t num_events = events_cpu.size(0);
    const int64_t radius_sq  = radius * radius;

    std::vector<int64_t> vec_features;
    std::vector<int64_t> vec_positions;
    std::vector<int64_t> vec_edges;

    vec_features.reserve(num_events);
    vec_positions.reserve(num_events * 3);
    vec_edges.reserve(num_events * 4);

    std::vector<std::vector<int64_t>> last_time(width,  std::vector<int64_t>(height, -1));
    std::vector<std::vector<int64_t>> last_idx(width,   std::vector<int64_t>(height, -1));

    int64_t node_idx = 0;

    for (int64_t i = 0; i < num_events; i++) {
        int64_t x = accessor[i][0];
        int64_t y = accessor[i][1];
        int64_t t = accessor[i][2];
        int64_t p = accessor[i][3];

        // skip duplicates
        if (last_time[x][y] == t) continue;

        // ---------------- STC PREFILTER OPTIONAL --------------------
        if (stc_enable && last_time[x][y] >= 0) {
            int64_t dt = t - last_time[x][y];
            if (dt > dt_confirm) {
                // reject this event (not supported by close previous event)
                last_time[x][y] = t;
                continue;
            }
        }
        // ------------------------------------------------------------

        // Add node (self-loop edge)
        vec_edges.push_back(node_idx);
        vec_edges.push_back(node_idx);

        vec_features.push_back(p);
        vec_positions.push_back(x);
        vec_positions.push_back(y);
        vec_positions.push_back(t);

        int x_start = std::max<int>(0, x - radius);
        int x_end   = std::min<int>(width - 1, x + radius);
        int y_start = std::max<int>(0, y - radius);
        int y_end   = std::min<int>(height - 1, y + radius);

        for (int nx = x_start; nx <= x_end; nx++) {
            for (int ny = y_start; ny <= y_end; ny++) {
                if (last_time[nx][ny] == -1) continue;

                int64_t dx = x - nx;
                int64_t dy = y - ny;
                int64_t dt = t - last_time[nx][ny];
                int64_t dist_sq = dx * dx + dy * dy + dt * dt;

                if (dist_sq <= radius_sq) {
                    vec_edges.push_back(node_idx);
                    vec_edges.push_back(last_idx[nx][ny]);
                }
            }
        }

        last_time[x][y] = t;
        last_idx[x][y] = node_idx;
        node_idx++;
    }

    Tensor features  = torch::from_blob(vec_features.data(),  { (long)vec_features.size() }, torch::kInt64).clone();
    Tensor positions = torch::from_blob(vec_positions.data(), { (long)vec_positions.size()/3, 3 }, torch::kInt64).clone();
    Tensor edges     = torch::from_blob(vec_edges.data(),     { (long)vec_edges.size()/2, 2 }, torch::kInt64).clone();

    return {features, positions, edges};
}

namespace py = pybind11;

PYBIND11_MODULE(matrix_neighbour, m) {
    m.doc() = "Event graph edge generator with optional STC filtering";

    m.def("generate_edges",
          &generate_edges_torch,
          py::arg("events"),
          py::arg("radius"),
          py::arg("width"),
          py::arg("height"),
          py::arg("stc_enable") = false,
          py::arg("dt_confirm") = 200,
          "Generate graph edges from event data with optional STC prefilter");
}
