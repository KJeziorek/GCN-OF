#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cstdint>

using namespace torch;

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
generate_edges_torch(const Tensor& events,
                     int radius_xy,
                     int radius_t,
                     int width,
                     int height,
                     bool stc_enable = false,
                     int64_t dt_confirm = 100)
{
    TORCH_CHECK(events.dim() == 2 && events.size(1) == 4,
                "events tensor must be [N,4]");

    Tensor events_cpu = events.cpu().to(torch::kInt64);
    auto accessor = events_cpu.accessor<int64_t, 2>();
    const int64_t num_events = events_cpu.size(0);
    const int64_t radius_sq  = static_cast<int64_t>(radius_t) * static_cast<int64_t>(radius_t);

    std::vector<int64_t> vec_features;
    std::vector<int64_t> vec_positions;
    std::vector<int64_t> vec_edges;
    std::vector<int64_t> vec_normals;
    std::vector<int64_t> vec_flows;

    vec_features.reserve(num_events);
    vec_positions.reserve(num_events * 3);
    vec_edges.reserve(num_events * 4);
    vec_normals.reserve(num_events * 3);
    vec_flows.reserve(num_events * 2);

    // separate valid timestamps from raw timestamps for STC logic
    std::vector<std::vector<int64_t>> last_time_valid(width,  std::vector<int64_t>(height, -1));
    std::vector<std::vector<int64_t>> last_time_raw(width,    std::vector<int64_t>(height, -1));
    std::vector<std::vector<int64_t>> last_idx(width,         std::vector<int64_t>(height, -1));
    std::vector<std::vector<int64_t>> last_p(width,           std::vector<int64_t>(height,  0));
    std::vector<std::vector<bool>>    waiting(width,          std::vector<bool>(height, false));

    int64_t node_idx = 0;

    for (int64_t i = 0; i < num_events; i++) {
        int64_t x = accessor[i][0];
        int64_t y = accessor[i][1];
        int64_t t = accessor[i][2];
        int64_t p = accessor[i][3];

        if (x < 0 || x >= width || y < 0 || y >= height) continue;

        // ---------- STC event-trail logic ----------
        int64_t prev_raw_t = last_time_raw[x][y];
        int64_t dt_raw = (prev_raw_t == -1 ? 0 : (t - prev_raw_t));

        if (stc_enable) {
            if (prev_raw_t == -1) {
                waiting[x][y]       = true;
                last_time_raw[x][y] = t;
                last_p[x][y]        = p;
                continue;
            }

            if (dt_raw <= dt_confirm && p == last_p[x][y] && waiting[x][y]) {
                waiting[x][y] = false; // ACCEPT
            } else {
                // reject this event; do NOT update valid timestamp or idx
                waiting[x][y]       = true;
                last_time_raw[x][y] = t;
                last_p[x][y]        = p;
                continue;
            }
        }
        // -------------------------------------------

        // accept event as graph node
        vec_edges.push_back(node_idx);
        vec_edges.push_back(node_idx);

        vec_features.push_back(p);
        vec_positions.push_back(x);
        vec_positions.push_back(y);
        vec_positions.push_back(t);

        int64_t sum_dx = 0, sum_dy = 0, sum_dt = 0;

        int x_start = std::max<int>(0, x - radius_xy);
        int x_end   = std::min<int>(width  - 1, x + radius_xy);
        int y_start = std::max<int>(0, y - radius_xy);
        int y_end   = std::min<int>(height - 1, y + radius_xy);

        // ---------- neighbor search ----------
        for (int nx = x_start; nx <= x_end; nx++) {
            for (int ny = y_start; ny <= y_end; ny++) {

                if (last_idx[nx][ny] == -1) continue;  // only valid events
                int64_t prev_t   = last_time_valid[nx][ny];
                int64_t prev_idx = last_idx[nx][ny];

                int64_t dx = x - nx;
                int64_t dy = y - ny;
                int64_t dt = t - prev_t;


                int64_t dist_sq = dx * dx + dy * dy + dt * dt;
                if (dist_sq <= radius_sq) {
                    vec_edges.push_back(node_idx);
                    vec_edges.push_back(prev_idx);

                    sum_dx += dx;
                    sum_dy += dy;
                    sum_dt += dt;
                }
            }
        }

        // ---------- normals & flow approximation ----------
        int64_t n_x = sum_dx, n_y = sum_dy, n_t = sum_dt;
        if (n_t < 0) { n_x = -n_x; n_y = -n_y; n_t = -n_t; }

        vec_normals.push_back(n_x);
        vec_normals.push_back(n_y);
        vec_normals.push_back(n_t);

        int64_t v_x = 0, v_y = 0;
        if (n_t != 0) {
            v_x = -n_x / n_t;
            v_y = -n_y / n_t;
        }
        vec_flows.push_back(v_x);
        vec_flows.push_back(v_y);

        // update only valid event trackers
        last_time_valid[x][y] = t;
        last_idx[x][y]        = node_idx;
        last_time_raw[x][y]   = t;
        last_p[x][y]          = p;

        node_idx++;
    }

    Tensor features  = torch::from_blob(vec_features.data(), { (long)vec_features.size() }, torch::kInt64).clone();
    Tensor positions = torch::from_blob(vec_positions.data(), { (long)vec_positions.size()/3, 3 }, torch::kInt64).clone();
    Tensor edges     = torch::from_blob(vec_edges.data(),     { (long)vec_edges.size()/2, 2 }, torch::kInt64).clone();
    Tensor normals   = torch::from_blob(vec_normals.data(),   { (long)vec_normals.size()/3, 3 }, torch::kInt64).clone();
    Tensor flows     = torch::from_blob(vec_flows.data(),     { (long)vec_flows.size()/2, 2 }, torch::kInt64).clone();

    return {features, positions, edges, normals, flows};
}


namespace py = pybind11;

PYBIND11_MODULE(matrix_neighbour, m) {
    m.doc() = "Event graph edge generator with STC and correct temporal consistency";

    m.def("generate_edges",
          &generate_edges_torch,
          py::arg("events"),
          py::arg("radius_xy"),
          py::arg("radius_t"),
          py::arg("width"),
          py::arg("height"),
          py::arg("stc_enable") = true,
          py::arg("dt_confirm") = 100);
}
