//
// Created by Zhongshi Jiang on 4/17/17.
//

#ifndef SCAFFOLD_TEST_TETRAHEDRAL_IMPROVEMENT_H
#define SCAFFOLD_TEST_TETRAHEDRAL_IMPROVEMENT_H
#include "../igl_dev/edge_removal.h"
#include "../igl_dev/multi_face_removal.h"
#include "../igl_dev/tet_refine_operations.h"
inline void edge_removal_pass(
    std::function<double(int, int,
                         int, int)> tet_quality,
    std::function<bool(int, int,
                       int, int)> orient3D,
    std::vector<Eigen::RowVector4i> &T,
    std::vector<Eigen::RowVector4i> &TT,
    std::vector<Eigen::RowVector4i> &TTif,
    std::vector<Eigen::Matrix<int, 4, 3>> &TTie) {

  // heap making
  int num_T = T.size();
  std::vector<int> tet_time_stamps(num_T, 0);
  using qual_t = std::tuple<double, size_t, int>;
  std::vector<qual_t> tet_qual_heap(num_T);
  for (int i = 0; i < num_T; i++) {
    auto Ti = T[i];
    tet_qual_heap[i] =
        std::make_tuple(-tet_quality(Ti(0), Ti(1), Ti(2), Ti(3)), i, 0);
  }
  std::make_heap(tet_qual_heap.begin(), tet_qual_heap.end());

  // main loop
  while (true) {
    double neg_qual;
    size_t tid;
    int stamp;
    std::pop_heap(tet_qual_heap.begin(), tet_qual_heap.end());
    std::tie(neg_qual, tid, stamp) = tet_qual_heap.back();
    tet_qual_heap.pop_back();

    assert(isfinite(neg_qual) && !isnan(neg_qual));
    if (stamp < tet_time_stamps[tid] || tid >= T.size()) continue;
    // tid >= T size is possible because of shirnkage.
    // or it can be avoided by incrementing stamp[t] that is discarded


    bool improved_flag = false;
    for (auto f:{1, 2, 3})  // all selected face are attached to v0.
      for (auto e:{3 - f, (4 - f) % 3}) {
        std::vector<int> new_tets_id;
        if (igl::dev::tet_tuple_edge_removal(tid, f, e, true, tet_quality,
                                             T, TT, TTif, TTie, new_tets_id)) {
          // put back to heap with additional stamp.
          tet_time_stamps.resize(T.size(), -1);
          for (auto tt:new_tets_id)
            tet_qual_heap.emplace_back(-tet_quality(T[tt][0], T[tt][1],
                                                    T[tt][2], T[tt][3]),
                                       tt, ++tet_time_stamps[tt]);
          improved_flag = true;
          break;
        }
      }
    if (!improved_flag) // don't want to do more. Good luck
      break;
  }
}

inline void face_removal_pass(
    std::function<double(int, int,
                         int, int)> tet_quality,
    std::function<bool(int, int,
                       int, int)> orient3D,
    std::vector<Eigen::RowVector4i> &T,
    std::vector<Eigen::RowVector4i> &TT,
    std::vector<Eigen::RowVector4i> &TTif,
    std::vector<Eigen::Matrix<int, 4, 3>> &TTie) {

  // heap making
  int num_T = T.size();
  std::vector<int> tet_time_stamps(num_T, 0);
  using qual_t = std::tuple<double, size_t, int>;
  std::vector<qual_t> tet_qual_heap(num_T);
  for (int i = 0; i < num_T; i++) {
    auto Ti = T[i];
    tet_qual_heap[i] =
        std::make_tuple(-tet_quality(Ti(0), Ti(1), Ti(2), Ti(3)), i, 0);
  }
  std::make_heap(tet_qual_heap.begin(), tet_qual_heap.end());

  // main loop
  while (true) {
    double neg_qual;
    size_t tid;
    int stamp;
    std::pop_heap(tet_qual_heap.begin(), tet_qual_heap.end());
    std::tie(neg_qual, tid, stamp) = tet_qual_heap.back();
    tet_qual_heap.pop_back();

    assert(isfinite(neg_qual) && !isnan(neg_qual));
    if (stamp < tet_time_stamps[tid] || tid >= T.size()) continue;
    // tid >= T size is possible because of shirnkage.
    // or it can be avoided by incrementing stamp[t] that is discarded


    bool improved_flag = false;
    for (auto f:{0, 1, 2, 3})  // all selected face are attached to v0.
    {
      std::vector<int> new_tets_id;
      if (igl::dev::tet_tuple_multi_face_removal(tid, f, 0, true,
                                                 tet_quality,
                                                 orient3D,
                                                 T,
                                                 TT,
                                                 TTif,
                                                 TTie,
                                                 new_tets_id)) {
        // put back to heap with additional stamp.
        tet_time_stamps.resize(T.size(), -1);
        for (auto tt:new_tets_id)
          tet_qual_heap.emplace_back(-tet_quality(T[tt][0], T[tt][1],
                                                  T[tt][2], T[tt][3]),
                                     tt, ++tet_time_stamps[tt]);
        improved_flag = true;
        break;
      }
    }

    if (!improved_flag) // want to do no more. Good luck
      break;
  }
}

inline void combined_improvement_pass(
    std::function<double(int, int,
                         int, int)> tet_quality,
    std::function<bool(int, int,
                       int, int)> orient3D,
    std::function<bool(int)> vertex_editable,
    std::vector<Eigen::RowVector3d> &V,
    std::vector<Eigen::RowVector4i> &T,
    std::vector<Eigen::RowVector4i> &TT,
    std::vector<Eigen::RowVector4i> &TTif,
    std::vector<Eigen::Matrix<int, 4, 3>> &TTie) {
  // heap making
  int num_T = T.size();
  std::vector<int> tet_time_stamps(num_T, 0);
  using qual_t = std::tuple<double, size_t, int>;
  std::vector<qual_t> tet_qual_heap(num_T);
  for (int i = 0; i < num_T; i++) {
    auto Ti = T[i];
    tet_qual_heap[i] =
        std::make_tuple(-tet_quality(Ti(0), Ti(1), Ti(2), Ti(3)), i, 0);
  }
  std::make_heap(tet_qual_heap.begin(), tet_qual_heap.end());

  // main loop
  while (true) {
    double neg_qual;
    size_t tid;
    int stamp;
    std::pop_heap(tet_qual_heap.begin(), tet_qual_heap.end());
    std::tie(neg_qual, tid, stamp) = tet_qual_heap.back();
    tet_qual_heap.pop_back();

    assert(isfinite(neg_qual) && !isnan(neg_qual));
    if (stamp < tet_time_stamps[tid] || tid >= T.size()) continue;
    // tid >= T size is possible because of shirnkage.
    // or it can be avoided by incrementing stamp[t] that is discarded


    bool improved_flag = false;
    for (auto f:{0, 1, 2, 3})
    {
      std::vector<int> new_tets_id;

      // smoothing pass
      if (igl::dev::laplacian_smart_smoothing(tid,
                                                 f,
                                                 0,
                                                 true,
                                                 tet_quality,
                                                vertex_editable,
                                                T,
                                                 TT,
                                                 TTif,
                                                 TTie,
                                              V,
                                                 new_tets_id)) {
        // put back to heap with additional stamp.
        tet_time_stamps.resize(T.size(), -1);
        for (auto tt:new_tets_id)
          tet_qual_heap.emplace_back(-tet_quality(T[tt][0], T[tt][1],
                                                  T[tt][2], T[tt][3]),
                                     tt, ++tet_time_stamps[tt]);
        improved_flag = true;
        break;
      }

      // multi_face pass
//      if (igl::dev::tet_tuple_multi_face_removal(tid,
//                                                 f,
//                                                 0,
//                                                 true,
//                                                 tet_quality,
//                                                 orient3D,
//                                                 T,
//                                                 TT,
//                                                 TTif,
//                                                 TTie,
//                                                 new_tets_id)) {
//         put back to heap with additional stamp.
//        tet_time_stamps.resize(T.size(), -1);
//        for (auto tt:new_tets_id)
//          tet_qual_heap.emplace_back(-tet_quality(T[tt][0], T[tt][1],
//                                                  T[tt][2], T[tt][3]),
//                                     tt, ++tet_time_stamps[tt]);
//        improved_flag = true;
//        break;
//      }

      // edge removal/contraction/split pass
      if(f!= 0)  // all selected face are attached to v0.
        for (auto e:{3 - f, (4 - f) % 3}) {
          if (igl::dev::tet_tuple_edge_removal(tid, f, e, true, tet_quality,
                                               T, TT, TTif, TTie, new_tets_id)
              || igl::dev::tet_tuple_edge_contraction(tid, f, e, true,
                                                      tet_quality, vertex_editable,
                                                      T, TT, TTif, TTie,
                                                      new_tets_id)
              ||igl::dev::tet_tuple_edge_split(tid, f, e, true,
                                                     tet_quality, V,
                                                     T, TT, TTif, TTie,
                                                     new_tets_id)) {
            // put back to heap with additional stamp.
            tet_time_stamps.resize(T.size(), -1);
            for (auto tt:new_tets_id)
              tet_qual_heap.emplace_back(-tet_quality(T[tt][0], T[tt][1],
                                                      T[tt][2], T[tt][3]),
                                         tt, ++tet_time_stamps[tt]);
            improved_flag = true;
            break;
          }
        }
      if(improved_flag) break;

    }

    if (!improved_flag) // don't want to do more. Good luck
      break;
  }
}
#endif //SCAFFOLD_TEST_TETRAHEDRAL_IMPROVEMENT_H
