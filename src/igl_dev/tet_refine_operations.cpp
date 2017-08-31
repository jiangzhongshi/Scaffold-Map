//
// Created by Zhongshi Jiang on 5/2/17.
//

#include "tet_refine_operations.h"
#include "tetrahedron_tuple.h"
#include "retain_tetrahedral_adjacency.h"
#include "tetrahedron_tetrahedron_adjacency.h"
#include <iostream>

namespace igl{ namespace dev {
bool tet_tuple_edge_contraction(int ti,
                                int fi,
                                int ei,
                                bool along,
                                std::function<double(int,
                                                     int,
                                                     int,
                                                     int)> tet_quality,
                                std::function<bool(int)> vertex_editable,
                                std::vector<Eigen::RowVector4i> &T,
                                std::vector<Eigen::RowVector4i> &TT,
                                std::vector<Eigen::RowVector4i> &TTif,
                                std::vector<Eigen::Matrix<int, 4, 3>> &TTie,
                                std::vector<int> &new_tets_id) {



  auto v1 = igl::dev::tet_tuple_get_vert(ti, fi, ei, !along,
                                         T, TT, TTif, TTie);
  int v0 = -1;
  bool v0_v1_can_exchange = true; // the ending point can either be 0 or 1
  if(!vertex_editable(v1)) {
    v0 = v1;
    along = !along;
    v1 = igl::dev::tet_tuple_get_vert(ti, fi, ei, !along,
                                      T, TT, TTif, TTie);
    if(!vertex_editable(v1))
      return false;
    v0_v1_can_exchange = false; // v1 not editable, v0 yes.
  } else {
    v0 = igl::dev::tet_tuple_get_vert(ti, fi, ei, along,
                                      T, TT, TTif, TTie);
  }

  auto T0 = igl::dev::tet_tuple_get_tets_with_vert(ti, fi, ei, along,
                                                   T, TT, TTif, TTie);
  auto T1 = igl::dev::tet_tuple_get_tets_with_vert(ti, fi, ei, !along,
                                                   T, TT, TTif, TTie);

  std::set<int> T_inter;
  std::set_intersection(T0.begin(), T0.end(), T1.begin(), T1.end(),
                        std::inserter(T_inter, T_inter.begin()));


  std::set<int> T_1_minus_0;
  std::set_difference(T1.begin(), T1.end(), T_inter.begin(), T_inter.end(),
                      std::inserter(T_1_minus_0, T_1_minus_0.begin()));
  std::set<int> T_0_minus_1;
  std::set_difference(T0.begin(), T0.end(), T1.begin(), T1.end(),
                      std::inserter(T_0_minus_1, T_0_minus_1.begin()));


  std::vector<Eigen::RowVector4i> new_tets0(T_1_minus_0.size());

  int i = 0;
  for (auto &s:T_1_minus_0) {
    for (int j :{0, 1, 2, 3}) {
      if (T[s][j] == v1)
        new_tets0[i][j] = v0;
      else
        new_tets0[i][j] = T[s][j];
    }
    i++;
  }

  double min_q0 = INFINITY;
  for (auto t0:new_tets0) {
    min_q0 = std::min(min_q0, tet_quality(t0[0], t0[1], t0[2], t0[3]));
  }

  double old_q = INFINITY;
  for (auto s:T1) {
    auto t0 = T[s];
    old_q = std::min(old_q, tet_quality(t0[0], t0[1], t0[2], t0[3]));
  }

  if (old_q > min_q0) {
    if(!v0_v1_can_exchange) return false;
    decltype(new_tets0) new_tets1(T_0_minus_1.size());
    int i = 0;
    for (auto &s:T_0_minus_1) {
      for (int j :{0, 1, 2, 3}) {
        if (T[s][j] == v1)
          new_tets1[i][j] = v0;
        else
          new_tets1[i][j] = T[s][j];
      }
      i++;
    }

//    auto new_tets1 = new_tets0;
//    for (auto &s:new_tets1)
//      for (int j:{0, 1, 2, 3}) {
//        if (s(j) == v0)
//          s(j) = v1;
//      }
    double min_q1 = INFINITY;
    for (auto t0:new_tets1) {
      min_q1 = std::min(min_q1, tet_quality(t0[0], t0[1], t0[2], t0[3]));
    }
    double old_q = INFINITY;
    for (auto s:T1) {
      auto t0 = T[s];
      old_q = std::min(old_q, tet_quality(t0[0], t0[1], t0[2], t0[3]));
    }
    if (old_q > min_q1) return false;

    // additional procedure for the recomputation of new tets.
    // is actually reusable for retain_tet_adj. But not much.
    auto& delete_tets = T0;
    auto& new_tets = new_tets1;
    new_tets_id.clear();
    new_tets_id.insert(new_tets_id.end(), delete_tets.begin(), delete_tets.end());
    for (int i = 0, loop_num = new_tets.size() - delete_tets.size(); i < loop_num;
         i++)
      new_tets_id.push_back(i + T.size());
    if (delete_tets.size() > new_tets.size()) new_tets_id.resize(new_tets.size());

    // delete v0
    std::set<int> influence_id;
    for (auto t:delete_tets)
      for (auto f:{0, 1, 2, 3})
        influence_id.insert(TT[t][f]);
    influence_id.erase(-1);

    // retain_connectivity.
    std::set<int> surround_id;
    std::set_difference(influence_id.begin(), influence_id.end(),
                        delete_tets.begin(), delete_tets.end(),
                        std::inserter(surround_id, surround_id.end()));

    retain_tetrahedral_adjacency(T0, surround_id, new_tets1, T, TT, TTif, TTie);
  } else {


    auto& delete_tets = T1;
    auto& new_tets = new_tets0;
    new_tets_id.clear();
    new_tets_id.insert(new_tets_id.end(), delete_tets.begin(), delete_tets.end());
    for (int i = 0, loop_num = new_tets.size() - delete_tets.size(); i < loop_num;
         i++)
      new_tets_id.push_back(i + T.size());
    if (delete_tets.size() > new_tets.size()) new_tets_id.resize(new_tets.size());

    std::set<int> influence_id;
    for (auto t:delete_tets)
      for (auto f:{0, 1, 2, 3})
        influence_id.insert(TT[t][f]);
    influence_id.erase(-1);

    // retain_connectivity.
    std::set<int> surround_id;
    std::set_difference(influence_id.begin(), influence_id.end(),
                        delete_tets.begin(), delete_tets.end(),
                        std::inserter(surround_id, surround_id.end()));

    retain_tetrahedral_adjacency(T1, surround_id, new_tets0, T, TT, TTif, TTie);
  }

//  std::cout << "Warning: Unused Vertex" << std::endl;
  return true;
}

// Sorry for the smart again, but this is from DSC
bool laplacian_smart_smoothing(
    int ti, int fi, int ei, bool along,
    std::function<double(int,
                         int,
                         int,
                         int)> tet_quality,
    std::function<bool(int)> vertex_editable,
    const std::vector <Eigen::RowVector4i> &T,
    const std::vector <Eigen::RowVector4i> &TT,
    const std::vector <Eigen::RowVector4i> &TTif,
    const std::vector <Eigen::Matrix<int, 4, 3>> &TTie,
    std::vector <Eigen::RowVector3d> &V,
    std::vector<int> &new_tets_id) {
  auto v0 = igl::dev::tet_tuple_get_vert(ti, fi, ei, true,
                                         T, TT, TTif, TTie);
  if(!vertex_editable(v0)) return false;

  auto T_neighbor = igl::dev::tet_tuple_get_tets_with_vert(ti, fi, ei, true,
                                                           T, TT, TTif, TTie);
  std::set<int> adj_verts;

  double old_q = INFINITY;
  for (auto s:T_neighbor) {
    auto t0 = T[s];
    old_q = std::min(old_q, tet_quality(t0[0], t0[1], t0[2], t0[3]));
    for (auto j:{0, 1, 2, 3})adj_verts.insert(t0[j]);
  }
  adj_verts.erase(v0);
  Eigen::RowVector3d barycenter(0, 0, 0);
  for (auto s:adj_verts) {
    barycenter += V[s];
  }
  barycenter /= adj_verts.size();
  auto saved = V[v0];
  V[v0] = barycenter;

  double new_q = INFINITY;
  for (auto s:T_neighbor) {
    auto t0 = T[s];
    new_q = std::min(old_q, tet_quality(t0[0], t0[1], t0[2], t0[3]));
  }
  if (new_q <= old_q) {
    V[v0] = saved;
    return false;
  }

  new_tets_id = std::vector<int>(T_neighbor.begin(), T_neighbor.end());
  return true;
}

bool tet_tuple_edge_split(
    int ti, int fi, int ei, bool ai,
    std::function<double(int,
                         int,
                         int,
                         int)> tet_quality,
    std::vector <Eigen::RowVector3d> &V,
    std::vector <Eigen::RowVector4i> &T,
    std::vector <Eigen::RowVector4i> &TT,
    std::vector <Eigen::RowVector4i> &TTif,
    std::vector <Eigen::Matrix<int, 4, 3>> &TTie,
    std::vector<int> &new_tets_id) {
  ai = true;
  int vert_a = igl::dev::tet_tuple_get_vert(ti, fi, ei, ai, T, TT, TTif, TTie);
  int vert_b = igl::dev::tet_tuple_get_vert(ti, fi, ei, !ai, T, TT, TTif, TTie);

  // get the edge pointing to another vertex.
  // swtich e/f
  igl::dev::tet_tuple_switch_edge(ti, fi, ei, ai, T, TT, TTif, TTie);
  igl::dev::tet_tuple_switch_face(ti, fi, ei, ai, T, TT, TTif, TTie);

  // loop through the one ring of tets around an edge.
  int f = fi, t = ti, e = ei;
  std::set<int> one_ring_tets;

  std::vector<int> one_ring_verts;
  double min_qual_old = INFINITY;
  do {
    one_ring_tets.insert(t);
    min_qual_old = std::min(min_qual_old, tet_quality(T[t](0), T[t](1),
                                                      T[t](2), T[t](3)));
    one_ring_verts.push_back(igl::dev::tet_tuple_get_vert(t, f, e, false, T, TT,
                                                          TTif, TTie));
    // swtch f/t/f/e
    f = igl::dev::tetrahedron_local_FF(f, (e + 2) % 3);
    std::tie(t, f, e) = std::make_tuple(TT[t][f], TTif[t][f],
                                        TTie[t](f, e));
    if (t == -1) return false;  // not an inner edge
    f = igl::dev::tetrahedron_local_FF(f, (e + 2) % 3);
    e = (e + (1)) % 3;
  } while (t != ti);

  // attempt addition
  V.push_back((V[vert_a] + V[vert_b]) / 2);
  int vert_m = V.size() - 1;

  std::vector <Eigen::RowVector4i> new_tets;
  new_tets.reserve(2 * one_ring_tets.size());
  for (auto old_v :{vert_a, vert_b})
    for (auto s:one_ring_tets) {
      auto ts = T[s];
      for (auto j:{0, 1, 2, 3})
        if (ts[j] == old_v) {
          ts[j] = vert_m;
          break;
        }
      new_tets.push_back(ts);
    }

  double min_qual_new = INFINITY;
  for (auto s:new_tets)
    min_qual_new = std::min(min_qual_new,
                            tet_quality(s[0], s[1], s[2], s[3]));

  // smart decision
  if (min_qual_new <= min_qual_old) {
    V.pop_back();
    return false;
  }

  std::set<int> influence_id;
  for (auto t:one_ring_tets)
    for (auto f:{0, 1, 2, 3})
      influence_id.insert(TT[t][f]);
  influence_id.erase(-1);

  auto& delete_tets = one_ring_tets;
  new_tets_id.clear();
  new_tets_id.insert(new_tets_id.end(), delete_tets.begin(), delete_tets.end());
  for (int i = 0, loop_num = new_tets.size() - delete_tets.size(); i < loop_num;
       i++)
    new_tets_id.push_back(i + T.size());
  if (delete_tets.size() > new_tets.size()) new_tets_id.resize(new_tets.size());

  // retain_connectivity.
  std::set<int> surround_id;
  std::set_difference(influence_id.begin(), influence_id.end(),
                      one_ring_tets.begin(), one_ring_tets.end(),
                      std::inserter(surround_id, surround_id.end()));
  retain_tetrahedral_adjacency(one_ring_tets, surround_id, new_tets, T, TT,
                               TTif, TTie);

  return true;

}
}}