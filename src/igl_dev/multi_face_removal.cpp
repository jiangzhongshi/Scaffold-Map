//
// Created by Zhongshi Jiang on 4/11/17.
//

#include <Eigen/Core>
#include <Eigen/StdVector>
#include "tetrahedron_tetrahedron_adjacency.h"
#include "tetrahedron_tuple.h"
#include "retain_tetrahedral_adjacency.h"
#include "edge_removal.h"

#include <iostream>
#include <tuple>
#include <set>
#include <algorithm>
#include <map>
#include <list>
#include <iterator>
namespace igl {
namespace dev {

struct face_removal_neighbor_test {
  template<typename DerivedT, typename DerivedTT,
           typename DerivedTTif, typename DerivedTTie>
  std::tuple<double, double, std::list<int>, std::list<std::tuple<int, int>>>
  recurse(int ti, int fi, int ei, bool ai,
          std::vector<DerivedT> &T,
          std::vector<DerivedTT> &TT,
          std::vector<DerivedTTif> &TTif,
          std::vector<DerivedTTie> &TTie,
          int a, int b,
          const std::function<double(int, int,
                                     int, int)> &tet_quality,
          const std::function<bool(int, int,
                                   int, int)> &orient3D) {
    using namespace igl::dev;

    auto u = tet_tuple_get_vert(ti, fi, ei, ai, T, TT, TTif, TTie);
    auto w = tet_tuple_get_vert(ti, fi, ei, !ai, T, TT, TTif, TTie);

    auto q_uw = tet_quality(a, b, u, w);

    // revolve around the edge : count one-ring tet and decide non-boundary.
    bool flag = true;
    int revolving_face_num = 0;
    std::tuple<int, int, int> other_sandwiched_face;
    int f = fi, t = ti, e = ei;
    do {
      // swtch t/f
      if (tet_tuple_is_on_boundary(t, f, e, true, T, TT, TTif, TTie)) {
        flag = false;
        break;
      }
      std::tie(t, f, e) = std::make_tuple(TT[t][f], TTif[t][f],
                                          TTie[t](f, e));
      f = igl::dev::tetrahedron_local_FF(f, (e + 2) % 3);

      revolving_face_num++;
      if (revolving_face_num == 2) {
        other_sandwiched_face = std::make_tuple(t, f, e);
      }
      if (revolving_face_num > 4) {
        flag = false;
        break;
      }
    } while (t != ti);

    if(revolving_face_num != 4) flag = false;

    if (flag) {
      std::tie(t, f, e) = other_sandwiched_face;
      int v = tet_tuple_get_vert(t, f, (e + 2) % 3, true, T, TT, TTif, TTie);

      auto j_uv = orient3D(a, b, u, v);
      auto j_vw = orient3D(a, b, v, w);
      auto j_wu = orient3D(a, b, w, u);

      double o_uv, n_uv, o_vw, n_vw;

      std::list<int> h_uv, h_vw;
      std::list<std::tuple<int, int>> dt_uv, dt_vw;
      if ((int) j_uv + (int) j_vw + (int) j_wu >= 2) {
        std::tie(o_uv, n_uv, h_uv, dt_uv) =
            recurse(t, f, (e + 1) % 3, false,
                    T, TT, TTif, TTie,
                    a, b, tet_quality, orient3D);
        std::tie(o_vw, n_vw, h_vw, dt_vw) =
            recurse(t, f, (e + 2) % 3, false,
                    T, TT, TTif, TTie,
                    a, b, tet_quality, orient3D);

        auto q_old = std::min({tet_quality(a, u, v, w), tet_quality(u, v, w,
                                                                    b),
                               o_uv, o_vw});
        auto q_new = std::min(n_uv, n_vw);

        if (q_new > q_old || q_new > q_uw) {
//        one_ring_of_removal.insert(current_pos, v);
          h_uv.push_back(v);
          h_uv.splice(h_uv.end(), h_vw);
          dt_uv.emplace_back(t, f);
          dt_uv.splice(dt_uv.end(), dt_vw);
          return std::make_tuple(q_old, q_new, h_uv, dt_uv);
        }
      }
    }

    return std::make_tuple(INFINITY, q_uw, std::list<int>(),
                           std::list<std::tuple<int, int>>());
  };
};

template<typename DerivedT, typename DerivedTT,
         typename DerivedTTif, typename DerivedTTie>
bool tet_tuple_multi_face_removal(int ti,
                                  int fi,
                                  int ei,
                                  bool ai,
                                  std::function<double(int, int,
                                                       int, int)> tet_quality,
                                  std::function<bool(int, int,
                                                     int, int)> orient3D,
                                  std::vector<DerivedT> &T,
                                  std::vector<DerivedTT> &TT,
                                  std::vector<DerivedTTif> &TTif,
                                  std::vector<DerivedTTie> &TTie,
                                  std::vector<int> &new_tets_id) {

  if (igl::dev::tet_tuple_is_on_boundary(ti, fi, ei, ai,
                                         T, TT, TTif, TTie))
    return false;

  int apex_a, apex_b;
  double q_old = tet_quality(T[ti](0), T[ti](1), T[ti](2), T[ti](3));
  double q_new = INFINITY;
  {
    // get apex_a: switch f/e/v
    auto t = ti, f = fi, e = ei;
    auto a = true;
    igl::dev::tet_tuple_switch_face(t, f, e, a, T, TT, TTif, TTie);
    igl::dev::tet_tuple_switch_edge(t, f, e, a, T, TT, TTif, TTie);
    apex_a = igl::dev::tet_tuple_get_vert(t, f, e, !a,
                                          T, TT, TTif, TTie);
  }
  {
    auto t = ti, f = fi, e = ei;
    auto a = true;
    igl::dev::tet_tuple_switch_tet(t, f, e, a, T, TT, TTif, TTie);
    igl::dev::tet_tuple_switch_face(t, f, e, a, T, TT, TTif, TTie);
    igl::dev::tet_tuple_switch_edge(t, f, e, a, T, TT, TTif, TTie);
    apex_b = igl::dev::tet_tuple_get_vert(t, f, e, !a,
                                          T, TT, TTif, TTie);
    q_old = std::min(q_old, tet_quality(T[t](0), T[t](1), T[t](2), T[t](3)));
  }

  assert(q_old > 0);

  face_removal_neighbor_test neighbor_test;
  std::list<int> polygon;
  std::list<std::tuple<int, int>> delete_faces{std::make_tuple(ti, fi)};;
  for (auto e:{0, 1, 2}) {
    std::list<int> added_points;
    decltype(delete_faces) local_del_faces;
    double local_q_old, local_q_new;
    polygon.push_back(igl::dev::tet_tuple_get_vert(ti, fi, e, ai,
                                                   T, TT, TTif, TTie));
    std::tie(local_q_old, local_q_new, added_points, local_del_faces) =
        neighbor_test.recurse(ti, fi, e, ai, T, TT, TTif, TTie, apex_a, apex_b,
                              tet_quality, orient3D);

    q_old = std::min(q_old, local_q_old);
    q_new = std::min(q_new, local_q_new);
    polygon.splice(polygon.end(), added_points);
    delete_faces.splice(delete_faces.end(), local_del_faces);
  }
  if (q_new <= q_old)
    return false;
//  if (q_new < 0) {
//    std::cout<<"Warning q_new < 0 in Multiface removal"<<std::endl;
//    return false;
//  }
  // delete those sandwiched between a/b
  std::set<int> delete_tets;
  std::set<int> influence_id;
  for (auto tf:delete_faces) {
    int t, f;
    std::tie(t, f) = tf;
    delete_tets.insert(t);
    delete_tets.insert(TT[t][f]);
  }

  for (auto t:delete_tets)
    for (auto f:{0, 1, 2, 3}) influence_id.insert(TT[t][f]);
  influence_id.erase(-1);

  std::vector<Eigen::RowVector4i> new_tets;
  auto p = std::rbegin(polygon);
  auto q = std::next(p);
  for (; q != std::rend(polygon); ++p, ++q) {
    new_tets.emplace_back(apex_a, apex_b, *p, *q);
  }
  if (!new_tets.empty())
    new_tets.emplace_back(apex_a, apex_b, *p, *std::rbegin(polygon));

  std::set<int> surround_id;
  std::set_difference(influence_id.begin(), influence_id.end(),
                      delete_tets.begin(), delete_tets.end(),
                      std::inserter(surround_id, surround_id.begin()));


  // additional procedure for the recomputation of new tets.
  // is actually reusable for retain_tet_adj. But not much.
  new_tets_id.clear();
  new_tets_id.insert(new_tets_id.end(), delete_tets.begin(), delete_tets.end());
  for (int i = 0, loop_num = new_tets.size() - delete_tets.size(); i < loop_num;
       i++)
    new_tets_id.push_back(i + T.size());
  if (delete_tets.size() > new_tets.size()) new_tets_id.resize(new_tets.size());

#ifndef NDEBUG
  {
    double qq = INFINITY;
    for (auto r:new_tets) {
      double new_qq = tet_quality(r[0], r[1], r[2], r[3]);
      if (qq > new_qq)
        qq = new_qq;
    }
    assert(qq == q_new);
  }
  assert(q_new > 0);

#endif
  retain_tetrahedral_adjacency(delete_tets, surround_id, new_tets, T, TT, TTif,
                               TTie);
#ifndef NDEBUG
  {
    double qq = INFINITY;
    for (auto i:new_tets_id) {
      auto r = T[i];
      double new_qq = tet_quality(r[0], r[1], r[2], r[3]);
      if (qq > new_qq)
        qq = new_qq;
    }
    assert(qq == q_new);
  }
#endif
  return true;
};

template<typename DerivedT, typename DerivedTT,
         typename DerivedTTif, typename DerivedTTie>
bool tet_tuple_multi_face_removal_force(int ti,
                                        int fi,
                                        int ei,
                                        bool ai,
                                        std::function<double(int,
                                                             int,
                                                             int,
                                                             int)> tet_quality,
                                        std::function<bool(int, int,
                                                           int, int)> orient3D,
                                        std::vector<DerivedT> &T,
                                        std::vector<DerivedTT> &TT,
                                        std::vector<DerivedTTif> &TTif,
                                        std::vector<DerivedTTie> &TTie) {

  if (igl::dev::tet_tuple_is_on_boundary(ti, fi, ei, ai,
                                         T, TT, TTif, TTie))
    return false;

  int apex_a, apex_b;
  double q_old = tet_quality(T[ti](0), T[ti](1), T[ti](2), T[ti](3));
  double q_new = INFINITY;
  {
    // get apex_a: switch f/e/v
    auto t = ti, f = fi, e = ei;
    auto a = true;
    igl::dev::tet_tuple_switch_face(t, f, e, a, T, TT, TTif, TTie);
    igl::dev::tet_tuple_switch_edge(t, f, e, a, T, TT, TTif, TTie);
    apex_a = igl::dev::tet_tuple_get_vert(t, f, e, !a,
                                          T, TT, TTif, TTie);
  }
  {
    auto t = ti, f = fi, e = ei;
    auto a = true;
    igl::dev::tet_tuple_switch_tet(t, f, e, a, T, TT, TTif, TTie);
    igl::dev::tet_tuple_switch_face(t, f, e, a, T, TT, TTif, TTie);
    igl::dev::tet_tuple_switch_edge(t, f, e, a, T, TT, TTif, TTie);
    apex_b = igl::dev::tet_tuple_get_vert(t, f, e, !a,
                                          T, TT, TTif, TTie);
    q_old = std::min(q_old, tet_quality(T[t](0), T[t](1), T[t](2), T[t](3)));
  }

  assert(q_old > 0);

  face_removal_neighbor_test neighbor_test;
  std::list<int> polygon;
  std::list<std::tuple<int, int>> delete_faces{std::make_tuple(ti, fi)};;
  for (auto e:{0, 1, 2}) {
    std::list<int> added_points;
    decltype(delete_faces) local_del_faces;
    double local_q_old, local_q_new;
    polygon.push_back(igl::dev::tet_tuple_get_vert(ti, fi, e, ai,
                                                   T, TT, TTif, TTie));
    std::tie(local_q_old, local_q_new, added_points, local_del_faces) =
        neighbor_test.recurse(ti, fi, e, ai, T, TT, TTif, TTie, apex_a, apex_b,
                              tet_quality, orient3D);

    q_old = std::min(q_old, local_q_old);
    q_new = std::min(q_new, local_q_new);
    polygon.splice(polygon.end(), added_points);
    delete_faces.splice(delete_faces.end(), local_del_faces);
  }
  // delete those sandwiched between a/b
  std::set<int> delete_tets;
  std::set<int> influence_id;
  for (auto tf:delete_faces) {
    int t, f;
    std::tie(t, f) = tf;
    delete_tets.insert(t);
    delete_tets.insert(TT[t][f]);
  }

  for (auto t:delete_tets)
    for (auto f:{0, 1, 2, 3}) influence_id.insert(TT[t][f]);
  influence_id.erase(-1);

  std::vector<Eigen::RowVector4i> new_tets;
  auto p = std::rbegin(polygon);
  auto q = std::next(p);
  for (; q != std::rend(polygon); ++p, ++q) {
    new_tets.emplace_back(apex_a, apex_b, *p, *q);
  }
  if (!new_tets.empty())
    new_tets.emplace_back(apex_a, apex_b, *p, *std::rbegin(polygon));

  std::set<int> surround_id;
  std::set_difference(influence_id.begin(), influence_id.end(),
                      delete_tets.begin(), delete_tets.end(),
                      std::inserter(surround_id, surround_id.begin()));


#ifndef NDEBUG
  {
    double qq = INFINITY;
    for (auto r:new_tets) {
      double new_qq = tet_quality(r[0], r[1], r[2], r[3]);
      if (qq > new_qq)
        qq = new_qq;
    }
    assert(qq == q_new);
  }
  assert(q_new >= 0);

#endif
  retain_tetrahedral_adjacency(delete_tets, surround_id, new_tets, T, TT, TTif,
                               TTie);
  return true;
};

}
}

#ifdef IGL_STATIC_LIBRARY
template bool igl::dev::tet_tuple_multi_face_removal<Eigen::Matrix<int, 1, 4,
                                                                   1, 1, 4>,
                                                     Eigen::Matrix<int,
                                                                   1,
                                                                   4,
                                                                   1,
                                                                   1,
                                                                   4>,
                                                     Eigen::Matrix<int,
                                                                   1,
                                                                   4,
                                                                   1,
                                                                   1,
                                                                   4>,
                                                     Eigen::Matrix<int,
                                                                   4,
                                                                   3,
                                                                   0,
                                                                   4,
                                                                   3> >(int,
                                                                        int,
                                                                        int,
                                                                        bool,
                                                                        std::__1::function<
                                                                            double(
                                                                                int,
                                                                                int,
                                                                                int,
                                                                                int)>,
                                                                        std::__1::function<
                                                                            bool(
                                                                                int,
                                                                                int,
                                                                                int,
                                                                                int)>,
                                                                        std::__1::vector<
                                                                            Eigen::Matrix<
                                                                                int,
                                                                                1,
                                                                                4,
                                                                                1,
                                                                                1,
                                                                                4>,
                                                                            std::__1::allocator<
                                                                                Eigen::Matrix<
                                                                                    int,
                                                                                    1,
                                                                                    4,
                                                                                    1,
                                                                                    1,
                                                                                    4> > > &,
                                                                        std::__1::vector<
                                                                            Eigen::Matrix<
                                                                                int,
                                                                                1,
                                                                                4,
                                                                                1,
                                                                                1,
                                                                                4>,
                                                                            std::__1::allocator<
                                                                                Eigen::Matrix<
                                                                                    int,
                                                                                    1,
                                                                                    4,
                                                                                    1,
                                                                                    1,
                                                                                    4> > > &,
                                                                        std::__1::vector<
                                                                            Eigen::Matrix<
                                                                                int,
                                                                                1,
                                                                                4,
                                                                                1,
                                                                                1,
                                                                                4>,
                                                                            std::__1::allocator<
                                                                                Eigen::Matrix<
                                                                                    int,
                                                                                    1,
                                                                                    4,
                                                                                    1,
                                                                                    1,
                                                                                    4> > > &,
                                                                        std::__1::vector<
                                                                            Eigen::Matrix<
                                                                                int,
                                                                                4,
                                                                                3,
                                                                                0,
                                                                                4,
                                                                                3>,
                                                                            std::__1::allocator<
                                                                                Eigen::Matrix<
                                                                                    int,
                                                                                    4,
                                                                                    3,
                                                                                    0,
                                                                                    4,
                                                                                    3> > > &,
                                                                        std::__1::vector<
                                                                            int,
                                                                            std::__1::allocator<
                                                                                int> > &);

template bool igl::dev::tet_tuple_multi_face_removal_force<Eigen::Matrix<int,
                                                                         1,
                                                                         4,
                                                                         1,
                                                                         1,
                                                                         4>,
                                                           Eigen::Matrix<int,
                                                                         1,
                                                                         4,
                                                                         1,
                                                                         1,
                                                                         4>,
                                                           Eigen::Matrix<int,
                                                                         1,
                                                                         4,
                                                                         1,
                                                                         1,
                                                                         4>,
                                                           Eigen::Matrix<int,
                                                                         4,
                                                                         3,
                                                                         0,
                                                                         4,
                                                                         3> >(
    int,
    int,
    int,
    bool,
    std::__1::function<double(int, int, int, int)>,
    std::__1::function<bool(int, int, int, int)>,
    std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>,
                     std::__1::allocator<Eigen::Matrix<int,
                                                       1,
                                                       4,
                                                       1,
                                                       1,
                                                       4> > > &,
    std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>,
                     std::__1::allocator<Eigen::Matrix<int,
                                                       1,
                                                       4,
                                                       1,
                                                       1,
                                                       4> > > &,
    std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>,
                     std::__1::allocator<Eigen::Matrix<int,
                                                       1,
                                                       4,
                                                       1,
                                                       1,
                                                       4> > > &,
    std::__1::vector<Eigen::Matrix<int, 4, 3, 0, 4, 3>,
                     std::__1::allocator<Eigen::Matrix<int,
                                                       4,
                                                       3,
                                                       0,
                                                       4,
                                                       3> > > &);

#endif
