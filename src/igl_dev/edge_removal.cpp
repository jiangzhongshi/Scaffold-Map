//
// Created by Zhongshi Jiang on 4/5/17.
//
#include "edge_removal.h"
#include "retain_tetrahedral_adjacency.h"
#include "tetrahedron_tetrahedron_adjacency.h"
#include "tetrahedron_tuple.h"
#include <set>
#include <algorithm>

namespace igl {
namespace dev {
template<typename QFunc>
inline double optimal_triangulation_Klincsek(std::vector<int> polygon,
                                             QFunc quality,
                                             std::vector<Eigen::RowVector3i>&
                                             new_tri) {
  using namespace Eigen;
  using namespace std;
  const int m = static_cast<int>(polygon.size());

  // build table
  // https://github.com/janba/DSC/blob/master/src/DSC.cpp#L208
  MatrixXd Q_table = MatrixXd::Constant(m - 1, m, INFINITY);
  static MatrixXi K_table;
  K_table = MatrixXi::Constant(m - 1, m, -1);
  for (auto i = m - 3; i >= 0; i--)
    for (auto j = i + 2; j < m; j++)
      for (auto k = i + 1; k < j; k++) {
        auto q = quality(polygon[i], polygon[j], polygon[k]);
        if (k < j - 1) q = std::min(q, Q_table(k, j));
        if (k > i + 1) q = std::min(q, Q_table(i, k));
        if (k == i + 1 || q > Q_table(i, j)) {
          Q_table(i, j) = q;
          K_table(i, j) = k;
        }
      }

  // recursively extract the triangulation.
  struct recurse_extract {
    static void op(int i, int j, const std::vector<int> &poly,
                   std::vector<Eigen::RowVector3i> &tri) {
      if (j >= i + 2) {
        int k = K_table(i, j);
        recurse_extract::op(i, k, poly, tri);
        recurse_extract::op(k, j, poly, tri);
        tri.emplace_back(poly[i], poly[j], poly[k]);
      }
    };
  };
  new_tri.clear();
  recurse_extract::op(0, m - 1, polygon, new_tri);

//  optimal_triangulation = Eigen::Map<Eigen::Matrix<int,-1,-1,Eigen::RowMajor>>
//  (&new_tri[0],new_tri.size(),3);
//  optimal_triangulation.resize(new_tri.size(), 3);
//  for (int i = 0; i < new_tri.size(); i++)
//    optimal_triangulation.row(i) = new_tri[i];

  return Q_table(0, m - 1);
}
}
}

template<typename DerivedT, typename DerivedTT,
         typename DerivedTTif, typename DerivedTTie>
IGL_INLINE bool igl::dev::tet_tuple_edge_removal(int ti,
                                                 int fi,
                                                 int ei,
                                                 bool ai,
                                                 std::function<double(int,
                                                                      int,
                                                                      int,
                                                                      int)> tet_quality,
                                                 std::vector<DerivedT> &T,
                                                 std::vector<DerivedTT> &TT,
                                                 std::vector<DerivedTTif> &TTif,
                                                 std::vector<DerivedTTie> &TTie,
                                                 std::vector<int> new_tets_id) {

  int vert_a = igl::dev::tet_tuple_get_vert(ti, fi, ei, ai, T, TT, TTif, TTie);
  int vert_b = igl::dev::tet_tuple_get_vert(ti, fi, ei, !ai, T, TT, TTif, TTie);
  // get the edge pointing to another vertex.
  // swtich e/f
    ai = true;
    igl::dev::tet_tuple_switch_edge(ti,fi,ei,ai,T,TT,TTif,TTie);
    igl::dev::tet_tuple_switch_face(ti,fi,ei,ai,T,TT,TTif,TTie);
//  ai = true;
//  ei = (ei + 2) % 3;
//  fi = igl::dev::tetrahedron_local_FF(fi, (ei + 2) % 3);

  // loop through the one ring.
  int f = fi, t = ti, e = ei;
  std::set<int> one_ring_tets;
  std::set<int> influence_id;
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
    if(t == -1) return false;
    f = igl::dev::tetrahedron_local_FF(f, (e + 2) % 3);
    e = (e + (1)) % 3;
  } while (t != ti);
  assert(f == fi && e == ei);
  assert(one_ring_tets.size() == one_ring_verts.size());

  // get triangulation of the new face
  std::vector<Eigen::RowVector3i> optimal_tri;
  if (min_qual_old >
      optimal_triangulation_Klincsek(one_ring_verts, [&](int u, int v, int
                                     w) {
                                       return std::min(tet_quality(vert_a, u, v, w),
                                                       tet_quality(u, v, w, vert_b));
                                     },
                                     optimal_tri))
    return false;
  // connect them all and store in new_tetrahedra
  std::vector<Eigen::RowVector4i> new_connected_tets(2 * optimal_tri.size());
  for (int i = 0; i < optimal_tri.size(); i++) {
    new_connected_tets[2 * i] << vert_a, optimal_tri[i];
    new_connected_tets[2 * i + 1] << optimal_tri[i], vert_b;
  }

  for (auto t:one_ring_tets)
    for (auto f:{0, 1, 2, 3})
      influence_id.insert(TT[t][f]);
  influence_id.erase(-1);

  // retain_connectivity.
  std::set<int> surround_id;
  std::set_difference(influence_id.begin(), influence_id.end(),
                      one_ring_tets.begin(), one_ring_tets.end(),
                      std::inserter(surround_id, surround_id.end()));


  // additional procedure for the recomputation of new tets.
  // is actually reusable for retain_tet_adj. But not much.
  new_tets_id.clear();
  new_tets_id.insert(new_tets_id.end(), one_ring_tets.begin(), one_ring_tets
      .end());
//  for (int i = 0; i < new_connected_tets.size() - one_ring_tets.size(); i++)
//    new_tets_id.push_back(i + T.size());
  for (int i = 0, loop_num = new_connected_tets.size() - one_ring_tets.size();
       i <
      loop_num;
       i++)
    new_tets_id.push_back(i + T.size());

  retain_tetrahedral_adjacency(one_ring_tets, surround_id,
                               new_connected_tets, T, TT, TTif, TTie);

  return true;
}

template<typename DerivedT, typename DerivedTT,
         typename DerivedTTif, typename DerivedTTie>
IGL_INLINE bool igl::dev::tet_tuple_edge_removal(int ti,
                                                 int fi,
                                                 int ei,
                                                 bool ai,
                                                 std::function<double(int,
                                                                      int,
                                                                      int,
                                                                      int)> tet_quality,
                                                 std::vector<DerivedT> &T,
                                                 std::vector<DerivedTT> &TT,
                                                 std::vector<DerivedTTif> &TTif,
                                                 std::vector<DerivedTTie>
                                                 &TTie) {
  std::vector<int> dummy;
  return tet_tuple_edge_removal(ti,fi,ei,ai,tet_quality,T,TT,TTif,TTie,dummy);
};

template<typename DerivedT, typename DerivedTT,
         typename DerivedTTif, typename DerivedTTie>
IGL_INLINE bool igl::dev::tet_tuple_edge_removal_force(int ti,
                                             int fi,
                                             int ei,
                                             bool ai,
                                             std::function<double(int,
                                                                  int,
                                                                  int,
                                                                  int)> tet_quality,
                                             std::vector<DerivedT> &T,
                                             std::vector<DerivedTT> &TT,
                                             std::vector<DerivedTTif> &TTif,
                                             std::vector<DerivedTTie>
                                             &TTie) {

  int vert_a = igl::dev::tet_tuple_get_vert(ti, fi, ei, ai, T, TT, TTif, TTie);
  int vert_b = igl::dev::tet_tuple_get_vert(ti, fi, ei, !ai, T, TT, TTif, TTie);
  // get the opposite edge within the tet with appropriate direction:
  // swtich e/f/e/f
  ai = true;
  ei = (ei + 2) % 3;
  fi = igl::dev::tetrahedron_local_FF(fi, (ei + 2) % 3);

  // loop through the one ring.
  int f = fi, t = ti, e = ei;
  std::set<int> one_ring_tets;
  std::set<int> influence_id;
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
    if (t == -1) return false;
    f = igl::dev::tetrahedron_local_FF(f, (e + 2) % 3);
    e = (e + (1)) % 3;
  } while (t != ti);
  assert(f == fi && e == ei);
  assert(one_ring_tets.size() == one_ring_verts.size());

  // get triangulation of the new face
  std::vector<Eigen::RowVector3i> optimal_tri;
  optimal_triangulation_Klincsek(one_ring_verts, [&](int u, int v, int
                                 w) {
                                   return std::min(tet_quality(vert_a, u, v, w),
                                                   tet_quality(u, v, w, vert_b));
                                 },
                                 optimal_tri);
  // connect them all and store in new_tetrahedra
  std::vector<Eigen::RowVector4i> new_connected_tets(2 * optimal_tri.size());
  for (int i = 0; i < optimal_tri.size(); i++) {
    new_connected_tets[2 * i] << vert_a, optimal_tri[i];
    new_connected_tets[2 * i + 1] << optimal_tri[i], vert_b;
  }

  for (auto t:one_ring_tets)
    for (auto f:{0, 1, 2, 3})
      influence_id.insert(TT[t][f]);
  influence_id.erase(-1);

  // retain_connectivity.
  std::set<int> surround_id;
  std::set_difference(influence_id.begin(), influence_id.end(),
                      one_ring_tets.begin(), one_ring_tets.end(),
                      std::inserter(surround_id, surround_id.end()));

  retain_tetrahedral_adjacency(one_ring_tets, surround_id,
                               new_connected_tets, T, TT, TTif, TTie);

  return true;
}

#ifdef IGL_STATIC_LIBRARY
template bool igl::dev::tet_tuple_edge_removal<Eigen::Matrix<int, 1, 4, 1, 1,
4>, Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 1, 4, 1, 1, 4>,
Eigen::Matrix<int, 4, 3, 0, 4, 3> >(int, int, int, bool, std::__1::function<double (int, int, int, int)>, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > >&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > >&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > >&, std::__1::vector<Eigen::Matrix<int, 4, 3, 0, 4, 3>, std::__1::allocator<Eigen::Matrix<int, 4, 3, 0, 4, 3> > >&);
template bool igl::dev::tet_tuple_edge_removal<Eigen::Matrix<int, 1, 4, 1, 1,
 4>, Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 4, 3, 0, 4, 3> >(int, int, int, bool, std::__1::function<double (int, int, int, int)>, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > >&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > >&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > >&, std::__1::vector<Eigen::Matrix<int, 4, 3, 0, 4, 3>, std::__1::allocator<Eigen::Matrix<int, 4, 3, 0, 4, 3> > >&, std::__1::vector<int, std::__1::allocator<int> >);

template bool igl::dev::tet_tuple_edge_removal_force<Eigen::Matrix<int, 1, 4,
                                                                   1, 1, 4>, Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 4, 3, 0, 4, 3> >(int, int, int, bool, std::__1::function<double (int, int, int, int)>, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > >&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > >&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > >&, std::__1::vector<Eigen::Matrix<int, 4, 3, 0, 4, 3>, std::__1::allocator<Eigen::Matrix<int, 4, 3, 0, 4, 3> > >&);
#endif