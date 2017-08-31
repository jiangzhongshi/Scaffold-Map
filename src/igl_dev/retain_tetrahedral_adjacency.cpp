//
// Created by Zhongshi Jiang on 4/3/17.
//
#include "retain_tetrahedral_adjacency.h"
#include "tetrahedron_tetrahedron_adjacency.h"

#include <array>
#include <set>


IGL_INLINE void retain_tetrahedral_adjacency(
    const std::set<int>& delete_id, const std::set<int>& surround_id,
    const std::vector<Eigen::RowVector4i>& new_T,
    std::vector<Eigen::RowVector4i>& T,
    std::vector<Eigen::RowVector4i>& TT,
    std::vector<Eigen::RowVector4i>& TTif,
    std::vector<Eigen::Matrix<int, 4, 3>>& TTie) {

  using namespace Eigen;
  using namespace std;
  using vecV2d = std::vector<Eigen::RowVector2d>;
//                           Eigen::aligned_allocator<Eigen::RowVector2d>>;
  using vecV3d = std::vector<Eigen::RowVector3d>;
  using vecV3i = std::vector<Eigen::RowVector3i>;
  using vecV4i = std::vector<Eigen::RowVector4i>;
//                           Eigen::aligned_allocator<Eigen::RowVector4i>>;
  using T_t = vecV4i;
  using TT_t = T_t;
  using TTie_t = std::vector<Eigen::Matrix<int, 4, 3>>;

  // input
  T_t new_tets = new_T;
  TT_t new_TT, new_TTif;
  TTie_t new_TTie;

  vector<int> new_tet_map;  // map the local indices to global
  vector<int> voided_tets_id; // only for num_addition < 0
  const int num_new = (int) new_T.size();
  const int num_influence = (int) surround_id.size();
  const int num_addition = int(num_new - delete_id.size());
  new_tet_map.reserve(num_new + num_influence);
  new_tet_map.insert(new_tet_map.end(), delete_id.begin(), delete_id.end());
  for (int i = 0; i < num_addition; i++)
    new_tet_map.push_back(i + T.size());
  if (num_addition < 0) {
    voided_tets_id.resize(-num_addition);
    for (int i = 0; i < -num_addition; i++)
      voided_tets_id[i] = (new_tet_map[i + num_new]);
    new_tet_map.resize(num_new);
  }
  new_tet_map.insert(new_tet_map.end(), surround_id.begin(), surround_id.end());

  // processing
  new_tets.resize(num_new + num_influence);
  for (int i = num_new; i < num_new + num_influence; i++) {
    new_tets[i] = T[new_tet_map[i]];
  }

  igl::dev::tetrahedron_tetrahedron_adjacency(new_tets, new_TT,
                                              new_TTif, new_TTie);
  // only t id are confused within the new_*'s
  // recover
  int old_size = T.size();
  if (num_addition > 0) {
    T.resize(old_size + num_addition);
    TT.resize(old_size + num_addition);
    TTif.resize(old_size + num_addition);
    TTie.resize(old_size + num_addition);
  }
  for (int i = old_size; i < old_size + num_addition; i++) {
    T[i] = -RowVector4i::Ones();
    TT[i] = -RowVector4i::Ones();
    TTif[i] = -RowVector4i::Ones();
    TTie[i] = -Matrix<int, 4, 3>::Ones();
  }

  // Bug fix snippet: nullify pointers to deleted tets.
  for(auto ds:surround_id) {
    for(auto j:{0,1,2,3}) {
      auto& adj_tet = TT[ds][j];
      if (delete_id.find(adj_tet) == delete_id.end()) continue;
      TT[ds][j] = -1;
      TTif[ds][j] = -1;
      TTie[ds].row(j) << -1,-1,-1;
    }
  }

  for (int i = 0; i < num_new; i++) {
    T[new_tet_map[i]] = new_tets[i];
    TTif[new_tet_map[i]] = new_TTif[i];
    TTie[new_tet_map[i]] = new_TTie[i];

    for (int j = 0; j < 4; j++) { // TT needs care because of t id
      auto adj_tet = new_TT[i][j];
      if (adj_tet == -1)
        TT[new_tet_map[i]][j] = -1;
      else
        TT[new_tet_map[i]][j] = new_tet_map[adj_tet];
    }
  }
  for (int i = num_new; i < num_new + num_influence; i++)
    for (int j = 0; j < 4; j++) {
      int adj_tet = new_TT[i][j];
      if (adj_tet == -1) continue;
      // got // some // valid // information
      TT[new_tet_map[i]][j] = new_tet_map[adj_tet];
      TTif[new_tet_map[i]][j] = new_TTif[i][j];
      TTie[new_tet_map[i]].row(j) = new_TTie[i].row(j);
    } // for, for

  if (num_addition < 0) {
    // take special care here.
    // fix ending's influence
    std::set<int> voided_set(voided_tets_id.begin(), voided_tets_id.end());
    std::set<int> ending_set;
    std::set<int> symmetric_diff;
    for(int i=0; i<-num_addition; i++)
      ending_set.insert(ending_set.end(), old_size + num_addition + i);
    std::set_symmetric_difference(voided_set.begin(), voided_set.end(),
                                  ending_set.begin(), ending_set.end(),
                                  std::inserter(symmetric_diff,
                                                symmetric_diff.end() ));
    int num_substituting = symmetric_diff.size()/2;
    auto ending_iter = symmetric_diff.rbegin();
    auto voided_iter = symmetric_diff.begin();
    for (int i = 0; i < num_substituting; i++) {
      int t = *ending_iter;
      int vi = *voided_iter;
      for (auto f:{0, 1, 2, 3}) {
        if (TT[t][f] == -1) continue;
        assert(TT[TT[t][f]][TTif[t][f]] == t);
        TT[TT[t][f]][TTif[t][f]] = vi;
      }
      TT[vi] = TT[t];
      TTif[vi] = TTif[t];

      ending_iter ++;
      voided_iter ++;
    }

    ending_iter = symmetric_diff.rbegin();
    voided_iter = symmetric_diff.begin();
    // assign back
    for (int i = 0; i < num_substituting; i++) {
      int t = *ending_iter;
      int vi = *voided_iter;
      T[vi] = T[t];
      TTie[vi] = TTie[t];

      ending_iter ++;
      voided_iter ++;
    }

    T.resize(old_size + num_addition);
    TT.resize(old_size + num_addition);
    TTif.resize(old_size + num_addition);
    TTie.resize(old_size + num_addition);

  }

#ifndef NDEBUG
  std::vector<Eigen::RowVector4i> dTT;
  std::vector<Eigen::RowVector4i> dTTif;
  std::vector<Eigen::Matrix<int, 4, 3>> dTTie;
  igl::dev::tetrahedron_tetrahedron_adjacency(T, dTT, dTTif, dTTie);
  for (int i = 0; i < T.size(); i++) {
    assert(dTT[i] == TT[i]);
    assert(dTTif[i] == TTif[i]);
    assert(dTTie[i] == TTie[i]);
  }
#endif

}


#ifdef IGL_STATIC_LIBRARY
//template void retain_tetrahedral_adjacency<std::__1::set<int,
//                                                       std::__1::less<int>,
//                                                 std::__1::allocator<int> >, std::__1::vector<int, std::__1::allocator<int> > >(std::__1::set<int, std::__1::less<int>, std::__1::allocator<int> > const&, std::__1::vector<int, std::__1::allocator<int> > const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > >&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > >&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > >&, std::__1::vector<Eigen::Matrix<int, 4, 3, 0, 4, 3>, std::__1::allocator<Eigen::Matrix<int, 4, 3, 0, 4, 3> > >&);
//template void retain_tetrahedral_adjacency<std::__1::vector<int,
//                                                   std::__1::allocator<int> >, std::__1::vector<int, std::__1::allocator<int> > >(std::__1::vector<int, std::__1::allocator<int> > const&, std::__1::vector<int, std::__1::allocator<int> > const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > >&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > >&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > >&, std::__1::vector<Eigen::Matrix<int, 4, 3, 0, 4, 3>, std::__1::allocator<Eigen::Matrix<int, 4, 3, 0, 4, 3> > >&);
//
#endif