// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2017 Zhongshi Jiang <zhongshi@cims.nyu.edu>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "tetrahedron_tuple.h"
#include "tetrahedron_tetrahedron_adjacency.h"
#include <tuple>

namespace igl {namespace dev {
template<typename DerivedT, typename DerivedTT,
         typename DerivedTTif, typename DerivedTTie>
IGL_INLINE void tet_tuple_switch_vert(
    int &ti, int &fi, int &ei, bool &along,
    const std::vector<DerivedT> &T,
    const std::vector<DerivedTT> &TT,
    const std::vector<DerivedTTif> &TTif,
    const std::vector<DerivedTTie> &TTie) {
  along = !along;
};

template<typename DerivedT, typename DerivedTT,
         typename DerivedTTif, typename DerivedTTie>
IGL_INLINE void tet_tuple_switch_edge(
    int &ti, int &fi, int &ei, bool &along,
    const std::vector<DerivedT> &T,
    const std::vector<DerivedTT> &TT,
    const std::vector<DerivedTTif> &TTif,
    const std::vector<DerivedTTie> &TTie) {
  ei = (ei + (along ? 2 : 1)) % 3;
  tet_tuple_switch_vert(ti, fi, ei, along, T, TT, TTif, TTie);
};

template<typename DerivedT, typename DerivedTT,
         typename DerivedTTif, typename DerivedTTie>
IGL_INLINE void tet_tuple_switch_face(
    int &ti, int &fi, int &ei, bool &along,
    const std::vector<DerivedT> &T,
    const std::vector<DerivedTT> &TT,
    const std::vector<DerivedTTif> &TTif,
    const std::vector<DerivedTTie> &TTie) {
  fi = tetrahedron_local_FF(fi, (ei + 2) % 3);
  tet_tuple_switch_vert(ti, fi, ei, along, T, TT, TTif, TTie);
};

template<typename DerivedT, typename DerivedTT,
         typename DerivedTTif, typename DerivedTTie>
IGL_INLINE void tet_tuple_switch_tet(
    int &ti, int &fi, int &ei, bool &along,
    const std::vector<DerivedT> &T,
    const std::vector<DerivedTT> &TT,
    const std::vector<DerivedTTif> &TTif,
    const std::vector<DerivedTTie> &TTie) {
  if (tet_tuple_is_on_boundary(ti, fi, ei, along, T, TT, TTif, TTie)) return;
  int tin = TT[ti][fi];
  int fin = TTif[ti][fi];
  int ein = TTie[ti](fi, ei);

  ti = tin;
  fi = fin;
  ei = ein;
  tet_tuple_switch_vert(ti, fi, ei, along, T, TT, TTif, TTie);
};

template<typename DerivedT, typename DerivedTT,
         typename DerivedTTif, typename DerivedTTie>
IGL_INLINE int tet_tuple_get_vert(
    const int &ti, const int &fi, const int &ei, const bool &along,
    const std::vector<DerivedT> &T,
    const std::vector<DerivedTT> &TT,
    const std::vector<DerivedTTif> &TTif,
    const std::vector<DerivedTTie> &TTie) {
  assert(ti >= 0);
  assert(ti < T.size());
  assert(fi <= 3);
  assert(fi >= 0);
  assert(ei >= 0);
  assert(ei <= 2);

  // legacy edge indexing
  return T[ti]
           [(tetrahedron_local_FF)(fi,
                                  along ? ei : (ei + 1) % 3)];
};

template<typename DerivedT,
         typename DerivedTT,
         typename DerivedTTif,
         typename DerivedTTie>
IGL_INLINE int tet_tuple_get_edge(
    const int &ti, const int &fi, const int &ei, const bool &along,
    const std::vector<DerivedT> &T,
    const std::vector<DerivedTT> &TT,
    const std::vector<DerivedTTif> &TTif,
    const std::vector<DerivedTTie> &TTie) {
  return ei;
};

template<typename DerivedT,
         typename DerivedTT,
         typename DerivedTTif,
         typename DerivedTTie>
IGL_INLINE int tet_tuple_get_face(
    const int &ti, const int &fi, const int &ei, const bool &along,
    const std::vector<DerivedT> &T,
    const std::vector<DerivedTT> &TT,
    const std::vector<DerivedTTif> &TTif,
    const std::vector<DerivedTTie> &TTie) {
  return fi;
};

template<typename DerivedT,
         typename DerivedTT,
         typename DerivedTTif,
         typename DerivedTTie>
IGL_INLINE int tet_tuple_get_tet(
    const int &ti, const int &fi, const int &ei, const bool &along,
    const std::vector<DerivedT> &T,
    const std::vector<DerivedTT> &TT,
    const std::vector<DerivedTTif> &TTif,
    const std::vector<DerivedTTie> &TTie) {
  return ti;
};

template<typename DerivedT,
         typename DerivedTT,
         typename DerivedTTif,
         typename DerivedTTie>
IGL_INLINE bool tet_tuple_next_in_one_ring(
    int &ti, int &fi, int &ei, bool &along,
    const std::vector<DerivedT> &T,
    const std::vector<DerivedTT> &TT,
    const std::vector<DerivedTTif> &TTif,
    const std::vector<DerivedTTie> &TTie) {
  if (tet_tuple_is_on_boundary(ti, fi, ei, along, T, TT, TTif, TTie)) {
    do {
      tet_tuple_switch_face(ti, fi, ei, along, T, TT, TTif, TTie);
      tet_tuple_switch_tet(ti, fi, ei, along, T, TT, TTif, TTie);
      tet_tuple_switch_face(ti, fi, ei, along, T, TT, TTif, TTie);
      tet_tuple_switch_edge(ti, fi, ei, along, T, TT, TTif, TTie);
    } while (!tet_tuple_is_on_boundary(ti, fi, ei, along, T, TT, TTif,
                                       TTie));
    tet_tuple_switch_edge(ti, fi, ei, along, T, TT, TTif, TTie);
    return false;
  } else {
    fi = tetrahedron_local_FF(fi, (ei + 2) % 3);
    std::tie(ti, fi, ei) = std::make_tuple(TT[ti][fi], TTif[ti][fi],
                                           TTie[ti](fi, ei));
    fi = tetrahedron_local_FF(fi, (ei + 2) % 3);
    ei = (ei + (along ? 1 : 2)) % 3;
//  equivalent to the following:
//    tet_tuple_switch_face(ti, fi, ei, along, T, TT, TTif, TTie);
//    tet_tuple_switch_tet(ti, fi, ei, along, T, TT, TTif, TTie);
//    tet_tuple_switch_face(ti, fi, ei, along, T, TT, TTif, TTie);
//    tet_tuple_switch_edge(ti, fi, ei, along, T, TT, TTif, TTie);
    return true;
  }
};

// Also possibly a boundary edge, but not the face.
template<typename DerivedT,
         typename DerivedTT,
         typename DerivedTTif,
         typename DerivedTTie>
IGL_INLINE bool tet_tuple_is_on_boundary(
    const int &ti, const int &fi, const int &ei, const bool &along,
    const std::vector<DerivedT> &T,
    const std::vector<DerivedTT> &TT,
    const std::vector<DerivedTTif> &TTif,
    const std::vector<DerivedTTie> &TTie) {
  return TT[ti][fi] == -1;
};

// let's do stupid recursion........
IGL_INLINE std::set<int> tet_tuple_get_tets_with_vert(
    int ti, int fi, int ei, bool along,
    const std::vector<Eigen::RowVector4i>& T,
    const std::vector<Eigen::RowVector4i>& TT,
    const std::vector<Eigen::RowVector4i>& TTif,
    const std::vector<Eigen::Matrix<int, 4, 3>>& TTie) {

  struct recursive_insertion {
    int vid_in_t(int tt) {
      auto Ttt = T[tt];
      for (int ii :{0, 1, 2, 3}) if (Ttt[ii] == global_v) return ii;
      return -1;
    };

    void recurse(int vid, int t) {
      if (vid == -1 || t == -1) return;
      int old_size = neighbor_tets.size();
      neighbor_tets.insert(t);
      if(neighbor_tets.size() == old_size) return;
      for (auto ii:{0, 1, 2, 3}) {
        if (ii != vid) {
          int tt = TT[t][ii];
          if(tt == -1) continue;
          int vv = vid_in_t(tt);
          recurse(vv, tt);
        }
      }
    }

    std::set<int> neighbor_tets;
    int global_v;
    const std::vector<Eigen::RowVector4i> &T;
    const std::vector<Eigen::RowVector4i> &TT;

    recursive_insertion(int _gv, decltype(T) _T, decltype(TT) _TT) :
        global_v(_gv), T(_T), TT(_TT) {}

  };

  int vid = (igl::dev::tetrahedron_local_FF)(fi, along ? ei : (ei + 1) % 3);
  int v = T[ti][vid];
  recursive_insertion insertor(v, T, TT);
  insertor.recurse(vid, ti);
  return insertor.neighbor_tets;
}

IGL_INLINE bool tet_tuples_equal(
    const int &t1, const int &f1, const int &e1, const bool &rev1,
    const int &t2, const int &f2, const int &e2, const bool &rev2) {
  return t1 == t2 &&
      f1 == f2 &&
      e1 == e2 &&
      rev1 == rev2;
};
}
}

#ifdef IGL_STATIC_LIBRARY
// Explicit template instantiation
template int igl::dev::tet_tuple_get_vert<Eigen::Matrix<int, 1, 4, 1, 1, 4>,
                                  Eigen::Matrix<int, 1, 4, 1, 1, 4>,
                                          Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 4, 3, 0, 4, 3> >(int const&, int const&, int const&, bool const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 4, 3, 0, 4, 3>, std::__1::allocator<Eigen::Matrix<int, 4, 3, 0, 4, 3> > > const&);

template bool igl::dev::tet_tuple_is_on_boundary<Eigen::Matrix<int, 1, 4, 1, 1,
                                                            4>,
                                                 Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 4, 3, 0, 4, 3> >(int const&, int const&, int const&, bool const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 4, 3, 0, 4, 3>, std::__1::allocator<Eigen::Matrix<int, 4, 3, 0, 4, 3> > > const&);


template void igl::dev::tet_tuple_switch_tet<Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 4, 3, 0, 4, 3> >(int&, int&, int&, bool&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 4, 3, 0, 4, 3>, std::__1::allocator<Eigen::Matrix<int, 4, 3, 0, 4, 3> > > const&);
template void igl::dev::tet_tuple_switch_edge<Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 4, 3, 0, 4, 3> >(int&, int&, int&, bool&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 4, 3, 0, 4, 3>, std::__1::allocator<Eigen::Matrix<int, 4, 3, 0, 4, 3> > > const&);
template void igl::dev::tet_tuple_switch_face<Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 1, 4, 1, 1, 4>, Eigen::Matrix<int, 4, 3, 0, 4, 3> >(int&, int&, int&, bool&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1, 4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 1, 4, 1, 1,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               4>, std::__1::allocator<Eigen::Matrix<int, 1, 4, 1, 1, 4> > > const&, std::__1::vector<Eigen::Matrix<int, 4, 3, 0, 4, 3>, std::__1::allocator<Eigen::Matrix<int, 4, 3, 0, 4, 3> > > const&);
#endif
