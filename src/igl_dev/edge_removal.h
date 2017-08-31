//
// Created by Zhongshi Jiang on 4/3/17.
//

#ifndef SCAFFOLD_TEST_EDGE_REMOVAL_H
#define SCAFFOLD_TEST_EDGE_REMOVAL_H
#include <vector>
#include <igl/igl_inline.h>
#include <functional>
#include <set>

namespace igl {
namespace dev {
// only if quality is improved
template<typename DerivedT, typename DerivedTT,
         typename DerivedTTif, typename DerivedTTie>
IGL_INLINE bool tet_tuple_edge_removal(int ti,
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
                                       std::vector<int,
                                                   std::allocator<int>> new_tets_id);
template<typename DerivedT, typename DerivedTT,
         typename DerivedTTif, typename DerivedTTie>
IGL_INLINE bool tet_tuple_edge_removal(int ti,
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
                                                 &TTie);

template<typename DerivedT, typename DerivedTT,
         typename DerivedTTif, typename DerivedTTie>
IGL_INLINE bool tet_tuple_edge_removal_force(int ti,
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
                                       &TTie);
}
}
#ifndef IGL_STATIC_LIBRARY
# include "edge_removal.cpp"
#endif
#endif //SCAFFOLD_TEST_EDGE_REMOVAL_H
