//
// Created by Zhongshi Jiang on 4/11/17.
//

#ifndef SCAFFOLD_TEST_MULTI_FACE_REMOVAL_H
#define SCAFFOLD_TEST_MULTI_FACE_REMOVAL_H
#include <igl/igl_inline.h>
#include <vector>
namespace igl { namespace dev {
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
                                  std::vector<int> &new_tets_id);

template<typename DerivedT, typename DerivedTT,
         typename DerivedTTif, typename DerivedTTie>
bool tet_tuple_multi_face_removal_force(int ti,
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
                                  std::vector<DerivedTTie> &TTie);
}}
#endif //SCAFFOLD_TEST_MULTI_FACE_REMOVAL_H
