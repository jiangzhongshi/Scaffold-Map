//
// Created by Zhongshi Jiang on 2/9/17.
//

#ifndef SCAFFOLD_TEST_TET_UTILS_H
#define SCAFFOLD_TEST_TET_UTILS_H

#include <igl/remove_duplicate_vertices.h>
namespace igl {

template <
    typename DerivedV,
    typename DerivedT,
    typename DerivedF,
    typename DerivedSV,
    typename DerivedSVI,
    typename DerivedSVJ,
    typename DerivedST,
    typename DerivedSF>
IGL_INLINE void remove_duplicate_vertices(
    const Eigen::MatrixBase<DerivedV>& V,
    const Eigen::MatrixBase<DerivedT>& T,
    const Eigen::MatrixBase<DerivedF>& F,
    const double epsilon,
    Eigen::PlainObjectBase<DerivedSV>& SV,
    Eigen::PlainObjectBase<DerivedSVI>& SVI,
    Eigen::PlainObjectBase<DerivedSVJ>& SVJ,
    Eigen::PlainObjectBase<DerivedST>& ST,
    Eigen::PlainObjectBase<DerivedSF>& SF);
}

#include "tet_utils.cpp"
#endif
