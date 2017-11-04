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


// https://github.com/janba/DSC/blob/master/is_mesh/util.h#L415
struct quality_utils {
  using vec3 = Eigen::RowVector3d;
  inline static double ms_length(const vec3 &a,
                                 const vec3 &b,
                                 const vec3 &c,
                                 const vec3 &d) {
    double result = 0.;
    result += (a - b).squaredNorm();
    result += (a - c).squaredNorm();
    result += (a - d).squaredNorm();
    result += (b - c).squaredNorm();
    result += (b - d).squaredNorm();
    result += (c - d).squaredNorm();
    return result / 6.;
  }

  inline static double rms_length(const vec3 &a,
                                  const vec3 &b,
                                  const vec3 &c,
                                  const vec3 &d) {
    return sqrt(ms_length(a, b, c, d));
  }

  inline static double signed_volume(const vec3 &a,
                                     const vec3 &b,
                                     const vec3 &c,
                                     const vec3 &d) {
    return (a - d).dot((b - d).cross(c - d)) / 6.;
  }

// https://hal.inria.fr/inria-00518327
  inline static double quality(const vec3 &a,
                               const vec3 &b,
                               const vec3 &c,
                               const vec3 &d) {
    double v = signed_volume(a, b, c, d);
    double lrms = rms_length(a, b, c, d);

    double q = 8.48528 * v / (lrms * lrms * lrms);
#ifdef DEBUG
    assert(!isnan(q));
#endif
    return q;
  }
};


#include "tet_utils.cpp"
#endif
