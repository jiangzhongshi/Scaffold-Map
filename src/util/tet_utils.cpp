//
// Created by Zhongshi Jiang on 2/9/17.
//

#include <igl/igl_inline.h>
#include <Eigen/Dense>
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
    Eigen::PlainObjectBase<DerivedSF>& SF) {
  using namespace Eigen;
  using namespace std;
  remove_duplicate_vertices(V, epsilon, SV, SVI, SVJ);
  SF.resizeLike(F);
  ST.resizeLike(T);
  for (int f = 0; f < F.rows(); f++) {
    for (int c = 0; c < F.cols(); c++) {
      SF(f, c) = SVJ(F(f, c));
    }
  }
  for (int f = 0; f < T.rows(); f++) {
    for (int c = 0; c < T.cols(); c++) {
      ST(f, c) = SVJ(T(f, c));
    }
  }

};
}
/*
template void igl::remove_duplicate_vertices<Eigen::Matrix<double, -1, -1, 0,
                                                          -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, double, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&);
*/