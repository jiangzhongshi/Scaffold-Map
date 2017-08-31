//
// Created by Zhongshi Jiang on 11/16/16.
//

#ifndef SCAFFOLD_TEST_TRIANGLE_UTILS_H
#define SCAFFOLD_TEST_TRIANGLE_UTILS_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include <igl/viewer/Viewer.h>
void read_mesh_with_uv_seam(std::string filename, Eigen::MatrixXd& V, 
                            Eigen::MatrixXi& F);

void write_viewer_to_png(igl::viewer::Viewer &v, std::string file_path);

void mesh_cat(const Eigen::MatrixXd &V1, const Eigen::MatrixXi &F1,
              const Eigen::MatrixXd &V2, const Eigen::MatrixXi &F2,
              Eigen::MatrixXd &Vo, Eigen::MatrixXi &Fo);


template<typename Scalar>
inline void soft_cat(
    const int dim,
    const Eigen::SparseMatrix<Scalar> &A,
    const Eigen::SparseMatrix<Scalar> &B,
    Eigen::SparseMatrix<Scalar> &C) {

  assert(dim == 1 || dim == 2);
  using namespace Eigen;
  // Special case if B or A is empty
  if (A.size() == 0) {
    C = B;
    return;
  }
  if (B.size() == 0) {
    C = A;
    return;
  }

  C = SparseMatrix<Scalar>(
      dim == 1 ? A.rows() + B.rows() : std::max(A.rows(), B.rows()),
      dim == 1 ? std::max(A.cols(), B.cols()) : A.cols() + B.cols());
  Eigen::VectorXi per_col = Eigen::VectorXi::Zero(C.cols());
  if (dim == 1) {
    for (int k = 0; k < A.outerSize(); ++k)
      for (typename SparseMatrix<Scalar>::InnerIterator it(A, k); it; ++it)
        per_col(k)++;
    for (int k = 0; k < B.outerSize(); ++k)
      for (typename SparseMatrix<Scalar>::InnerIterator it(B, k); it; ++it)
        per_col(k)++;

  } else {
    for (int k = 0; k < A.outerSize(); ++k)
      for (typename SparseMatrix<Scalar>::InnerIterator it(A, k); it; ++it)
        per_col(k)++;
    for (int k = 0; k < B.outerSize(); ++k)
      for (typename SparseMatrix<Scalar>::InnerIterator it(B, k); it; ++it)
        per_col(A.cols() + k)++;
  }
  C.reserve(per_col);
  if (dim == 1) {
    for (int k = 0; k < A.outerSize(); ++k) {
      for (typename SparseMatrix<Scalar>::InnerIterator it(A, k); it; ++it) {
        C.insert(it.row(), k) = it.value();
      }
    }
    for (int k = 0; k < B.outerSize(); ++k)
      for (typename SparseMatrix<Scalar>::InnerIterator it(B, k); it; ++it) {
        C.insert(A.rows() + it.row(), k) = it.value();
      }

  } else {
    for (int k = 0; k < A.outerSize(); ++k) {
      for (typename SparseMatrix<Scalar>::InnerIterator it(A, k); it; ++it) {
        C.insert(it.row(), k) = it.value();
      }
    }
    for (int k = 0; k < B.outerSize(); ++k) {
      for (typename SparseMatrix<Scalar>::InnerIterator it(B, k); it; ++it) {
        C.insert(it.row(), A.cols() + k) = it.value();
      }
    }
  }
  C.makeCompressed();

};


void scaffold_generator(const Eigen::MatrixXd &V0, const Eigen::MatrixXi &F0,
                        double max_area_cons, Eigen::MatrixXd &uv,
                        Eigen::MatrixXi &F_out);

void
scaffold_interpolation(const Eigen::MatrixXd &w_V, const Eigen::MatrixXi &w_F,
                       const Eigen::MatrixXd &target_uv,
                       const Eigen::VectorXi &out_bnd,
                       const Eigen::VectorXi &inn_bnd, int harmonic_order,
                       Eigen::MatrixXd &interp);

void my_split(const Eigen::MatrixXd &V,
              const Eigen::MatrixXi &F,
              int mv_num, int mf_num,
              Eigen::MatrixXd &new_V,
              Eigen::MatrixXi &new_F);

template<
    typename DerivedA,
    typename DerivedR,
    typename DerivedT,
    typename DerivedU,
    typename DerivedS,
    typename DerivedV>
void polar_svd2x2(
    const Eigen::PlainObjectBase<DerivedA> &A,
    Eigen::PlainObjectBase<DerivedR> &R,
    Eigen::PlainObjectBase<DerivedT> &T,
    Eigen::PlainObjectBase<DerivedU> &U,
    Eigen::PlainObjectBase<DerivedS> &S,
    Eigen::PlainObjectBase<DerivedV> &V);

int count_flips(const Eigen::MatrixXd &V,
                const Eigen::MatrixXi &F,
                const Eigen::MatrixXd &uv);

void get_flips(const Eigen::MatrixXd &V,
               const Eigen::MatrixXi &F,
               const Eigen::MatrixXd &uv,
               std::vector<int> &flip_idx);

void read_mesh_and_init();

void triangle_improving_edge_flip(const Eigen::MatrixXd &V,
                                  Eigen::MatrixXi &F,
                                  Eigen::MatrixXi &E,
                                  Eigen::MatrixXi &EF,
                                  Eigen::MatrixXi &EV,
                                  Eigen::VectorXi &EMAP_vec);

template<typename... Types>
void tet_improve(Types... args) {};

inline int get_obtuse_angle(const Eigen::MatrixXd &V,
                            const Eigen::RowVector3i &F) {
  Eigen::RowVectorXd u01 = V.row(F(1)) - V.row(F(0));
  Eigen::RowVectorXd u02 = V.row(F(2)) - V.row(F(0));
  Eigen::RowVectorXd u12 = V.row(F(2)) - V.row(F(1));

  if (u01.dot(u02) < 0) return 0;
  if (u01.dot(u12) > 0) return 1;
  if (u02.dot(u12) < 0) return 2;
  return -1;
}

void smooth_single_vertex(const int &f,
                          const int &e,
                          const bool &a,
                          const Eigen::MatrixXi &F,
                          const Eigen::MatrixXi &FF,
                          const Eigen::MatrixXi &FFi,
                          Eigen::MatrixXd &V);

template<typename DerivedV, typename DerivedF>
void adjusted_grad(const Eigen::MatrixBase<DerivedV> &V,
                          const Eigen::MatrixBase<DerivedF> &F,
                          Eigen::SparseMatrix<typename DerivedV::Scalar> &G,
                          double eps);

template<typename DerivedV, typename DerivedF>
void adjusted_local_basis(
    const Eigen::MatrixBase<DerivedV> &V,
    const Eigen::MatrixBase<DerivedF> &F,
    Eigen::MatrixBase<DerivedV> &B1,
    Eigen::MatrixBase<DerivedV> &B2,
    Eigen::MatrixBase<DerivedV> &B3,
    double eps);
#endif //SCAFFOLD_TEST_TRIANGLE_UTILS_H
