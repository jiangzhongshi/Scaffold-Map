//
// Created by Zhongshi Jiang on 4/20/17.
//
#include "ReWeightedARAP.h"
#include "ScafData.h"
#include "util/triangle_utils.h"

#include "igl/arap.h"
#include "igl/cat.h"
#include "igl/doublearea.h"
#include "igl/grad.h"
#include "igl/local_basis.h"
#include "igl/per_face_normals.h"
#include "igl/slice_into.h"
#include "igl/serialize.h"

#include <igl/flip_avoiding_line_search.h>
#include <igl/boundary_facets.h>
#include <igl/unique.h>
#include <igl/slim.h>
#include <igl/grad.h>
#include <igl/is_symmetric.h>
#include <igl/polar_svd.h>
#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/edge_lengths.h>
#include <igl/local_basis.h>
#include <igl/readOBJ.h>
#include <igl/repdiag.h>
#include <igl/vector_area_matrix.h>
#include <iostream>
#include <igl/slice.h>
#include <igl/colon.h>
#include <igl/min_quad_with_fixed.h>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <map>
#include <set>
#include <vector>
#include <igl/Timer.h>
#include <igl/edge_flaps.h>

void ReWeightedARAP::solve_weighted_arap(Eigen::MatrixXd &uv) {
  using namespace Eigen;
  using namespace std;
  int dim = d_.dim;
  igl::Timer timer;
  timer.start();

  const VectorXi& bnd_ids = d_.frame_ids;

  const auto bnd_n = bnd_ids.size();
  assert(bnd_n > 0);
  MatrixXd bnd_pos;
  igl::slice(d_.w_uv, bnd_ids, 1, bnd_pos);

  ArrayXi known_ids(bnd_n * dim);
  ArrayXi unknown_ids((v_n - bnd_n) * dim);

  { // get the complement of bnd_ids.
    int assign = 0, i = 0;
    for (int get = 0; i < v_n && get < bnd_ids.size(); i++) {
      if (bnd_ids(get) == i) get++;
      else unknown_ids(assign++) = i;
    }
    while (i < v_n) unknown_ids(assign++) = i++;
    assert(assign + bnd_ids.size() == v_n);
  }

  VectorXd known_pos(bnd_ids.size() * dim);
  for (int d = 0; d < dim; d++) {
    auto n_b = bnd_ids.rows();
    known_ids.segment(d * n_b, n_b) = bnd_ids.array() + d * v_n;
    known_pos.segment(d * n_b, n_b) = bnd_pos.col(d);
    unknown_ids.block(d * (v_n - n_b), 0, v_n - n_b, unknown_ids.cols()) =
        unknown_ids.topRows(v_n - n_b) + d * v_n;
  }
  //std::cout<<"Slicing Knowns "<<timer.getElapsedTime()<<std::endl;
  //timer.start();

  Eigen::SparseMatrix<double> L;
  Eigen::VectorXd rhs;

  // fixed frame solving:
  // x_e as the fixed frame, x_u for unknowns (mesh + unknown scaffold)
  // min ||(A_u*x_u + A_e*x_e) - b||^2
  // => A_u'*A_u*x_u + A_u'*A_e*x_e = Au'*b
  // => A_u'*A_u*x_u  = Au'* (b - A_e*x_e) := Au'* b_u
  // => L * x_u = rhs
  //
  // separate matrix build:
  // min ||A_m x_m - b_m||^2 + ||A_s x_all - b_s||^2 + soft + proximal
  // First change dimension of A_m to fit for x_all
  // (Not just at the end, since x_all is flattened along dimensions)
  // L = A_m'*A_m + A_s'*A_s + soft + proximal
  // rhs = A_m'* b_m + A_s' * b_s + soft + proximal
  //
  using namespace std;
  Eigen::SparseMatrix<double> L_m, L_s;
  Eigen::VectorXd rhs_m, rhs_s;
  build_surface_linear_system(L_m, rhs_m);  // complete Am, with soft
  build_scaffold_linear_system(L_s, rhs_s); // complete As, without proximal
  // we don't need proximal term

  L = L_m + L_s;
  rhs = rhs_m + rhs_s;
  L.makeCompressed();

  //std::cout<<"Constructing matrices "<<timer.getElapsedTime()<<std::endl;
//  VectorXd uv_flat(dim * v_n);
//  for (int i = 0; i < dim; i++)
//    for (int j = 0; j < v_n; j++)
//      uv_flat(v_n * i + j) = d_.w_uv(j, i);
//
//  VectorXd uv_flat_unknown(dim * (v_n - bnd_ids.size()));
//  igl::slice(uv_flat, unknown_ids, 1, uv_flat_unknown);

  timer.start();
  Eigen::VectorXd unknown_Uc((v_n - d_.frame_ids.size()) * dim), Uc(dim * v_n);
  bool solve_with_cg = false;
  if (solve_with_cg) {
//    for (auto t:{1e-6}) {
//      timer.start();
//      ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Upper>
//          CGsolver;
//      CGsolver.setTolerance(t);
//      unknown_Uc = CGsolver.compute(L).solveWithGuess(rhs, uv_flat_unknown);
//      cout << t << "CGSolve = " << timer.getElapsedTime() << endl;
//      std::cout << "#iterations:     " << CGsolver.iterations() << std::endl;
//      std::cout << "estimated error: " << CGsolver.error() << std::endl;
//    }
  } else {
    SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
    unknown_Uc = solver.compute(L).solve(rhs);
    //cout << "Direct Solve = " << timer.getElapsedTime() << endl;
  }
  //timer.start();
  igl::slice_into(unknown_Uc, unknown_ids.matrix(), 1, Uc);
  igl::slice_into(known_pos, known_ids.matrix(), 1, Uc);

  for (int i = 0; i < dim; i++)
    uv.col(i) = Uc.block(i * v_n, 0, v_n, 1);
  //cout << "Slice back = " << timer.getElapsedTime() << endl;
}

void ReWeightedARAP::build_surface_linear_system(Eigen::SparseMatrix<double> &L,
                                         Eigen::VectorXd &rhs) const {
  const int f_n = W_m.rows();
  const int v_n = (int)(Dx_s.cols()) - (int)(d_.frame_ids.size());
  // to get the  complete A
  const int dim = d_.dim;

  Eigen::VectorXd sqrtM = d_.m_M.array().sqrt();
  Eigen::SparseMatrix<double> A(dim * dim * f_n, dim * v_n);
  auto decoy_Dx_m = Dx_m;
  decoy_Dx_m.conservativeResize(W_m.rows(), v_n);
  auto decoy_Dy_m = Dy_m;
  decoy_Dy_m.conservativeResize(W_m.rows(), v_n);
  if(dim == 2) {
    buildAm(sqrtM, decoy_Dx_m, decoy_Dy_m, W_m, A);
  } else {
    auto decoy_Dz_m = Dz_m;
    decoy_Dz_m.conservativeResize(W_m.rows(), v_n);
    buildAm(sqrtM, decoy_Dx_m, decoy_Dy_m, decoy_Dz_m, W_m, A);
  }

  Eigen::SparseMatrix<double> At = A.transpose();
  At.makeCompressed();

  Eigen::SparseMatrix<double> id_m(At.rows(), At.rows());
  id_m.setIdentity();

  L = At * A;

  Eigen::VectorXd frhs;
  buildRhs(sqrtM, W_m, Ri_m, frhs);
  rhs = At*frhs;

  // add soft constraints.
  for (auto const &x:d_.soft_cons) {
    int v_idx = x.first;

    for (int d = 0; d < dim; d++) {
      rhs(d * (v_n) + v_idx) += d_.soft_const_p * x.second(d); // rhs
      L.coeffRef(d * v_n + v_idx,
                 d * v_n + v_idx) += d_.soft_const_p; // diagonal
    }
  }
}

void ReWeightedARAP::build_scaffold_linear_system(Eigen::SparseMatrix<double>
                                                  &L,
                                                 Eigen::VectorXd &rhs) const {
  using namespace Eigen;

  const int f_n = W_s.rows();
  const int v_n = Dx_s.cols();
  const int dim = d_.dim;

  Eigen::VectorXd sqrtM = d_.s_M.array().sqrt();
  Eigen::SparseMatrix<double> A(dim * dim * f_n, dim * v_n);
  if (dim == 2)
    buildAm(sqrtM, Dx_s, Dy_s, W_s, A);
  else
    buildAm(sqrtM, Dx_s, Dy_s, Dz_s, W_s, A);

  const VectorXi& bnd_ids = d_.frame_ids;

  auto bnd_n = bnd_ids.size(); assert(bnd_n > 0);
  MatrixXd bnd_pos;
  igl::slice(d_.w_uv, bnd_ids, 1, bnd_pos);

  ArrayXi known_ids(bnd_ids.size() * dim);
  ArrayXi unknown_ids((v_n - bnd_ids.rows()) * dim);

  { // get the complement of bnd_ids.
    int assign = 0, i = 0;
    for (int get = 0; i < v_n && get < bnd_ids.size(); i++) {
      if (bnd_ids(get) == i) get++;
      else unknown_ids(assign++) = i;
    }
    while (i < v_n) unknown_ids(assign++) = i++;
    assert(assign + bnd_ids.size() == v_n);
  }

  VectorXd known_pos(bnd_ids.size() * dim);
  for (int d = 0; d < dim; d++) {
    auto n_b = bnd_ids.rows();
    known_ids.segment(d * n_b, n_b) = bnd_ids.array() + d * v_n;
    known_pos.segment(d * n_b, n_b) = bnd_pos.col(d);
    unknown_ids.block(d * (v_n - n_b), 0, v_n - n_b, unknown_ids.cols()) =
        unknown_ids.topRows(v_n - n_b) + d * v_n;
  }
  Eigen::VectorXd sqrt_M = d_.s_M.array().sqrt();

  // slice
  // 'manual slicing for A(:, unknown/known)'
  Eigen::SparseMatrix<double> Au, Ae;
  {
    using TY = double;
    using TX = double;
    auto &X = A;

    int xm = X.rows();
    int xn = X.cols();
    int ym = xm;
    int yn = unknown_ids.size();
    int ykn = known_ids.size();

    std::vector<int> CI(xn, -1);
    std::vector<int> CKI(xn, -1);
    // initialize to -1
    for (int i = 0; i < yn; i++) CI[unknown_ids(i)] = (i);
    for (int i = 0; i < ykn; i++) CKI[known_ids(i)] = i;
    Eigen::DynamicSparseMatrix<TY, Eigen::ColMajor> dyn_Y(ym, yn);
    Eigen::DynamicSparseMatrix<TY, Eigen::ColMajor> dyn_K(ym, ykn);
    // Take a guess at the number of nonzeros (this assumes uniform distribution
    // not banded or heavily diagonal)
    dyn_Y.reserve(A.nonZeros());
    dyn_K.reserve(A.nonZeros() * ykn / xn);
    // Iterate over outside
    for (int k = 0; k < X.outerSize(); ++k) {
      // Iterate over inside
      if (CI[k] != -1)
        for (typename Eigen::SparseMatrix<TX>::InnerIterator it(X, k); it;
             ++it) {
          dyn_Y.coeffRef(it.row(), CI[it.col()]) = it.value();
        }
      else
        for (typename Eigen::SparseMatrix<TX>::InnerIterator it(X, k); it;
             ++it) {
          dyn_K.coeffRef(it.row(), CKI[it.col()]) = it.value();
        }
    }
    Au = Eigen::SparseMatrix<TY>(dyn_Y);
    Ae = Eigen::SparseMatrix<double>(dyn_K);
  }

  Eigen::SparseMatrix<double> Aut = Au.transpose();
  Aut.makeCompressed();

  Eigen::SparseMatrix<double> id(Aut.rows(), Aut.rows());
  id.setIdentity();

  L = Aut * Au;

  Eigen::VectorXd frhs;
  buildRhs(sqrtM, W_s, Ri_s, frhs);

  rhs = Aut * (frhs - Ae * known_pos);
}

void ReWeightedARAP::buildAm(const Eigen::VectorXd &sqrt_M,
                             const Eigen::SparseMatrix<double> &Dx,
                             const Eigen::SparseMatrix<double> &Dy,
                             const Eigen::MatrixXd &W,
                             Eigen::SparseMatrix<double> &Am)  {
  using namespace Eigen;
  // formula (35) in paper
  std::vector<Triplet<double> > IJV;
  const int f_n = W.rows();
  const int v_n = Dx.cols();

  IJV.reserve(4 * (Dx.outerSize() + Dy.outerSize()));

  /*A = [W11*Dx, W12*Dx;
       W11*Dy, W12*Dy;
       W21*Dx, W22*Dx;
       W21*Dy, W22*Dy];*/
  for (int k = 0; k < Dx.outerSize(); ++k) {
    for (SparseMatrix<double>::InnerIterator it(Dx, k); it; ++it) {
      int dx_r = it.row();
      int dx_c = it.col();
      double val = it.value() * sqrt_M(dx_r);

      IJV.push_back(Triplet<double>(dx_r, dx_c, val * W(dx_r,0)));
      IJV.push_back(Triplet<double>(dx_r, v_n + dx_c, val * W(dx_r,1)));

      IJV.push_back(Triplet<double>(2 * f_n + dx_r, dx_c, val * W(dx_r,2)));
      IJV.push_back(
          Triplet<double>(2 * f_n + dx_r, v_n + dx_c, val * W(dx_r,3)));
    }
  }

  for (int k = 0; k < Dy.outerSize(); ++k) {
    for (SparseMatrix<double>::InnerIterator it(Dy, k); it; ++it) {
      int dy_r = it.row();
      int dy_c = it.col();
      double val = it.value() * sqrt_M(dy_r);

      IJV.push_back(Triplet<double>(f_n + dy_r, dy_c,
                                    val * W(dy_r,0)));
      IJV.push_back(Triplet<double>(f_n + dy_r, v_n + dy_c,
                                    val * W(dy_r,1)));

      IJV.push_back(Triplet<double>(3 * f_n + dy_r, dy_c,
                                    val * W(dy_r,2)));
      IJV.push_back(Triplet<double>(3 * f_n + dy_r, v_n + dy_c,
                          val * W(dy_r,3)));
    }
  }
  Am.setFromTriplets(IJV.begin(), IJV.end());
}

void ReWeightedARAP::buildAm(const Eigen::VectorXd &sqrt_M,
                             const Eigen::SparseMatrix<double> &Dx,
                             const Eigen::SparseMatrix<double> &Dy,
                             const Eigen::SparseMatrix<double> &Dz,
                             const Eigen::MatrixXd &W,
                             Eigen::SparseMatrix<double> &Am)  {
  using namespace Eigen;
  // formula (35) in paper
  std::vector<Triplet<double> > IJV;
  IJV.reserve(9 * (Dx.outerSize() + Dy.outerSize() + Dz.outerSize()));

  const int f_n = W.rows();
  const int v_n = Dx.cols();

  /*A = [W11*Dx, W12*Dx, W13*Dx;
         W11*Dy, W12*Dy, W13*Dy;
         W11*Dz, W12*Dz, W13*Dz;
         W21*Dx, W22*Dx, W23*Dx;
         W21*Dy, W22*Dy, W23*Dy;
         W21*Dz, W22*Dz, W23*Dz;
         W31*Dx, W32*Dx, W33*Dx;
         W31*Dy, W32*Dy, W33*Dy;
         W31*Dz, W32*Dz, W33*Dz;];*/
  for (int k = 0; k < Dx.outerSize(); k++) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(Dx, k); it; ++it) {
      int dx_r = it.row();
      int dx_c = it.col();
      double val = it.value();

      double m_0 = sqrt_M(dx_r);
      double m_3 = sqrt_M(dx_r);
      double m_6 = sqrt_M(dx_r);
      IJV.push_back(Eigen::Triplet<double>(dx_r, dx_c, m_0 * val * W(dx_r,0)));
      IJV.push_back(Eigen::Triplet<double>(dx_r,
                                           v_n + dx_c,
                                           m_0 * val * W(dx_r,1)));
      IJV.push_back(Eigen::Triplet<double>(dx_r,
                                           2 * v_n + dx_c,
                                           m_0 * val * W(dx_r,2)));

      IJV.push_back(Eigen::Triplet<double>(3 * f_n + dx_r,
                                           dx_c,
                                           m_3 * val * W(dx_r,3)));
      IJV.push_back(Eigen::Triplet<double>(3 * f_n + dx_r,
                                           v_n + dx_c,
                                           m_3 * val * W(dx_r,4)));
      IJV.push_back(Eigen::Triplet<double>(3 * f_n + dx_r,
                                           2 * v_n + dx_c,
                                           m_3 * val * W(dx_r,5)));

      IJV.push_back(Eigen::Triplet<double>(6 * f_n + dx_r,
                                           dx_c,
                                           m_6 * val * W(dx_r,6)));
      IJV.push_back(Eigen::Triplet<double>(6 * f_n + dx_r,
                                           v_n + dx_c,
                                           m_6 * val * W(dx_r,7)));
      IJV.push_back(Eigen::Triplet<double>(6 * f_n + dx_r,
                                           2 * v_n + dx_c,
                                           m_6 * val * W(dx_r,8)));
    }
  }

  for (int k = 0; k < Dy.outerSize(); k++) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(Dy, k); it; ++it) {
      int dy_r = it.row();
      int dy_c = it.col();
      double val = it.value();

      double m_1 = sqrt_M(dy_r);
      double m_4 = sqrt_M(dy_r);
      double m_7 = sqrt_M(dy_r);
      IJV.push_back(Eigen::Triplet<double>(f_n + dy_r,
                                           dy_c,
                                           m_1 * val * W(dy_r,0)));
      IJV.push_back(Eigen::Triplet<double>(f_n + dy_r,
                                           v_n + dy_c,
                                           m_1 * val * W(dy_r,1)));
      IJV.push_back(Eigen::Triplet<double>(f_n + dy_r,
                                           2 * v_n + dy_c,
                                           m_1 * val * W(dy_r,2)));

      IJV.push_back(Eigen::Triplet<double>(4 * f_n + dy_r,
                                           dy_c,
                                           m_4 * val * W(dy_r,3)));
      IJV.push_back(Eigen::Triplet<double>(4 * f_n + dy_r,
                                           v_n + dy_c,
                                           m_4 * val * W(dy_r,4)));
      IJV.push_back(Eigen::Triplet<double>(4 * f_n + dy_r,
                                           2 * v_n + dy_c,
                                           m_4 * val * W(dy_r,5)));

      IJV.push_back(Eigen::Triplet<double>(7 * f_n + dy_r,
                                           dy_c,
                                           m_7 * val * W(dy_r,6)));
      IJV.push_back(Eigen::Triplet<double>(7 * f_n + dy_r,
                                           v_n + dy_c,
                                           m_7 * val * W(dy_r,7)));
      IJV.push_back(Eigen::Triplet<double>(7 * f_n + dy_r,
                                           2 * v_n + dy_c,
                                           m_7 * val * W(dy_r,8)));
    }
  }

  for (int k = 0; k < Dz.outerSize(); k++) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(Dz, k); it; ++it) {
      int dz_r = it.row();
      int dz_c = it.col();
      double val = it.value();
      double m_2 = sqrt_M(dz_r);
      double m_5 = sqrt_M(dz_r);
      double m_8 = sqrt_M(dz_r);
      IJV.push_back(Eigen::Triplet<double>(2 * f_n + dz_r,
                                           dz_c,
                                           m_2 * val * W(dz_r,0)));
      IJV.push_back(Eigen::Triplet<double>(2 * f_n + dz_r,
                                           v_n + dz_c,
                                           m_2 * val * W(dz_r,1)));
      IJV.push_back(Eigen::Triplet<double>(2 * f_n + dz_r,
                                           2 * v_n + dz_c,
                                           m_2 * val * W(dz_r,2)));

      IJV.push_back(Eigen::Triplet<double>(5 * f_n + dz_r,
                                           dz_c,
                                           m_5 * val * W(dz_r,3)));
      IJV.push_back(Eigen::Triplet<double>(5 * f_n + dz_r,
                                           v_n + dz_c,
                                           m_5 * val * W(dz_r,4)));
      IJV.push_back(Eigen::Triplet<double>(5 * f_n + dz_r,
                                           2 * v_n + dz_c,
                                           m_5 * val * W(dz_r,5)));

      IJV.push_back(Eigen::Triplet<double>(8 * f_n + dz_r,
                                           dz_c,
                                           m_8 * val * W(dz_r,6)));
      IJV.push_back(Eigen::Triplet<double>(8 * f_n + dz_r,
                                           v_n + dz_c,
                                           m_8 * val * W(dz_r,7)));
      IJV.push_back(Eigen::Triplet<double>(8 * f_n + dz_r,
                                           2 * v_n + dz_c,
                                           m_8 * val * W(dz_r,8)));
    }
  }

  Am.setFromTriplets(IJV.begin(), IJV.end());
}


void ReWeightedARAP::buildRhs(const Eigen::VectorXd &sqrt_M,
                              const Eigen::MatrixXd &W,
                              const Eigen::MatrixXd& Ri,
                              Eigen::VectorXd &f_rhs)  {
  const int dim = (W.cols() == 4) ? 2:3;
  const int f_n = W.rows();
  f_rhs.resize(dim * dim * f_n);

  if (dim == 2) {
    /*b = [W11*R11 + W12*R21; (formula (36))
         W11*R12 + W12*R22;
         W21*R11 + W22*R21;
         W21*R12 + W22*R22];*/
    for (int i = 0; i < f_n; i++) {
      auto sqrt_area = sqrt_M(i);
      f_rhs(i + 0 * f_n) =sqrt_area*(W(i, 0) * Ri(i, 0) + W(i, 1) * Ri(i, 1));
      f_rhs(i + 1 * f_n) =sqrt_area*(W(i, 0) * Ri(i, 2) + W(i, 1) * Ri(i, 3));
      f_rhs(i + 2 * f_n) =sqrt_area*(W(i, 2) * Ri(i, 0) + W(i, 3) * Ri(i, 1));
      f_rhs(i + 3 * f_n) =sqrt_area*(W(i, 2) * Ri(i, 2) + W(i, 3) * Ri(i, 3));
    }
  }else {
    /*b = [W11*R11 + W12*R21 + W13*R31;
         W11*R12 + W12*R22 + W13*R32;
         W11*R13 + W12*R23 + W13*R33;
         W21*R11 + W22*R21 + W23*R31;
         W21*R12 + W22*R22 + W23*R32;
         W21*R13 + W22*R23 + W23*R33;
         W31*R11 + W32*R21 + W33*R31;
         W31*R12 + W32*R22 + W33*R32;
         W31*R13 + W32*R23 + W33*R33;];*/
    for (int i = 0; i < f_n; i++) {
      auto sqrt_area = sqrt_M(i);
      f_rhs(i + 0 * f_n) =sqrt_area*
          (W(i,0) * Ri(i, 0) + W(i,1) * Ri(i, 1) + W(i,2) * Ri(i, 2));
      f_rhs(i + 1 * f_n) =sqrt_area*
          (W(i,0) * Ri(i, 3) + W(i,1) * Ri(i, 4) + W(i,2) * Ri(i, 5));
      f_rhs(i + 2 * f_n) =sqrt_area*
          (W(i,0) * Ri(i, 6) + W(i,1) * Ri(i, 7) + W(i,2) * Ri(i, 8));
      f_rhs(i + 3 * f_n) =sqrt_area*
          (W(i,3) * Ri(i, 0) + W(i,4) * Ri(i, 1) + W(i,5) * Ri(i, 2));
      f_rhs(i + 4 * f_n) =sqrt_area*
          (W(i,3) * Ri(i, 3) + W(i,4) * Ri(i, 4) + W(i,5) * Ri(i, 5));
      f_rhs(i + 5 * f_n) =sqrt_area*
          (W(i,3) * Ri(i, 6) + W(i,4) * Ri(i, 7) + W(i,5) * Ri(i, 8));
      f_rhs(i + 6 * f_n) =sqrt_area*
          (W(i,6) * Ri(i, 0) + W(i,7) * Ri(i, 1) + W(i,8) * Ri(i, 2));
      f_rhs(i + 7 * f_n) =sqrt_area*
          (W(i,6) * Ri(i, 3) + W(i,7) * Ri(i, 4) + W(i,8) * Ri(i, 5));
      f_rhs(i + 8 * f_n) =sqrt_area*
          (W(i,6) * Ri(i, 6) + W(i,7) * Ri(i, 7) + W(i,8) * Ri(i, 8));
    }
  }

}

