#include "triangle_utils.h"

#include <igl/png/writePNG.h>
#include <igl/triangle/triangulate.h>
#include <igl/boundary_loop.h>
#include <igl/cat.h>
#include <igl/lscm.h>
#include <igl/local_basis.h>
#include <map>
#include <iostream>
#include <igl/edge_flaps.h>
#include <igl/harmonic.h>
#include <igl/slice.h>
#include <igl/edge_flaps.h>
#include <igl/edge_lengths.h>
#include <igl/flip_edge.h>
#include <igl/collapse_edge.h>
#include <queue>
#include <igl/Timer.h>


#include "igl/remove_duplicates.h"
void mesh_cat(const Eigen::MatrixXd &V1, const Eigen::MatrixXi &F1,
              const Eigen::MatrixXd &V2, const Eigen::MatrixXi &F2,
              Eigen::MatrixXd &Vo, Eigen::MatrixXi &Fo) {
  using namespace Eigen;
  using namespace std;

  //check compatibility of VF1 and VF2.
  assert(F1.cols() == F2.cols() && V1.cols() == V2.cols() &&
      "Input mesh not compatible!");
  const int nv_up = V1.rows();
  MatrixXd Vc; MatrixXi Fc;
  igl::cat(1,V1,V2, Vc);
  MatrixXi F2_m = F2.array() + nv_up;
  igl::cat(1,F1,F2_m, Fc);
  VectorXi I;
  igl::remove_duplicates(Vc,Fc,Vo,Fo,I);
}

template<
    typename DerivedA,
    typename DerivedR,
    typename DerivedT,
    typename DerivedU,
    typename DerivedS,
    typename DerivedV>
void polar_svd2x2(
    const Eigen::MatrixBase<DerivedA> &A,
    Eigen::MatrixBase<DerivedR> &R,
    Eigen::MatrixBase<DerivedT> &T,
    Eigen::MatrixBase<DerivedU> &U,
    Eigen::MatrixBase<DerivedS> &S,
    Eigen::MatrixBase<DerivedV> &V) {
  //J. Blinn. Consider the lowly 2x2 matrix
  //A = [a,b;c,d]
  assert(A.rows() == 2 && A.cols() == 2);

  auto e = (A(0, 0) + A(1, 1)) / 2;
  auto f = (A(0, 0) - A(1, 1)) / 2;
  auto g = (A(0, 1) + A(1, 0)) / 2;
  auto h = (A(0, 1) - A(1, 0)) / 2;

  auto eh_sq = sqrt(e * e + h * h);
  auto fg_sq = sqrt(g * g + f * f);

  T.resize(2, 2);
  U.resize(2, 2);
  V.resize(2, 2);
  S.resize(2, 1);

  S << eh_sq + fg_sq, eh_sq - fg_sq;

  auto atan_gf = atan(g / f);
  auto atan_he = atan(h / e);
  auto gamma = (atan_gf + atan_he) / 2;
  auto beta = (-atan_gf + atan_he) / 2;
  U << cos(beta), sin(beta), -sin(beta), cos(beta);
  V << cos(gamma), -sin(gamma), sin(gamma), cos(gamma); //note the transpose

  R = U * V.transpose();
  T = V * S.asDiagonal() * V.transpose();
};

#define LAYER_COUNT 10
#define SQUARE_COUNT 100

void get_flips(const Eigen::MatrixXd &V,
               const Eigen::MatrixXi &F,
               const Eigen::MatrixXd &uv,
               std::vector<int> &flip_idx) {
  flip_idx.resize(0);
  for (int i = 0; i < F.rows(); i++) {

    Eigen::Vector2d v1_n = uv.row(F(i, 0));
    Eigen::Vector2d v2_n = uv.row(F(i, 1));
    Eigen::Vector2d v3_n = uv.row(F(i, 2));

    Eigen::MatrixXd T2_Homo(3, 3);
    T2_Homo.col(0) << v1_n(0), v1_n(1), 1;
    T2_Homo.col(1) << v2_n(0), v2_n(1), 1;
    T2_Homo.col(2) << v3_n(0), v3_n(1), 1;
    double det = T2_Homo.determinant();
    assert (det == det);
    if (det < 0) {
      //cout << "flip at face #" << i << " det = " << T2_Homo.determinant() << endl;
      flip_idx.push_back(i);
    }
  }
}

int count_flips(const Eigen::MatrixXd &V,
                const Eigen::MatrixXi &F,
                const Eigen::MatrixXd &uv) {

  std::vector<int> flip_idx;
  get_flips(V, F, uv, flip_idx);

  return flip_idx.size();
}

void write_viewer_to_png(igl::opengl::glfw::Viewer &viewer, std::string file_path) {
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>
      R(1280 * 4, 800 * 4);
  auto G = R, B = R, A = R;

  viewer.core.draw_buffer(viewer.data(), false, R, G, B, A);
  igl::png::writePNG(R, G, B, A, file_path);
}
void triangle_improving_edge_flip(const Eigen::MatrixXd &V,
                                  Eigen::MatrixXi &F,
                                  Eigen::MatrixXi &E,
                                  Eigen::MatrixXi &EF,
                                  Eigen::MatrixXi &EV,
                                  Eigen::VectorXi &EMAP_vec) {
  using namespace Eigen;
  using namespace std;
  //          e0                 e0
  //          /|\                / \
  //         / | \              /f0 \
  //     v1 /f1|f0\ v0  =>  v1 /_____\ v0
  //        \  |  /            \ f1  /
  //         \ | /              \   /
  //          \|/                \ /
  //          e1                 e1
  //        e0 -> e1    =>    v1 -> v0

  assert(V.cols() == 2 && F.cols() == 3 && "Not A Planar Triangle Mesh");

  auto EMAP = [&F, &EMAP_vec](size_t f, size_t e) -> int & {
    return EMAP_vec(e * F.rows() + f);
  };

  Eigen::VectorXd Elen;
  igl::edge_lengths(V, E, Elen);
  auto quality_by_length = [](double a, double b, double c) -> double {
    double s = (a + b + c) / 2;
    double area_sq = (s * (s - a) * (s - b) * (s - c));
//    assert(area_sq > -1e-10);
    if (area_sq < 0) area_sq = 0;
    return sqrt(area_sq) / (a * a + b * b + c * c);
  };
  auto qualities_for_flap = [&](size_t e) -> Eigen::RowVector2d {
    Eigen::MatrixXi F0(2, 3);
    F0 << E(e, 0), E(e, 1), EV(e, 0),
        E(e, 1), E(e, 0), EV(e, 1);
    assert(F0.minCoeff() >= 0); // no boundary involved.

    Eigen::MatrixXd len;
    igl::edge_lengths(V, F0, len);
    return Eigen::RowVector2d(quality_by_length(len(0, 0),
                                                len(0, 1),
                                                len(0, 2)),
                              quality_by_length(len(1, 0),
                                                len(1, 1),
                                                len(1, 2)));
  };

  std::vector<int> edge_stamp(E.rows(), 0);
  using improve = std::tuple<double, size_t, int>; // q improve, id, stamp
  std::vector<improve> edge_heap(E.rows()); // init as 0,0,0s

  Eigen::MatrixXd flap_qual = -MatrixXd::Ones(E.rows(), 2); // init -1,
  // iso-to-EFlap
  std::vector<double> face_quality(F.rows(), -1);
  for (size_t f = 0; f < F.rows(); f++) {
    face_quality[f] = quality_by_length(Elen(EMAP(f, 1)),
                                        Elen(EMAP(f, 2)),
                                        Elen(EMAP(f, 0)));
  }

  for (size_t e = 0; e < E.rows(); e++)
    for (auto f:{0, 1})
      if (EF(e, f) != -1)
        flap_qual(e, f) = face_quality[EF(e, f)];

  auto min_qlty_after = [&](size_t e) -> double {
    if (EV(e, 1) < 0 || EV(e, 0) < 0) return -1;
    Eigen::MatrixXd F0(2, 3);
    F0 << E(e, 0), EV(e, 1), EV(e, 0),
        E(e, 1), EV(e, 0), EV(e, 1);
    Eigen::MatrixXd len;
    igl::edge_lengths(V, F0, len);
    return (std::min)(quality_by_length(len(0, 0),
                                      len(0, 1),
                                      len(0, 2)),
                    quality_by_length(len(1, 0),
                                      len(1, 1),
                                      len(1, 2)));

  };
  auto quality_improvement = [&](size_t e) -> double {
    double before = (std::min)(flap_qual(e, 0), flap_qual(e, 1));
    double after = min_qlty_after(e);
    return after - before;
  };

  for (size_t e = 0; e < E.rows(); e++) {
    edge_heap[e] = std::make_tuple(quality_improvement(e), e, 0);
  }
  std::make_heap(edge_heap.begin(), edge_heap.end());

  // finish preparation. Begin loop
  while (true) { // well, it can never be empty.
    std::pop_heap(edge_heap.begin(), edge_heap.end());
    improve chosen_edge = edge_heap.back();
    edge_heap.pop_back();
    assert(std::get<0>(chosen_edge) >= std::get<0>(edge_heap[0]));

    double q_improv = std::get<0>(chosen_edge);
    size_t edge_id = std::get<1>(chosen_edge);
    int marker = std::get<2>(chosen_edge);

    if (q_improv <= 1e-7) break; // No possible improvement.
    // time stamp and manifold check
    if (marker < edge_stamp[edge_id])
      continue;

    // perform flip: handle F, E, EMAP, EF, EV
    //igl::flip_edge(F,E,ue,EMAP, uE2E, edge_id);
    const auto e0 = E(edge_id, 0);
    const auto e1 = E(edge_id, 1);
    const auto f0 = EF(edge_id, 0);
    const auto f1 = EF(edge_id, 1);
    const auto v0 = EV(edge_id, 0);
    const auto v1 = EV(edge_id, 1);
    if (v0 < 0 || v1 < 0) continue; // skip boundary.
    assert(f0 >= 0 && f1 >= 0);

    auto counter_clock_orient = [&V](size_t u0, size_t u1, size_t u2) -> bool {
      Eigen::Matrix2d det;
      det << V.row(u1) - V.row(u0), V.row(u2) - V.row(u0);
      return det.determinant() > 0;
    };
    // test if triangle (v1 v0 e0) and (v0 v1 e1) has correct orientation
    // manifold check.
    if (!counter_clock_orient(v1, v0, e0) || !counter_clock_orient(v0, v1, e1))
      continue;

    int _m0, _m1;
    for (auto m:{0, 1, 2}) {
      if (EMAP(f0, m) == edge_id) _m0 = m;
      if (EMAP(f1, m) == edge_id) _m1 = m;
    }
    const auto m0 = _m0;
    const auto m1 = _m1;
    assert(EMAP(f0, m0) == edge_id);
    assert(EMAP(f1, m1) == edge_id);

    E.row(edge_id) << v1, v0; // to be compatible with igl::flip_edge
    EV.row(edge_id) << e0, e1;
    // EF.row(edge_id) << EF.row(edge_id);
    // The three lines above determines the ordering below

    // iff EMAP is counter-clockwise ordering.
    const auto m02 = EMAP(f0, (m0 + 2) % 3);
    const auto m11 = EMAP(f1, (m1 + 1) % 3);
    const auto m12 = EMAP(f1, (m1 + 2) % 3);
    const auto m01 = EMAP(f0, (m0 + 1) % 3);

    EMAP(f0, (m0 + 1) % 3) = m02;
    EMAP(f0, (m0 + 2) % 3) = m11;
    EMAP(f1, (m1 + 1) % 3) = m12;
    EMAP(f1, (m1 + 2) % 3) = m01;

    if (EV(m02, 0) == e1) {
      EV(m02, 0) = v1;
    } else {
      assert(EV(m02, 1) == e1);
      EV(m02, 1) = v1;
    }

    if (EV(m12, 0) == e0) {
      EV(m12, 0) = v0;
    } else {
      assert(EV(m12, 1) == e0);
      EV(m12, 1) = v0;
    }

    if (EF(m01, 0) == f0) {
      EF(m01, 0) = f1;
      assert(EV(m01, 0) == e0);
      EV(m01, 0) = v1;
    } else {
      assert(EF(m01, 1) == f0);
      assert(EV(m01, 1) == e0);
      EF(m01, 1) = f1;
      EV(m01, 1) = v1;
    }

    if (EF(m11, 0) == f1) {
      assert(EV(m11, 0) == e1);
      EF(m11, 0) = f0;
      EV(m11, 0) = v0;
    } else {
      assert(EF(m11, 1) == f1);
      assert(EV(m11, 1) == e1);
      EF(m11, 1) = f0;
      EV(m11, 1) = v0;
    }
//    deprecated because EMAP may not be consistent with F. (WHY?)
//    assert(F(f0, (m0 + 2) % 3) == e1);
//    assert(F(f1, (m1 + 2) % 3) == e0);
//    F(f0, (m0 + 2) % 3) = v1;
//    F(f1, (m1 + 2) % 3) = v0;

    for (auto v:{0, 1, 2}) {
      if (F(f0, v) == e1) F(f0, v) = v1;
      if (F(f1, v) == e0) F(f1, v) = v0;
    }


    // flap_qual
    RowVector2d new_flap_qual = qualities_for_flap(edge_id);

    flap_qual.row(edge_id) << new_flap_qual;
    for (auto ff:{0, 1}) {
      auto f = EF(edge_id, ff);
      for (auto ee:{0, 1, 2}) {
        auto e = EMAP(f, ee);
        if (e != edge_id) {// this should skip exactly twice.
          flap_qual(e, EF(e, 0) == f ? 0 : 1) = new_flap_qual(ff);
        }
      }
    }

    // improve
    edge_heap.push_back(std::make_tuple(-q_improv, edge_id,
                                        ++edge_stamp[edge_id]));
    std::push_heap(edge_heap.begin(), edge_heap.end());
    for (auto e: {m02, m11, m12, m01}) {
      edge_heap.push_back(std::make_tuple(quality_improvement(e), e,
                                          ++edge_stamp[e]));
      std::push_heap(edge_heap.begin(), edge_heap.end());
    }

  }
//  for_each(edge_heap.begin(), edge_heap.end(), [](improve& i){
//    assert(std::get<0>(i) <= 0);
//  });
}

#include "../igl_dev/triangle_tuple.h"
#include <igl/readOBJ.h>
void smooth_single_vertex(const int &f0,
                          const int &e0,
                          const bool &a0,
                          const Eigen::MatrixXi &F,
                          const Eigen::MatrixXi &FF,
                          const Eigen::MatrixXi &FFi,
                          Eigen::MatrixXd &V) {
  Eigen::RowVectorXd avg_V(V.cols());
  int f = f0;
  int e = e0;
  bool a = a0;
  int n_neighbor = 0;
  int v0 = igl::triangle_tuple_get_vert(f, e, a, F, FF, FFi);
  do {
    igl::triangle_tuple_next_in_one_ring(f, e, a, F, FF, FFi);
    int v_op = igl::triangle_tuple_get_vert(f, e, !a, F, FF, FFi);
    avg_V += V.row(v_op);
    n_neighbor++;
  } while (f != f0);

  avg_V /= n_neighbor;
  V.row(v0) = avg_V;
}

#include "igl/grad.h"
template<typename DerivedV, typename DerivedF>
inline void adjusted_grad(const Eigen::MatrixBase<DerivedV> &V,
                          const Eigen::MatrixBase<DerivedF> &F,
                          Eigen::SparseMatrix<typename DerivedV::Scalar> &G,
                          double eps) {
  Eigen::Matrix<typename DerivedV::Scalar, Eigen::Dynamic, 3>
      eperp21(F.rows(), 3), eperp13(F.rows(), 3);
  int fixed = 0;
  for (int i = 0; i < F.rows(); ++i) {
    // renaming indices of vertices of triangles for convenience
    int i1 = F(i, 0);
    int i2 = F(i, 1);
    int i3 = F(i, 2);

    // #F x 3 matrices of triangle edge vectors, named after opposite vertices
    Eigen::Matrix<typename DerivedV::Scalar, 1, 3> v32 = V.row(i3) - V.row(i2);
    Eigen::Matrix<typename DerivedV::Scalar, 1, 3> v13 = V.row(i1) - V.row(i3);
    Eigen::Matrix<typename DerivedV::Scalar, 1, 3> v21 = V.row(i2) - V.row(i1);
    Eigen::Matrix<typename DerivedV::Scalar, 1, 3> n = v32.cross(v13);
    // area of parallelogram is twice area of triangle
    // area of parallelogram is || v1 x v2 ||
    // This does correct l2 norm of rows, so that it contains #F list of twice
    // triangle areas
    double dblA = std::sqrt(n.dot(n));
    Eigen::Matrix<typename DerivedV::Scalar, 1, 3> u(0,0,1);
    if (dblA > eps) {
      // now normalize normals to get unit normals
      u = n / dblA;
    } else {
      // Abstract equilateral triangle v1=(0,0), v2=(h,0), v3=(h/2, (sqrt(3)/2)*h)
      fixed ++;
      // get h (by the area of the triangle)
      dblA = eps;
      double h = sqrt((dblA) / sin(
          M_PI / 3.0)); // (h^2*sin(60))/2. = Area => h = sqrt(2*Area/sin_60)

      Eigen::Vector3d v1, v2, v3;
      v1 << 0, 0, 0;
      v2 << h, 0, 0;
      v3 << h / 2., (sqrt(3) / 2.) * h, 0;

      // now fix v32,v13,v21 and the normal
      v32 = v3 - v2;
      v13 = v1 - v3;
      v21 = v2 - v1;
      n = v32.cross(v13);
    }

    // rotate each vector 90 degrees around normal
    double norm21 = std::sqrt(v21.dot(v21));
    double norm13 = std::sqrt(v13.dot(v13));
    eperp21.row(i) = u.cross(v21);
    eperp21.row(i) =
        eperp21.row(i) / std::sqrt(eperp21.row(i).dot(eperp21.row(i)));
    eperp21.row(i) *= norm21 / dblA;
    eperp13.row(i) = u.cross(v13);
    eperp13.row(i) =
        eperp13.row(i) / std::sqrt(eperp13.row(i).dot(eperp13.row(i)));
    eperp13.row(i) *= norm13 / dblA;
  }

  std::vector<int> rs;
  rs.reserve(F.rows() * 4 * 3);
  std::vector<int> cs;
  cs.reserve(F.rows() * 4 * 3);
  std::vector<double> vs;
  vs.reserve(F.rows() * 4 * 3);

  // row indices
  for (int r = 0; r < 3; r++) {
    for (int j = 0; j < 4; j++) {
      for (int i = r * F.rows(); i < (r + 1) * F.rows(); i++) rs.push_back(i);
    }
  }

  // column indices
  for (int r = 0; r < 3; r++) {
    for (int i = 0; i < F.rows(); i++) cs.push_back(F(i, 1));
    for (int i = 0; i < F.rows(); i++) cs.push_back(F(i, 0));
    for (int i = 0; i < F.rows(); i++) cs.push_back(F(i, 2));
    for (int i = 0; i < F.rows(); i++) cs.push_back(F(i, 0));
  }

  // values
  for (int i = 0; i < F.rows(); i++) vs.push_back(eperp13(i, 0));
  for (int i = 0; i < F.rows(); i++) vs.push_back(-eperp13(i, 0));
  for (int i = 0; i < F.rows(); i++) vs.push_back(eperp21(i, 0));
  for (int i = 0; i < F.rows(); i++) vs.push_back(-eperp21(i, 0));
  for (int i = 0; i < F.rows(); i++) vs.push_back(eperp13(i, 1));
  for (int i = 0; i < F.rows(); i++) vs.push_back(-eperp13(i, 1));
  for (int i = 0; i < F.rows(); i++) vs.push_back(eperp21(i, 1));
  for (int i = 0; i < F.rows(); i++) vs.push_back(-eperp21(i, 1));
  for (int i = 0; i < F.rows(); i++) vs.push_back(eperp13(i, 2));
  for (int i = 0; i < F.rows(); i++) vs.push_back(-eperp13(i, 2));
  for (int i = 0; i < F.rows(); i++) vs.push_back(eperp21(i, 2));
  for (int i = 0; i < F.rows(); i++) vs.push_back(-eperp21(i, 2));

  // create sparse gradient operator matrix
  G.resize(3 * F.rows(), V.rows());
  std::vector<Eigen::Triplet<typename DerivedV::Scalar> > triplets;
  for (int i = 0; i < (int) vs.size(); ++i) {
    triplets.push_back(Eigen::Triplet<typename DerivedV::Scalar>(rs[i],
                                                                 cs[i],
                                                                 vs[i]));
  }
  G.setFromTriplets(triplets.begin(), triplets.end());
//  std::cout<<"Adjusted"<<fixed<<std::endl;
};

void read_mesh_with_uv_seam(std::string filename,
                            Eigen::MatrixXd &out_V,
                            Eigen::MatrixXi &UV_F) {
  // let's ignore sanity checks.
  Eigen::MatrixXd corner_normals;
  Eigen::MatrixXi fNormIndices;

  Eigen::MatrixXd UV_V;
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;

  igl::readOBJ(
      filename,
      V, UV_V, corner_normals, F, UV_F, fNormIndices);
 if(UV_V.rows()==0){
  out_V = V;
  UV_F = F;
  return;
}
  out_V.resize(UV_V.rows(), 3);
  std::vector<bool> filled(out_V.rows(),false);
  for(int i=0; i<UV_F.rows(); i++) {
    for(int j:{0,1,2}) {
      auto f = F(i,j);
      auto f_uv = UV_F(i,j);
      if(!filled[f_uv])
      {
        out_V.row(f_uv) = V.row(f);
        filled[f_uv] = true;
      }
    }
  }
}

// Explicit Instantiations.
template void adjusted_grad<Eigen::Matrix<double, -1, -1, 0, -1, -1>,
                            Eigen::Matrix<int, -1, -1, 0, -1, -1> >
    (Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const &,
     Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const &,
     Eigen::SparseMatrix<Eigen::Matrix<double, -1, -1, 0, -1, -1>::Scalar,
                         0,
                         int> &,
     double);


