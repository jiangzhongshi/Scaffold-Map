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
#include <igl/lim/lim.h>
#include <igl/slice.h>
#include <igl/edge_flaps.h>
#include <igl/edge_lengths.h>
#include <igl/flip_edge.h>
#include <igl/collapse_edge.h>
#include <queue>
#include <igl/Timer.h>

//OMG a buggy function
void my_split(const Eigen::MatrixXd &V,
              const Eigen::MatrixXi &F,
              int mv_num, int mf_num,
              Eigen::MatrixXd &new_V,
              Eigen::MatrixXi &new_F) {
  using namespace Eigen;
  using namespace std;
  new_V = V;
  new_F = F;

  return;
  Eigen::MatrixXi E, EF, EI;
  Eigen::VectorXi EMAP;

  std::vector<Eigen::RowVector2d> new_vert;
  int new_vert_id = V.rows();
  std::vector<Eigen::RowVector3i> new_face;

  igl::edge_flaps(F, E, EMAP, EF, EI);

  new_F = F;

  auto vert_in_mesh = [&](int f) {
    return new_F(f, 0) < mv_num && new_F(f, 1) < mv_num && new_F(f, 2) < mv_num;
  };
  // only loop in scaffold Faces
  for (int f = mf_num; f < F.rows(); f++) {
    // new_F is intentionally modified, so wouldn't be modified twice.
    if (vert_in_mesh(f)) {
      for (int v = 0; v < 3; v++) {
        int e = EMAP(v * F.rows() + f);
        if (EF(e, 0) >= mf_num && EF(e, 1) >= mf_num)
          // scaffold edge : split it!
          // iff connecting scaffold triangles.
        {
          new_vert.push_back((V.row(E(e, 0)) + V.row(E(e, 1))) / 2.);

          for (int i = 0; i < 2; i++) {
            new_F.row(EF(e, i)) << F(EF(e, i), EI(e, i)), E(e, i), new_vert_id;
            new_face.push_back(RowVector3i(
                F(EF(e, i), EI(e, i)), new_vert_id, E(e, (i + 1) % 2)
            ));
          }

          new_vert_id++;

          break;
        }
      }
    }
  }

  new_V.resize(V.rows() + new_vert.size(), 2);
  new_V.topRows(V.rows()) = V;
  for (int i = V.rows(); i < new_V.rows(); i++)
    new_V.row(i) = new_vert[i - V.rows()];

  new_F.conservativeResize(F.rows() + new_face.size(), 3);
  for (int i = F.rows(); i < new_F.rows(); i++)
    new_F.row(i) = new_face[i - F.rows()];

};

void scaffold_interpolation(const Eigen::MatrixXd &w_uv,
                            const Eigen::MatrixXi &w_F,
                            const Eigen::MatrixXd &target_uv,
                            const Eigen::VectorXi &out_bnd,
                            const Eigen::VectorXi &inn_bnd,
                            int harmonic_order,
                            Eigen::MatrixXd &interp) {

  using namespace Eigen;
  if (true) {
    VectorXi two_bnd(inn_bnd.size() + out_bnd.size());
    two_bnd << inn_bnd, out_bnd;

    MatrixXd bc_diff(two_bnd.size(), 2);
    bc_diff.topRows(inn_bnd.size()) = igl::slice(target_uv, inn_bnd, 1)
        - igl::slice(w_uv, inn_bnd, 1);
    bc_diff.bottomRows(out_bnd.size()) = MatrixXd::Zero(out_bnd.size(), 2);

    MatrixXd bc_coord = igl::slice(w_uv, two_bnd, 1) + bc_diff;

    MatrixXd linterp, hinterp;
//        igl::lscm(w_uv, w_F, two_bnd, bc_coord,linterp);
//        interp = linterp;
//        interp.col(0) = linterp.col(1);
//        interp.col(1) = linterp.col(0);
    if (harmonic_order == 0)
      igl::harmonic(w_F, two_bnd, bc_coord, 1, hinterp);
    else
      igl::harmonic(w_uv, w_F, two_bnd, bc_coord, harmonic_order, hinterp);

//        std::cout<<"hlscm:"<<(hinterp - linterp).norm()<<std::endl;
    interp = hinterp;



    //interp = w_uv + dinterp;
    return;
  }
}

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
void scaffold_generator(const Eigen::MatrixXd &V0, const Eigen::MatrixXi &F0,
                        double max_area_cons, Eigen::MatrixXd &uv_s,
                        Eigen::MatrixXi &F_out_s) {
  using namespace Eigen;
  using namespace std;

  MatrixXd uv0, V1, V, H, uv2, V_rect;
  MatrixXi E, F1, F2;
  uv0 = V0.leftCols(2);
  Matrix2d ob;// = rect_corners;
  {
    VectorXd uv_max = V0.colwise().maxCoeff();
    VectorXd uv_min = V0.colwise().minCoeff();
    VectorXd uv_mid = (uv_max + uv_min) / 2.;

    double scaf_range = 3;
    ob.row(0) = uv_mid + scaf_range * (uv_min - uv_mid);
    ob.row(1) = uv_mid + scaf_range * (uv_max - uv_mid);
    std::cout << ob << std::endl;
  }

  std::vector<std::vector<int> > all_bnd;
  igl::boundary_loop(F0, all_bnd); //problem specific: not only one bnd

  int bnd_pieces = all_bnd.size();

  int bnd_total_size = 0;
  for (auto &bnd:all_bnd) bnd_total_size += bnd.size();

  VectorXi bnd1(bnd_total_size);
  {
    int current_size = 0;
    for (auto &bnd:all_bnd) {
      int len = static_cast<int>(bnd.size());
      for (int i = 0; i < len; i++)
        bnd1(current_size + i) = bnd[i];
      current_size += len;
    }
  }

  // choose some barycenter within the original model as holes
  // maybe due to inherent bug in *triangle* lib, one is not enough
  // so we are using 10 for each.
  H = MatrixXd::Zero(10 * bnd_pieces, 2);
  for (int l = 0, next_h = 0; l < bnd_pieces; l++)
    for (int f = l * 2 * SQUARE_COUNT; f < l * 2 * SQUARE_COUNT + 10; f++) {
      for (int i = 0; i < 3; i++) H.row(next_h) += V0.row(F0(f, i));
      next_h++;
    }
  H /= 3.;

  MatrixXd V_bnd;
  V_bnd.resize(bnd1.size(), uv0.cols());
  for (int i = 0; i < bnd1.size(); i++) {
    V_bnd.row(i) = uv0.row(bnd1(bnd1.size() - i - 1));
  }

  // construct V_outer. based on frame_pts.
  Vector2d rect_len;
  rect_len << ob(1, 0) - ob(0, 0), ob(1, 1) - ob(0, 1);
  int frame_points = (int) (
      rect_len.maxCoeff() / sqrt(4 * max_area_cons / sqrt(3)) + 1);
  frame_points = 10;
  V_rect.resize(4 * frame_points, 2);
  for (int i = 0; i < frame_points; i++) {
    // 0,0;0,1
    V_rect.row(i) << ob(0, 0), ob(0, 1) + i * rect_len(1) / frame_points;
    // 0,0;1,1
    V_rect.row(i + frame_points)
        << ob(0, 0) + i * rect_len(0) / frame_points, ob(1, 1);
    // 1,0;1,1
    V_rect.row(i + 2 * frame_points) << ob(1, 0), ob(1, 1) - i * rect_len(1) /
        frame_points;
    // 1,0;0,1
    V_rect.row(i + 3 * frame_points)
        << ob(1, 0) - i * rect_len(0) / frame_points, ob(0, 1);
    // 0,0;0,1
  }

  // Concatenate Vert and Edge
  igl::cat(1, V_bnd, V_rect, V);

  E.resize(V.rows(), 2);
  for (int i = 0; i < E.rows(); i++)
    E.row(i) << i, i + 1;
  {
    int current_size = 0;
    for (auto &bnd:all_bnd) {
      int len = bnd.size();
      E(current_size + len - 1, 1) = current_size;
      current_size += len;
    }
    E(V.rows() - 1, 1) = static_cast<int>(current_size);
  }

  igl::triangle::triangulate(V, E, H,
                             "a" + std::to_string(max_area_cons) + "qYYQ", uv2,
                             F2);

  MatrixXd uv;
  MatrixXi F_out;
  mesh_cat(uv0, F0, uv2, F2, uv, F_out);

  // V_whole = MatrixXd::Zero(uv.rows(),3);
  // V_whole.leftCols(2) = uv;
  my_split(uv, F_out, uv0.rows(), F0.rows(), uv_s, F_out_s);
}

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

void write_viewer_to_png(igl::viewer::Viewer &viewer, std::string file_path) {
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>
      R(1280 * 4, 800 * 4);
  auto G = R, B = R, A = R;

  viewer.core.draw_buffer(viewer.data, viewer.opengl, false, R, G, B, A);
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
    return std::min(quality_by_length(len(0, 0),
                                      len(0, 1),
                                      len(0, 2)),
                    quality_by_length(len(1, 0),
                                      len(1, 1),
                                      len(1, 2)));

  };
  auto quality_improvement = [&](size_t e) -> double {
    double before = std::min(flap_qual(e, 0), flap_qual(e, 1));
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

#include <igl/triangle_tuple.h>
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
    Eigen::Matrix<typename DerivedV::Scalar, 1, 3> u;
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

template<typename DerivedV, typename DerivedF>
inline void adjusted_local_basis(
    const Eigen::MatrixBase<DerivedV> &V,
    const Eigen::MatrixBase<DerivedF> &F,
    Eigen::MatrixBase<DerivedV> &B1,
    Eigen::MatrixBase<DerivedV> &B2,
    Eigen::MatrixBase<DerivedV> &B3,
    double eps) {
  using namespace Eigen;
  using namespace std;
  B1.derived().resize(F.rows(), 3);
  B2.derived().resize(F.rows(), 3);
  B3.derived().resize(F.rows(), 3);

  for (unsigned i = 0; i < F.rows(); ++i) {
    Eigen::Matrix<typename DerivedV::Scalar, 1, 3>
        v1 = (V.row(F(i, 1)) - V.row(F(i, 0))).normalized();
    Eigen::Matrix<typename DerivedV::Scalar, 1, 3>
        t = V.row(F(i, 2)) - V.row(F(i, 0));
    Eigen::Matrix<typename DerivedV::Scalar, 1, 3>
        v3 = v1.cross(t).normalized();
    Eigen::Matrix<typename DerivedV::Scalar, 1, 3>
        v2 = v1.cross(v3).normalized();

    // if((bool)U(i)) {
    //   v1 << 1, 0, 0;
    //   v2 << 0, -1, 0;
    //   v3 << 0, 0, 0;

//    }
    B1.row(i) = v1;
    B2.row(i) = -v2;
    B3.row(i) = v3;
  }
}
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
template void adjusted_local_basis<Eigen::Matrix<double, -1, -1, 0, -1, -1>,
                                   Eigen::Matrix<int, -1, -1, 0, -1, -1> >
    (Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const &,
     Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const &,
     Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > &,
     Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > &,
     Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > &, double);

template void adjusted_grad<Eigen::Matrix<double, -1, -1, 0, -1, -1>,
                            Eigen::Matrix<int, -1, -1, 0, -1, -1> >
    (Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const &,
     Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const &,
     Eigen::SparseMatrix<Eigen::Matrix<double, -1, -1, 0, -1, -1>::Scalar,
                         0,
                         int> &,
     double);


