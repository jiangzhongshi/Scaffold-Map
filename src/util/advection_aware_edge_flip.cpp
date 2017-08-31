//
// Created by Zhongshi Jiang on 3/9/17.
//
#include <tuple>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <iostream>
#include <igl/viewer/Viewer.h>
#include <igl/flip_avoiding_line_search.h>
#include <igl/triangle_triangle_adjacency.h>

namespace igl {
namespace flip_avoiding {
extern double get_min_pos_root_2D(const Eigen::MatrixXd &uv,
                                  const Eigen::MatrixXi &F,
                                  const Eigen::MatrixXd &d,
                                  int f);
}
}

// Test Case
/*
 *
 MatrixXi F0(6,3), F1(6,3);
F0 << 0,1,2,1,3,2,
0,4,1, 1,5,3,
0,2,7,2,3,6;
F1 << 0,1,3,0,3,2,
0,4,1, 1,5,3,
0,2,7,2,3,6;

MatrixXi FF0,FFi0,FF1,FFi1;
igl::triangle_triangle_adjacency(F0,FF0,FFi0);
igl::triangle_triangle_adjacency(F1,FF1,FFi1);
simple_edge_flip(F0,FF0,FFi0,1,2);
 ASSERT_EQ(F0,F1);
 ASSERT_EQ(FF0,FF1);
 ASSERT_EQ(FFi0,FFi1);
 */
void simple_edge_flip(Eigen::MatrixXi &F, Eigen::MatrixXi &FF,
                      Eigen::MatrixXi &FFi, int f0,
                      int e0) {

  const int f1 = FF(f0, e0);
  const int e1 = FFi(f0, e0);

  const int e01 = (e0 + 1) % 3;
  const int e02 = (e0 + 2) % 3;
  const int e11 = (e1 + 1) % 3;
  const int e12 = (e1 + 2) % 3;
  const int f01 = FF(f0, e01);
  const int f02 = FF(f0, e02);
  const int f11 = FF(f1, e11);
  const int f12 = FF(f1, e12);

  F(f0, e01) = F(f1, e12);
  F(f1, e11) = F(f0, e02);

  assert(FF(f0, e0) == f1);
  assert(FF(f0, e01) == f01);
  FF(f0, e0) = f11;
  FF(f0, e01) = f1;
  assert(FF(f1, e1) == f0);
  assert(FF(f1, e11) == f11);
  FF(f1, e1) = f01;
  FF(f1, e11) = f0;
  if (f11 != -1)
    FF(f11, FFi(f1, e11)) = f0;
  if (f01 != -1)
    FF(f01, FFi(f0, e01)) = f1;

  assert(FFi(f0, e0) == e1);
  FFi(f0, e0) = FFi(f1, e11);
  assert(FFi(f1, e1) == e0);
  FFi(f1, e1) = FFi(f0, e01);
  FFi(f0, e01) = e11;
  FFi(f1, e11) = e01;

  // use updated FFi([f0 f1],:)
  if (f11 != -1)
    FFi(f11, FFi(f0, e0)) = e0;
//  if (f12 != -1)
//    FFi(f12, FFi(f1, e12)) = e12;
  if (f01 != -1)
    FFi(f01, FFi(f1, e1)) = e1;
//  if (f02 != -1)
//    FFi(f02, FFi(f0, e02)) = e02;
}
inline double get_next_quad_zero(double a, double b, double c, double alpha) {
  // double precision: alpha < 1, so meaningful digits extends to 1e-15.
  using namespace std;
  double ret = -1.0;
  auto eval = [&](double x){return a*x*x+b*x+c;};
//  assert(eval(alpha) >= 0);
  if (std::abs(a) < 1.0e-13) // Oh this is so delicate.
  {
    if (b >= 0) ret = INFINITY;
    else ret = -c / b;
  } else if (a > 0) {
    if (b >= 0)
      ret = INFINITY;
    else {
      double delta = pow(b, 2) - 4 * a * c;
      double t = 2*c / (-b + sqrt(delta));
      ret = t > alpha ? t : INFINITY;
    }
  } else { // a<0
    double delta = pow(b, 2) - 4 * a * c;
    if(b >=0)
      ret =  - (b + sqrt(delta)) / (2 * a);
    else
      ret =  2*c / (- b + sqrt(delta)) ;
    // larger root either way
  }

  // append a bisection.
  if(std::isfinite(ret) && eval(ret) < 0)
  {
    double l = alpha;
    double r = ret;
    while(true) {
      double mid = (l+r)/2;
      if(mid >= r || mid <= l) break;
      double val = eval(mid);
      if(val >= 0) {
        l = mid;
        if(val <= 1e-14) break;
      } else r = mid;
    }
    ret = l;
  }
  assert(!isfinite(ret) || eval(ret) >= 0);
  return ret;
}

inline double get_flipping_root_2D(const Eigen::MatrixXd &uv,
                                   const Eigen::MatrixXi &F,
                                   const Eigen::MatrixXd &d,
                                   int f, double alpha) {
  // copied from igl::flip_avoiding_line_search
  int v1 = F(f, 0);
  int v2 = F(f, 1);
  int v3 = F(f, 2);
  // get quadratic coefficients (ax^2 + b^x + c)
  const double &U11 = uv(v1, 0);
  const double &U12 = uv(v1, 1);
  const double &U21 = uv(v2, 0);
  const double &U22 = uv(v2, 1);
  const double &U31 = uv(v3, 0);
  const double &U32 = uv(v3, 1);

  const double &V11 = d(v1, 0);
  const double &V12 = d(v1, 1);
  const double &V21 = d(v2, 0);
  const double &V22 = d(v2, 1);
  const double &V31 = d(v3, 0);
  const double &V32 = d(v3, 1);

  double a = V11 * V22 - V12 * V21 - V11 * V32 + V12 * V31 + V21 * V32
      - V22 * V31;
  double b = U11 * V22 - U12 * V21 - U21 * V12 + U22 * V11 - U11 * V32
      + U12 * V31 + U31 * V12 - U32 * V11 + U21 * V32 - U22 * V31
      - U31 * V22 + U32 * V21;
  double c = U11 * U22 - U12 * U21 - U11 * U32 + U12 * U31 + U21 * U32
      - U22 * U31;

  return get_next_quad_zero(a, b, c, alpha);
}
double advection_aware_edge_flip(const Eigen::MatrixXd &V,
                                 const Eigen::MatrixXd &d,
                                 double alpha_max,
                                 Eigen::MatrixXi &F_raw,
                                 Eigen::MatrixXi &FF_raw,
                                 Eigen::MatrixXi &FFi_raw) {
  using namespace std;
  Eigen::MatrixXi F=F_raw, FF = FF_raw, FFi = FFi_raw;
  assert(V.cols() == 2 && F.cols() == 3 && "Not A Planar Triangle Mesh");
  auto fn = F.rows();
  auto vn = V.rows();
  double result_step = 0;

  using step_wrapper_t = std::tuple<double, int, int>;
  auto get_mid_vert_local = [](auto a, auto b, auto c) {
    // still need to enhance numerical stability
    int i = 0;
    if (std::abs(a[0] - b[0])<1e-10 && std::abs(b[0]-c[0])<1e-10)
      i = 1;

    bool ab = a[i] < b[i];
    if (ab == (c[i] < a[i]))
      return 0;
    else if (ab == (b[i] < c[i]))
      return 1;
    else
      return 2;
  };

  // construct heap
  std::vector<step_wrapper_t> face_heap(fn);
  std::vector<int> stamps(fn, 0);
  for (int f = 0; f < fn; f++) {
    face_heap[f] = std::make_tuple( -get_flipping_root_2D(V, F, d, f, 0), f, 0);
  }
  std::make_heap(face_heap.begin(), face_heap.end());
  std::vector<step_wrapper_t> flip_history;
  std::vector<double> alpha_history; alpha_history.push_back(0.0);
  while (true) {
    std::pop_heap(face_heap.begin(), face_heap.end());
    step_wrapper_t critical = face_heap.back();
    face_heap.pop_back();

    double a = -std::get<0>(critical);
    int f = std::get<1>(critical);

    if (a > alpha_max) {  // larger than desired
      if(0.9*a > alpha_max)
      {
        std::tie(F_raw,FF_raw,FFi_raw) = std::make_tuple(F,FF,FFi);
        return alpha_max;
      }
      result_step = alpha_max;
      alpha_history.push_back(result_step);
      break;
    }
    result_step = a;

    if (std::get<2>(critical) < stamps[f]) continue;
    alpha_history.push_back(result_step);

    int v = get_mid_vert_local(V.row(F(f, 0)) + a * d.row(F(f, 0)),
                               V.row(F(f, 1)) + a * d.row(F(f, 1)),
                               V.row(F(f, 2)) + a * d.row(F(f, 2)));

    int e = (v + 1) % 3;
    if (FF(f, e) == -1) {
      break;
    } else {
      // not taking care of anything.
      int ff = FF(f, e);

      simple_edge_flip(F, FF, FFi, f, e);

      for (auto fff:{f, ff}) {
        double next_root = get_flipping_root_2D(V, F, d, fff, a);
        if(next_root < a) {
          std::cout << "Step decrease and break by" << next_root - a << endl;
          a = next_root;
        }
        face_heap.push_back(std::make_tuple(-next_root, fff, ++stamps[fff]));
        std::push_heap(face_heap.begin(), face_heap.end());
      }

      flip_history.push_back(std::make_tuple(a, f, e));
    }
  }

  int stable_step = alpha_history.size() - 2;
  for(; stable_step >= 0; stable_step --) {
    if(alpha_history[stable_step] >= 0.9 * alpha_history.back()) continue;
    if(alpha_history[stable_step + 1] - alpha_history[stable_step] > 1e-3) {
      result_step = 0.9*alpha_history[stable_step + 1] +
          0.1*alpha_history[stable_step];
      break;
    }
  }
  assert(stable_step>=0 && "Really would be no feasible flip interval?");
  for(int i=0; i<stable_step; i++) {
    int f=-1, e=-1;
    std::tie(std::ignore,f,e) = flip_history[i];
    simple_edge_flip(F_raw, FF_raw, FFi_raw, f, e);
  }

  return result_step;
}
double advection_aware_edge_flip(const Eigen::MatrixXd &V,
                                 const Eigen::MatrixXd &d,
                                 Eigen::MatrixXi &F) {
  Eigen::MatrixXi FF, FFi;
  igl::triangle_triangle_adjacency(F, FF, FFi);
  return advection_aware_edge_flip(V, d, 1.0, F, FF, FFi);
}
