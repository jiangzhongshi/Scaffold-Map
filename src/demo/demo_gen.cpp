//
// Created by Zhongshi Jiang on 10/21/16.
//
#include <igl/viewer/Viewer.h>
#include <igl/triangle/triangulate.h>
#include <igl/boundary_loop.h>
#include <igl/slice.h>
#include <igl/cat.h>
#include <igl/slice_into.h>
#include <igl/local_basis.h>
#include <igl/read_triangle_mesh.h>
#include <igl/polar_svd.h>
#include <sstream>
#include <iostream>
#include <cstdio>

#include <igl/write_triangle_mesh.h>
#include "../util/triangle_utils.h"

#define PROJECT_PATH "/Users/zhongshi/Workspace/Scaffold/src/scaffold-test/"


void demo_generator(Eigen::MatrixXd &V_w, Eigen::MatrixXi &F_w,
                    Eigen::MatrixXd &V_s3, Eigen::MatrixXi &F_s)
{
  using namespace Eigen;
  using namespace std;

  Eigen::MatrixXd V, Vo;
  Eigen::MatrixXi E, Fo;

  //basic frame data
  //
  // 11----------10
  //  |           |
  //  |  6-----7  |
  //  |  |     |  |
  //  |  |     8--9
  //  |  |
  //  |  |       3--2
  //  |  |       |  |
  //  |  5-------4  |
  //  0-------------1
  //
  V.resize(12, 2);
  V <<
    0, 0, 3, 0,        //0, 1
      3, .95, 2.4, .95,   //2, 3
      2.4, .5, .5, .5,      //4, 5
      .5, 1.5, 2, 1.5,      //6, 7
      2, 1.05, 2.6, 1.05,   //8, 9
      2.6, 2, 0, 2;        //10, 11

  E.resize(V.rows(), 2);
  for(int i = 0; i < E.rows(); i++)
    E.row(i) << i, i + 1;
  E(V.rows() - 1, 1) = 0;

  //triangle
  igl::triangle::triangulate(V, E, MatrixXd(), "a0.002qQ", Vo, Fo);

  //generate scaffold
  Matrix2d ob;
  ob << -1, -1, 3, 4;
  scaffold_generator(Vo, Fo, 0.05, V_w, F_w);

  // extrude and intersect.
  MatrixXd V_s = Vo;

  struct line_func
  {
    double a, b;

    double operator()(double y)
    { return a * y + b; };
  };
  auto linear_stretch = [](double s0, double t0, double s1, double t1)
  { // source0, target0, source1, target1
    Matrix2d S;
    S << s0, 1, s1, 1;
    Vector2d t;
    t << t0, t1;
    Vector2d coef = S.colPivHouseholderQr().solve(t);
    return line_func{coef[0], coef[1]};
  };
  auto f_up = linear_stretch(V(7, 1), V(7, 1), V(9, 1), 1.45);
  auto f_down = linear_stretch(V(4, 1), V(4, 1), V(2, 1), 0.55);
  for(int i = 0; i < Vo.rows(); i++)
    if(Vo(i, 0) > 1)
    {
      double &y = V_s(i, 1);
      if(1 < y && y < 1.5)
      {
        y = f_up(y);
      }
      else if(0.5 < y && y < 1)
      {
        y = f_down(y);
      }
    }

  V_s3.resize(V_s.rows(), 3);
  V_s3.leftCols(2) = V_s;
  F_s = Fo;

  // squeez a bit V_w ///TODO
//    f_up = linear_stretch(V(7,1),V(7,1), V(8,1), 1+0.01);
//    f_down = linear_stretch(V(4,1),V(4,1),V(3,1), 1-0.01);
//    for(int i=0; i<V_w.rows(); i++)
//        if (V(8,0) <= V_w(i,0) && V_w(i,0) <= V(1,0))
//        {
//            double &y = V_w(i,1);
//            if(1.001<y && y<1.5)
//                y = f_up(y);
//            else if (.5 < y && y < 1-0.01)
//                y = f_down(y);
//        }
}


