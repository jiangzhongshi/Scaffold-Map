#include <igl/readOBJ.h>
#include <igl/readMESH.h>
#include <igl/viewer/Viewer.h>
#include <igl/edge_flaps.h>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <igl/tetrahedron_tetrahedron_adjacency.h>
#include "../src/igl_dev/tetrahedron_tetrahedron_adjacency.h"
#include "../src/igl_dev/tetrahedron_tuple.h"
#include "../src/igl_dev/retain_tetrahedral_adjacency.h"
#include "../src/igl_dev/edge_removal.h"
#include "../src/igl_dev/multi_face_removal.h"
#include "../src/util/tetrahedral_improvement.h"

#include <iostream>
#include <list>
using vecV2d = std::vector<Eigen::RowVector2d>;
//                           Eigen::aligned_allocator<Eigen::RowVector2d>>;
using vecV3d = std::vector<Eigen::RowVector3d>;
using vecV3i = std::vector<Eigen::RowVector3i>;
using vecV4i = std::vector<Eigen::RowVector4i>;
using T_t = vecV4i;
using TT_t = T_t;
using TTie_t = std::vector<Eigen::Matrix<int, 4, 3>>;


// https://github.com/janba/DSC/blob/master/is_mesh/util.h#L415
struct test_utils {
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


template <int enable=1>
int test_tetrahedron_adjacency() {
  using namespace std;
  using namespace Eigen;

  vecV3d V(9);
  vecV4i T(7);
  V[0] << 0, 0, -1;
  V[1] << 0, 0, 1;
  V[2] << 0, 1, 0;
  V[3] << -1, 0, 0;
  V[4] << -0.8, -0.5, 0;
  V[5] << -0.2, -0.8, 0;
  V[6] << 0.2, -0.8, 0;
  V[7] << 0.8, -0.5, 0;
  V[8] << 1, 0, 0;

  T[0] << 0, 2, 3, 1;
  T[1] << 0, 3, 4, 1;
  T[2] << 0, 4, 5, 1;
  T[3] << 0, 5, 6, 1;
  T[4] << 0, 6, 7, 1;
  T[5] << 0, 7, 8, 1;
  T[6] << 0, 8, 2, 1;

  MatrixXi
      mT = Eigen::Map<Matrix<int, -1, 4, RowMajor>>(T[0].data(), T.size(), 4);
  MatrixXi mTT, mTTif, mTTie;
  vecV4i TT, TTif;
  TTie_t TTie;
  igl::tetrahedron_tetrahedron_adjacency(mT, mTT, mTTif, mTTie);
  igl::dev::tetrahedron_tetrahedron_adjacency(T, TT, TTif, TTie);

//  std::cout << mT << std::endl << endl
//            << mTT << endl << endl
//            << mTTie << endl;
//  std::cout << "Here goes our new function" << endl;
//  for (auto r:T) cout << r << endl;
//  for (auto r:TT) cout << r << endl;
//  for (auto r:TTif) cout << r << endl;
//  for (auto r:TTie) cout << r << endl;

  return 0;
}

template <int enable=1>
int test_retain_tetrahedron_adjacency() {
  using namespace Eigen;
  using namespace std;

  // test a 2-3 flip
  vecV4i T(7);
  T[0] << 0, 2, 3, 1;
  T[1] << 0, 3, 4, 1;
  T[2] << 0, 4, 5, 1;
  T[3] << 0, 5, 6, 1;
  T[4] << 0, 6, 7, 1;
  T[5] << 0, 7, 8, 1;
  T[6] << 0, 8, 2, 1;

  auto manual_T = T;
  manual_T.resize(8);
  manual_T[0] << 0, 8, 3, 1;
  manual_T[6] << 8, 2, 3, 1;
  manual_T[7] << 0, 2, 3, 8;
  T_t new_T;
  new_T.push_back(manual_T[0]);
  new_T.push_back(manual_T[6]);
  new_T.push_back(manual_T[7]);

  vecV4i TT, TTif;
  TTie_t TTie;
  igl::dev::tetrahedron_tetrahedron_adjacency(T, TT, TTif, TTie);
//  std::cout << "raw: T,TT,TTi:" << endl;
//  for (auto r:T) cout << r << endl;cout<<endl;
//  for (auto r:TT) cout << r << endl;cout<<endl;
//  for (auto r:TTif) cout << r << endl;cout<<endl;
//  for (auto r:TTie) cout << r << endl;cout<<endl;
  vecV4i mTT, mTTif;
  TTie_t mTTie;
  igl::dev::tetrahedron_tetrahedron_adjacency(manual_T, mTT, mTTif, mTTie);
//  std::cout << "manual: T,TT,TTi:" << endl;
//  for (auto r:manual_T) cout << r << endl;cout<<endl;
//  for (auto r:TT) cout << r << endl;cout<<endl;
//  for (auto r:TTif) cout << r << endl;cout<<endl;
//  for (auto r:TTie) cout << r << endl<<endl;cout<<endl;

  std::set<int> delete_id {0,6};
  std::set<int> surround_id {1,5};

  retain_tetrahedral_adjacency(delete_id, surround_id,new_T, T,TT,TTif,TTie);
  for (int i = 0; i < T.size(); i++) {
    assert(mTT[i] == TT[i]);
    assert(mTTif[i] == TTif[i]);
    assert(mTTie[i] == TTie[i]);
  }
//  std::cout << "modified: T,TT,TTi:" << endl;
//  for (auto r:T) cout << r << endl;cout<<endl;
//  for (auto r:TT) cout << r << endl;cout<<endl;
//  for (auto r:TTif) cout << r << endl;cout<<endl;
//  for (auto r:TTie) cout << r << endl<<endl;cout<<endl;

  return 0;
}


template <int enable=1>
int test_retain_fix_end() {

  using namespace Eigen;
  using namespace std;
  using vecV2d = std::vector<Eigen::RowVector2d>;
  using vecV3d = std::vector<Eigen::RowVector3d>;
  using vecV3i = std::vector<Eigen::RowVector3i>;
  using vecV4i = std::vector<Eigen::RowVector4i>;
  using T_t = vecV4i;
  using TT_t = T_t;
  using TTie_t = std::vector<Eigen::Matrix<int, 4, 3>>;

  vecV3d V(9);
  vecV4i T(7);
  V[0] << 0, -0, 1;
  V[1] << 0, -0, -1;
  V[2] << 0, 1, 0;
  V[3] << -1, 0, 0;
  V[4] << -0.8, -0.5, 0;
  V[5] << -0.2, -0.8, 0;
  V[6] << 0.2, -0.8, 0;
  V[7] << 0.8, -0.5, 0;
  V[8] << 1, 0, 0;

  T[0] << 1, 2, 3, 0;
  T[1] << 1, 3, 4, 0;
  T[2] << 1, 4, 5, 0;
  T[3] << 1, 5, 6, 0;
  T[4] << 1, 6, 7, 0;
  T[5] << 1, 7, 8, 0;
  T[6] << 1, 8, 2, 0;

  std::set<int> delete_id, influence_id;
  delete_id.insert(0);
  delete_id.insert(1);
  delete_id.insert(3);
  delete_id.insert(5);


  vecV4i TT, TTif;
  TTie_t TTie;
  igl::dev::tetrahedron_tetrahedron_adjacency(T, TT, TTif, TTie);

  for (auto t:delete_id)
    for (auto f:{0, 1, 2, 3})
      influence_id.insert(TT[t][f]);
  influence_id.erase(-1);

  // retain_connectivity.
  std::set<int> surround_id;
  std::set_difference(influence_id.begin(), influence_id.end(),
                      delete_id.begin(), delete_id.end(),
                      std::inserter(surround_id, surround_id.end()));


  vecV4i new_tets(2);
//  new_tets[2] << 1, 2, 3, 0;
  new_tets[0] << 1, 3, 4, 0;
  new_tets[1] << 1, 7, 8, 0;


  retain_tetrahedral_adjacency(delete_id, surround_id,new_tets, T,TT,TTif,TTie);
  return 0;
}

template <int enable=1>
int test_extraction_recurse()
{
  using namespace Eigen;
  using namespace std;

  static MatrixXi K_table(4,6);
  K_table << 0,0,2,3,4,4,
  0,0,0,3,4,3,
  0,0,0,0,4,4,
  0,0,0,0,0,5;
  K_table -= MatrixXi::Ones(4,6);

  static std::vector<Eigen::RowVector3i> new_tri;
  struct recurse_extract {
    static void op(int i,int j){
      if(j>= i+2) {
        int k=K_table(i,j);
        recurse_extract::op(i,k);
        recurse_extract::op(k,j);
        new_tri.emplace_back(i,j,k);
      }
    };
  };
  recurse_extract::op(0,5);
//  std::cout<<K_table<<std::endl;
//  for(auto r:new_tri) cout<<r<<endl;
  return 0;
}



template <int enable=1>
int test_single_edge_removal() {
  using namespace Eigen;
  using namespace std;

  vecV3d V(9);
  vecV4i T(7);
  V[0] << 0, 0, 1;
  V[1] << 0, 0, -1;
  V[2] << 0, 10, 0;
  V[3] << -1, 0, 0;
  V[4] << -0.8, -0.5, 0;
  V[5] << -0.2, -0.8, 0;
  V[6] << 0.2, -0.8, 0;
  V[7] << 0.8, -0.5, 0;
  V[8] << 1, 0, 0;

  T[0] << 0, 2, 3, 1;
  T[1] << 0, 3, 4, 1;
  T[2] << 0, 4, 5, 1;
  T[3] << 0, 5, 6, 1;
  T[4] << 0, 6, 7, 1;
  T[5] << 0, 7, 8, 1;
  T[6] << 0, 8, 2, 1;

  vecV4i TT, TTif; TTie_t TTie;
  igl::dev::tetrahedron_tetrahedron_adjacency(T, TT, TTif, TTie);

  auto tet_quality = [&V](int a, int b, int c, int d) {
    return test_utils::quality(V[a],V[b],V[c],V[d]);
  };

  igl::dev::tet_tuple_edge_removal(0,
                                   1,
                                   1,
                                   true,
                                   tet_quality,
                                   T,
                                   TT,
                                   TTif,
                                   TTie );
//  for (auto r:T) cout << r << endl;cout<<endl;
  decltype(TT) nTT;
  decltype(TTif) nTTif;
  decltype(TTie) nTTie;
  igl::dev::tetrahedron_tetrahedron_adjacency(T, nTT, nTTif, nTTie);
  for (int i = 0; i < T.size(); i++) {
    assert(nTT[i] == TT[i]);
    assert(nTTif[i] == TTif[i]);
    assert(nTTie[i] == TTie[i]);
  }
  return 0;
}


template <int enable=1>
int test_simple_face_removal() {


  using namespace std;
  using namespace Eigen;

  std::list<std::tuple<int, int>> dt;
  dt.emplace_back(8, 0);
  std::vector<std::tuple<int, int>> vdt(dt.begin(), dt.end());
  using vecV2d = std::vector<Eigen::RowVector2d>;
//                           Eigen::aligned_allocator<Eigen::RowVector2d>>;
  using vecV3d = std::vector<Eigen::RowVector3d>;
  using vecV3i = std::vector<Eigen::RowVector3i>;
  using vecV4i = std::vector<Eigen::RowVector4i>;
  using T_t = vecV4i;
  using TT_t = T_t;
  using TTie_t = std::vector<Eigen::Matrix<int, 4, 3>>;

  vecV3d V(9);
  vecV4i T(10);
  V[0] << 0.7, -0.1, 1;
  V[1] << 0.7, -0.1, -1;
  V[2] << 0, 1, 0;
  V[3] << -1, 0, 0;
  V[4] << -0.8, -0.5, 0;
  V[5] << -0.2, -0.8, 0;
  V[6] << 0.2, -0.8, 0;
  V[7] << 0.8, -0.5, 0;
  V[8] << 1, 0, 0;

  T[0] << 1, 3, 5, 4;
  T[1] << 3, 5, 4, 0;
  T[2] << 1, 3, 6, 5;
  T[3] << 3, 6, 5, 0;
  T[4] << 1, 3, 7, 6;
  T[5] << 3, 7, 6, 0;
  T[6] << 1, 3, 8, 7;
  T[7] << 3, 8, 7, 0;
  T[8] << 1, 3, 2, 8;
  T[9] << 3, 2, 8, 0;

  vecV4i TT, TTif;
  TTie_t TTie;
  igl::dev::tetrahedron_tetrahedron_adjacency(T, TT, TTif, TTie);

  auto tet_quality = [&V](int a, int b, int c, int d) -> double {
    return -test_utils::quality(V[a], V[b], V[c], V[d]);
  };
  auto correct_orientation = [&V](int a, int b, int c, int d) -> bool {
    return -test_utils::signed_volume(V[a], V[b], V[c], V[d]) > 0;
  };
  std::vector<int> dummy;
  igl::dev::tet_tuple_multi_face_removal(7,
                                         3,
                                         0,
                                         true,
                                         tet_quality,
                                         correct_orientation,
                                         T,
                                         TT,
                                         TTif,
                                         TTie,
                                         dummy);
  decltype(TT) nTT;
  decltype(TTif) nTTif;
  decltype(TTie) nTTie;
  igl::dev::tetrahedron_tetrahedron_adjacency(T, nTT, nTTif, nTTie);
  for (int i = 0; i < T.size(); i++) {
    assert(nTT[i] == TT[i]);
    assert(nTTif[i] == TTif[i]);
    assert(nTTie[i] == TTie[i]);
  }
  return 0;
}

// topological improvement
template <int enable=1>
int test_edge_removal_pass() {


  using namespace std;
  using namespace Eigen;

  std::list<std::tuple<int, int>> dt;
  dt.emplace_back(8, 0);
  std::vector<std::tuple<int, int>> vdt(dt.begin(), dt.end());
  using vecV2d = std::vector<Eigen::RowVector2d>;
//                           Eigen::aligned_allocator<Eigen::RowVector2d>>;
  using vecV3d = std::vector<Eigen::RowVector3d>;
  using vecV3i = std::vector<Eigen::RowVector3i>;
  using vecV4i = std::vector<Eigen::RowVector4i>;
  using T_t = vecV4i;
  using TT_t = T_t;
  using TTie_t = std::vector<Eigen::Matrix<int, 4, 3>>;

  vecV3d V(9);
  vecV4i T(7);
  V[0] << 0, -0, 10;
  V[1] << 0, -0, -1;
  V[2] << 0, 1, 0;
  V[3] << -1, 0, 0;
  V[4] << -0.8, -0.5, 0;
  V[5] << -0.2, -0.8, 0;
  V[6] << 0.2, -0.8, 0;
  V[7] << 0.8, -0.5, 0;
  V[8] << 1, 0, 0;


  T[0] << 0, 2, 3, 1;
  T[1] << 0, 3, 4, 1;
  T[2] << 0, 4, 5, 1;
  T[3] << 0, 5, 6, 1;
  T[4] << 0, 6, 7, 1;
  T[5] << 0, 7, 8, 1;
  T[6] << 0, 8, 2, 1;

  vecV4i TT, TTif;
  TTie_t TTie;
  igl::dev::tetrahedron_tetrahedron_adjacency(T, TT, TTif, TTie);

  auto tet_quality = [&V](int a, int b, int c, int d) -> double {
    return test_utils::quality(V[a], V[b], V[c], V[d]);
  };
  auto correct_orientation = [&V](int a, int b, int c, int d) -> bool {
    return test_utils::signed_volume(V[a], V[b], V[c], V[d]) > 0;
  };

//  for(auto r:T) cout<<r<<":"<<tet_quality(r(0),r(1),r(2),r(3)) <<endl;
//  cout<<endl;
  edge_removal_pass(tet_quality, correct_orientation, T,TT,TTif,TTie);

//  for(auto r:T) cout<<r<<":"<<tet_quality(r(0),r(1),r(2),r(3)) <<endl;
}

// topological improvement
template <int enable=1>
int test_tetmesh_topology_improvement() {

// https://github.com/janba/DSC/blob/master/is_mesh/util.h#L415
  struct test_utils {
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


  using namespace std;
  using namespace Eigen;

  std::list<std::tuple<int, int>> dt;
  dt.emplace_back(8, 0);
  std::vector<std::tuple<int, int>> vdt(dt.begin(), dt.end());
  using vecV2d = std::vector<Eigen::RowVector2d>;
//                           Eigen::aligned_allocator<Eigen::RowVector2d>>;
  using vecV3d = std::vector<Eigen::RowVector3d>;
  using vecV3i = std::vector<Eigen::RowVector3i>;
  using vecV4i = std::vector<Eigen::RowVector4i>;
  using T_t = vecV4i;
  using TT_t = T_t;
  using TTie_t = std::vector<Eigen::Matrix<int, 4, 3>>;

  MatrixXd TV; MatrixXi TTo,TF;
  igl::readMESH("../models/bumpy.mesh",TV,TTo,TF);

  vecV3d V(TV.rows());
  vecV4i T(TTo.rows());

  for(int i=0; i< V.size(); i++) V[i]<<TV.row(i);
  for(int i=0; i< T.size(); i++) T[i]<<TTo.row(i);

  auto tet_quality = [&V](int a, int b, int c, int d) -> double {
    return -test_utils::quality(V[a], V[b], V[c], V[d]);
  };
  auto correct_orientation = [&V](int a, int b, int c, int d) -> bool {
    return -test_utils::signed_volume(V[a], V[b], V[c], V[d]) > 0;
  };

  vecV4i TT, TTif;
  TTie_t TTie;
  igl::dev::tetrahedron_tetrahedron_adjacency(T, TT, TTif, TTie);
//  {double old_q= INFINITY;
//    for(auto r:T) old_q = std::min(old_q, tet_quality(r(0),r(1),r(2),r(3)));
//    cout<<old_q<<endl;}
  // let's try to perturb this.
  int perturb = 0;
  for(auto t=0; t<T.size(); t++)
    for (auto f:{0,1,2,3})
      for(auto e:{0,1,2}) {
        if (igl::dev::tet_tuple_edge_removal_force(t,
                                                   f,
                                                   e,
                                                   true,
                                                   tet_quality,
//                                                         correct_orientation,
                                                   T,
                                                   TT,
                                                   TTif,
                                                   TTie)) {
          perturb++; //t+=10;
          if(perturb == 3)
            goto out;
          break;
        }
      }
out: ;
  /*{
    MatrixXd iV(8,3);
    MatrixXi F(8,3);
    int x=0;
    for (auto t:{0,1398}) {
      for(auto v:{0,1,2,3})
        iV.row(x++) = V[T[t](v)];
    }
    F<< 3, 2, 1,
        2, 3, 0,
        1, 0, 3,
        0, 1, 2,
        7, 6, 5,
        6, 7, 4,
        5, 4, 7,
        4, 5, 6;
    igl::viewer::Viewer vvvv;
    vvvv.data.set_mesh(iV, F);
    igl::writeOBJ("temp.obj",iV,F);
    vvvv.launch();
  }*/

  double old_q = INFINITY, new_q= INFINITY;
  for(auto r:T) old_q = std::min(old_q, tet_quality(r(0),r(1),r(2),r(3)));
//  cout<<old_q<<endl;
  face_removal_pass(tet_quality, correct_orientation, T,TT,TTif,TTie);
  for(auto r:T) new_q = std::min(new_q, tet_quality(r(0),r(1),r(2),r(3)));
//  cout<<new_q<<endl;
}


template <int enable=1>
int test_bumpy_vert_tet_query() {
  using namespace std;
  using namespace Eigen;

  MatrixXd V, Vall; MatrixXi F;
  MatrixXd TV;
  MatrixXi T,TF;

  igl::readMESH("../models/bumpy.mesh", TV,T,TF);

  std::vector<RowVector4i> vT(T.rows()),vTT,vTTif;
  std::vector<Matrix<int,4,3>> vTTie;
  for(int i=0; i<T.rows(); i++) vT[i] = T.row(i);

  igl::dev::tetrahedron_tetrahedron_adjacency(vT,vTT,vTTif,vTTie);

  for(int ti = 0; ti<T.rows(); ti++) {
    for (int fi:{0, 1, 2, 3}) {
      int ei = 0;

      int query_v =
          igl::dev::tet_tuple_get_vert(ti, fi, ei, true, vT, vTT, vTTif, vTTie);

      auto get_neighbor =
          igl::dev::tet_tuple_get_tets_with_vert(ti, fi, ei, true,
                                                 vT, vTT, vTTif, vTTie);

      std::set<int> test_neighbor;
      for (int i = 0; i < T.rows(); i++) {
        for (int j:{0, 1, 2, 3})
          if (T(i, j) == query_v)
            test_neighbor.insert(i);
      }

      assert(get_neighbor == test_neighbor);
    }
  }
  return 0;
}

#include "../src/igl_dev/tet_refine_operations.h"
template <int enable=1>
int test_edge_split()
{
  using namespace Eigen;
  using namespace std;

  using vecV2d = std::vector<Eigen::RowVector2d>;
  using vecV3d = std::vector<Eigen::RowVector3d>;
  using vecV3i = std::vector<Eigen::RowVector3i>;
  using vecV4i = std::vector<Eigen::RowVector4i>;
  using T_t = vecV4i;
  using TT_t = T_t;
  using TTie_t = std::vector<Eigen::Matrix<int, 4, 3>>;

  vecV3d V(9);
  vecV4i T(7);
  V[0] << 0, -0, 1;
  V[1] << 0, -0, -1;
  V[2] << 0, 1, 0;
  V[3] << -1, 0, 0;
  V[4] << -0.8, -0.5, 0;
  V[5] << -0.2, -0.8, 0;
  V[6] << 0.2, -0.8, 0;
  V[7] << 0.8, -0.5, 0;
  V[8] << 1, 0, 0;


  T[0] << 1, 2, 3, 0;
  T[1] << 1, 3, 4, 0;
  T[2] << 1, 4, 5, 0;
  T[3] << 1, 5, 6, 0;
  T[4] << 1, 6, 7, 0;
  T[5] << 1, 7, 8, 0;
  T[6] << 1, 8, 2, 0;

  auto tet_quality = [&V](int a, int b, int c, int d) -> double {
    return -test_utils::quality(V[a], V[b], V[c], V[d]);
  };
//  for(auto r:T) cout<<r<<":"<<tet_quality(r(0),r(1),r(2),r(3)) <<endl;

  vecV4i TT, TTif;
  TTie_t TTie;
  igl::dev::tetrahedron_tetrahedron_adjacency(T, TT, TTif, TTie);

//  cout<< igl::dev::tet_tuple_get_vert(0, 1, 1, true,
//                                      T,TT, TTif, TTie)
//      <<"->"
//      << igl::dev::tet_tuple_get_vert(0, 1, 1, false,
//                                      T,TT, TTif, TTie)<<endl;


  igl::dev::tet_tuple_edge_split(0, 1, 1, true, tet_quality,[](int){return
                                     true;}, V,T,TT, TTif,
                                 TTie);
//  for(auto r:T) cout<<r<<":"<<tet_quality(r(0),r(1),r(2),r(3)) <<endl;
  return 0;
}

template <int enable = 1>
int test_edge_contraction() {

  using namespace Eigen;
  using namespace std;

  using vecV2d = std::vector<Eigen::RowVector2d>;
  using vecV3d = std::vector<Eigen::RowVector3d>;
  using vecV3i = std::vector<Eigen::RowVector3i>;
  using vecV4i = std::vector<Eigen::RowVector4i>;
  using T_t = vecV4i;
  using TT_t = T_t;
  using TTie_t = std::vector<Eigen::Matrix<int, 4, 3>>;

  vecV3d V(9);
  vecV4i T(7);
  V[0] << 0, -0, 1;
  V[1] << 0, -0, -1;
  V[2] << 0, 1, 0;
  V[3] << -1, 0, 0;
  V[4] << -0.8, -0.5, 0;
  V[5] << -0.2, -0.8, 0;
  V[6] << 0.2, -0.8, 0;
  V[7] << 0.8, -0.5, 0;
  V[8] << 1, 0, 0;


  T[0] << 1, 2, 3, 0;
  T[1] << 1, 3, 4, 0;
  T[2] << 1, 4, 5, 0;
  T[3] << 1, 5, 6, 0;
  T[4] << 1, 6, 7, 0;
  T[5] << 1, 7, 8, 0;
  T[6] << 1, 8, 2, 0;

  auto tet_quality = [&V](int a, int b, int c, int d) -> double {
    return -test_utils::quality(V[a], V[b], V[c], V[d]);
  };
//   for(auto r:T) cout<<r<<":"<<tet_quality(r(0),r(1),r(2),r(3)) <<endl;

  vecV4i TT, TTif;
  TTie_t TTie;
  igl::dev::tetrahedron_tetrahedron_adjacency(T, TT, TTif, TTie);
//
//   cout<< igl::dev::tet_tuple_get_vert(0, 1, 1, true,
//                                       T,TT, TTif, TTie)
//       <<"->"
//       << igl::dev::tet_tuple_get_vert(0, 1, 1, false,
//                                       T,TT, TTif, TTie)<<endl;

  std::vector<int> dummy;

  igl::dev::tet_tuple_edge_split(0, 1, 1, true, tet_quality,[](int){return
      true;}, V,T,TT, TTif,
                                 TTie, dummy);
//   for(auto r:T) cout<<r<<":"<<tet_quality(r(0),r(1),r(2),r(3)) <<endl;

  V[1](2) -= 9;
  V[9](2) -= 1;
//   cout<< igl::dev::tet_tuple_get_vert(7, 1, 1, true,
//                                       T,TT, TTif, TTie)
//       <<"->"
//       << igl::dev::tet_tuple_get_vert(7, 1, 1, false,
//                                       T,TT, TTif, TTie)<<endl;
  igl::dev::tet_tuple_edge_split(7, 1, 1, true, tet_quality,[](int){return
                                     true;}, V,T,TT, TTif,
                                 TTie,dummy);

  V[9](2) += 0.9;
  V[1](2) += 9;
//  for(auto r:T) cout<<r<<endl;
  igl::dev::tet_tuple_edge_contraction(7,
                                       1,
                                       1,
                                       true,
                                       tet_quality,[](int){return
          true;},
                                       T,
                                       TT,
                                       TTif,
                                       TTie,
                                       dummy);

//  for(auto r:T) cout<<r<<":"<<tet_quality(r(0),r(1),r(2),r(3)) <<endl;
  return 0;
}
/*
int test_all() {
  test_edge_split<1>();
  test_edge_contraction();
  test_bumpy_vert_tet_query();
  test_tetmesh_topology_improvement();
  test_simple_face_removal();
  test_edge_removal_pass();
  test_single_edge_removal();
  test_extraction_recurse();
  test_retain_fix_end();
  test_retain_tetrahedron_adjacency();
  test_tetrahedron_adjacency();
}
*/