//
// Created by Zhongshi Jiang on 2/9/17.
//


#include "tetgenio_parser.h"
#include <igl/copyleft/tetgen/mesh_to_tetgenio.h>
#include <igl/copyleft/tetgen/tetgenio_to_tetmesh.h>
#include <igl/matrix_to_list.h>
#include <igl/list_to_matrix.h>
#include <igl/boundary_facets.h>
#include <iostream>

int igl::dev::tetgen::tetrahedralize(const std::vector<std::vector<REAL> > &V,
                                          const std::vector<std::vector<int> > &F,
                                          const std::vector<std::vector<REAL> > &H,
                                          const std::string switches,
                                          std::vector<std::vector<REAL> > &TV,
                                          std::vector<std::vector<int> > &TT,
                                          std::vector<std::vector<int> > &TF,
                                          std::vector<int> &TR) {
  using namespace std;
  tetgenio in, out;
  igl::copyleft::tetgen::mesh_to_tetgenio(V, F, in);

  in.numberofholes = static_cast<int>(H.size());
  in.holelist = new REAL[in.numberofholes * 3];
  for (int i = 0; i < in.numberofholes; i++) {
    assert(H[i].size() == 3);
    in.holelist[i * 3 + 0] = H[i][0];
    in.holelist[i * 3 + 1] = H[i][1];
    in.holelist[i * 3 + 2] = H[i][2];
  }

//  in.numberofregions = 2;
//  in.regionlist = new REAL[2*3]{0,0,0,9.8,9.8,9.8};

  try {
    char *cswitches = new char[switches.size() + 1];
    std::strcpy(cswitches, switches.c_str());
    ::tetrahedralize(cswitches, &in, &out);
    delete[] cswitches;
  } catch (int e) {
    cerr << "^" << __FUNCTION__ << ": TETGEN CRASHED... KABOOOM!!!" << endl;
    return 1;
  }
  if (out.numberoftetrahedra == 0) {
    cerr << "^" << __FUNCTION__ << ": Tetgen failed to create tets" << endl;
    return 2;
  }
  if (!igl::copyleft::tetgen::tetgenio_to_tetmesh(out, TV, TT, TF)) {
    return -1;
  }
  TR.clear();
  for(int i=0; i<out.numberoftetrahedronattributes; i++)
  {
    TR.push_back(out.tetrahedronattributelist[i]);
  }
  //boundary_facets(TT,TF);
  return 0;
}

template<
    typename DerivedV,
    typename DerivedF,
    typename DerivedH,
    typename DerivedTV,
    typename DerivedTT,
    typename DerivedTF,
    typename DerivedTR>
int igl::dev::tetgen::tetrahedralize(const Eigen::PlainObjectBase<
    DerivedV> &V,
                                          const Eigen::PlainObjectBase<
                                              DerivedF> &F,
                                          const Eigen::PlainObjectBase<
                                              DerivedH> &H,
                                          const std::string switches,
                                          Eigen::PlainObjectBase<DerivedTV> &TV,
                                          Eigen::PlainObjectBase<DerivedTT> &TT,
                                          Eigen::PlainObjectBase<DerivedTF> &TF,
                                          Eigen::PlainObjectBase<DerivedTR> &TR) {
  using namespace std;
  vector<vector<REAL> > vV, vH, vTV;
  vector<int> vTR;
  vector<vector<int> > vF, vTT, vTF;
  matrix_to_list(V, vV);
  matrix_to_list(F, vF);
  matrix_to_list(H, vH);

  int e = tetrahedralize(vV,
                         vF,
                         vH,
                         switches,
                         vTV,
                         vTT,
                         vTF,
                         vTR);
  if (e == 0) {
    if (!list_to_matrix(vTV, TV)) {
      return 3;
    }
    bool TT_rect = list_to_matrix(vTT, TT);
    if (!TT_rect) {
      return 3;
    }
    bool TF_rect = list_to_matrix(vTF, TF);
    if (!TF_rect) {
      return 3;
    }
    if (!list_to_matrix(vTR, TR)) {
      return 3;
    }
  }
  return e;
}

template int igl::dev::tetgen::tetrahedralize<Eigen::Matrix<double, -1, -1, 0,
                                                          -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);