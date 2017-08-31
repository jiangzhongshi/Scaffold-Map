//
// Created by Zhongshi Jiang on 2/9/17.
//

#ifndef SCAFFOLD_TEST_TETGENIO_PARSER_H
#define SCAFFOLD_TEST_TETGENIO_PARSER_H

#ifndef TETLIBRARY
#  define TETLIBRARY
#endif
#include <igl/igl_inline.h>
#include "tetgen.h" // Defined tetgenio, REAL
#include <vector>
#include <Eigen/Core>


namespace igl {
namespace dev {
namespace tetgen {
// Load a vertex list and face list into a tetgenio object
// Inputs:
//   V  #V by 3 vertex position list
//   F  #F list of polygon face indices into V (0-indexed)
// Outputs:
//   in  tetgenio input object
// Returns true on success, false on error

IGL_INLINE int tetrahedralize(const std::vector<std::vector<REAL> > &V,
                              const std::vector<std::vector<int> > &F,
                              const std::vector<std::vector<REAL> > &H,
                              const std::string switches,
                              std::vector<std::vector<REAL> > &TV,
                              std::vector<std::vector<int> > &TT,
                              std::vector<std::vector<int> > &TF,
                              std::vector<int> &TR);

template<
    typename DerivedV,
    typename DerivedF,
    typename DerivedH,
    typename DerivedTV,
    typename DerivedTT,
    typename DerivedTF,
    typename DerivedTR>
IGL_INLINE int tetrahedralize(const Eigen::PlainObjectBase<
    DerivedV> &V,
                              const Eigen::PlainObjectBase<
                                  DerivedF> &F,
                              const Eigen::PlainObjectBase<
                                  DerivedH> &H,
                              const std::string switches,
                              Eigen::PlainObjectBase<DerivedTV> &TV,
                              Eigen::PlainObjectBase<DerivedTT> &TT,
                              Eigen::PlainObjectBase<DerivedTF> &TF,
                              Eigen::PlainObjectBase<DerivedTR> &TR);
}
}
}



#endif //SCAFFOLD_TEST_TETGENIO_PARSER_H
