//
// Created by Zhongshi Jiang on 1/20/17.
//

#ifndef SCAFFOLD_TEST_STATEMANAGER_H
#define SCAFFOLD_TEST_STATEMANAGER_H

#include "ReWeightedARAP.h"
#include "ScafData.h"
#include "BFGS/SequentialLBFGS.h"

#include <string>
#include <memory>
#include <iostream>


struct StateManager
{
  StateManager(){} //empty constructor
  StateManager(std::string filename); //new from mesh_file

  friend std::ostream& operator<<(std::ostream& os, const StateManager& sm);

  void load(std::string);
  void save(std::string);

  //data
  ScafData scaf_data;
  int iter_count = 0;
  std::string model_file = std::string("NA");

  std::shared_ptr<ReWeightedARAP> ws_solver = nullptr;

  //tweaking
  bool optimize_scaffold = true;
  bool predict_reference = false;
  bool fix_reference = false;

  //display

};


#endif //SCAFFOLD_TEST_STATEMANAGER_H
