//
// Created by Zhongshi Jiang on 1/20/17.
//

#ifndef SCAFFOLD_TEST_STATEMANAGER_H
#define SCAFFOLD_TEST_STATEMANAGER_H

#include "ReWeightedARAP.h"
#include "ScafData.h"
#include <string>
#include <memory>
#include <iostream>

enum DemoType{
PACKING, FLOW, PARAM, BARS
};

struct StateManager
{
  StateManager(){} //empty constructor
  StateManager(DemoType demo_type, std::string filename, std::string target_file=""); //new from mesh_file

  friend std::ostream& operator<<(std::ostream& os, const StateManager& sm);

  void load(std::string);
  void save(std::string);

  DemoType demo_type;

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
