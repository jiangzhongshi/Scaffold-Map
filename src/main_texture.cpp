#include "ReWeightedARAP.h"
#include "StateManager.h"
#include "UI/TextureGUI.h"
#include "util/triangle_utils.h"
#include <igl/Timer.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <memory>
#include <CLI/CLI.hpp>

int main(int argc, char* argv[]) {
  CLI::App command_line{"Scaffold Mapping: Parameterization"};
  std::string filename = "";
  filename = "/Users/zhongshi/Workspace/Scaffold-Map/camel_b.obj";
  command_line.add_option("-m,--mesh", filename, "Mesh path")->check(CLI::ExistingFile);
  bool show_gui = true;
  command_line.add_option("-g,--gui", show_gui, "Show GUI");
  try {
      command_line.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
      return command_line.exit(e);
  }

  StateManager s_(DemoType::PACKING, filename);

TextureGUI gui(s_);
gui.launch();
  return 0;
}
