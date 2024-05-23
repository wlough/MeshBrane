/**
 * @file main.cpp
 * @brief main function for the cbrane project.
 */
#include "MeshLoader.hpp"
#include "brane_utils.hpp"
#include <iostream>
//
//
#include <vtkActor.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolygon.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

void visualizeSurface(const VertexFaceList &vfl) {
  // Create a vtkPoints object and store the points in it
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  for (const auto &vertex : vfl.vertices) {
    points->InsertNextPoint(vertex.data());
  }

  // Create a vtkCellArray to store the faces
  vtkSmartPointer<vtkCellArray> polygons = vtkSmartPointer<vtkCellArray>::New();
  for (const auto &face : vfl.faces) {
    vtkSmartPointer<vtkPolygon> polygon = vtkSmartPointer<vtkPolygon>::New();
    polygon->GetPointIds()->SetNumberOfIds(3);
    for (int i = 0; i < 3; ++i) {
      polygon->GetPointIds()->SetId(i, face[i]);
    }
    polygons->InsertNextCell(polygon);
  }

  // Create a PolyData
  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
  polyData->SetPoints(points);
  polyData->SetPolys(polygons);

  // Visualize
  vtkSmartPointer<vtkPolyDataMapper> mapper =
      vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputData(polyData);

  vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);

  vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renderWindow =
      vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
      vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renderWindowInteractor->SetRenderWindow(renderWindow);

  renderer->AddActor(actor);
  renderer->SetBackground(.3, .6, .3); // Background color green

  renderWindow->Render();
  renderWindowInteractor->Start();
}

int main(int argc, char *argv[]) {
  std::string filepath = "./data/ply_files/dumbbell.ply";
  std::pair<std::vector<std::array<double, 3>>,
            std::vector<std::array<uint32_t, 3>>>
      vf_mesh = load_vertex_face_list_from_ply(filepath);

  return EXIT_SUCCESS;
}