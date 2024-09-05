// half_edge_base_utils.cpp

#include "half_edge_base_utils.hpp"

int find_halfedge_index_of_twin(const MatrixXi &H, int h) {
  Eigen::RowVector2i hedge_twin = H.row(h).reverse();
  for (int i = 0; i < H.rows(); ++i) {
    if (H.row(i) == hedge_twin) {
      return i;
    }
  }
  return -1;
}

MatrixXi find_V_of_F(const MatrixXd &xyz_coord_V, const VectorXi &h_out_V,
                     const VectorXi &v_origin_H, const VectorXi &h_next_H,
                     const VectorXi &h_twin_H, const VectorXi &f_left_H,
                     const VectorXi &h_bound_F, const VectorXi &h_right_B) {
  int Nf = h_bound_F.size();
  MatrixXi V_of_F(Nf, 3);
  for (int f = 0; f < Nf; ++f) {
    int h = h_bound_F[f];
    int h_start = h;
    int _v = 0;
    while (true) {
      V_of_F(f, _v) = v_origin_H[h];
      h = h_next_H[h];
      _v += 1;
      if (h == h_start) {
        break;
      }
    }
  }
  return V_of_F;
}

VectorXi find_h_right_B(const MatrixXd &xyz_coord_V, const VectorXi &h_out_V,
                        VectorXi &v_origin_H, const VectorXi &h_next_H,
                        const VectorXi &h_twin_H, VectorXi &f_left_H,
                        const VectorXi &h_bound_F) {
  int Nhedges = v_origin_H.size();
  set<int> H_need2visit;
  for (int h = 0; h < Nhedges; ++h) {
    if (f_left_H[h] < 0) {
      H_need2visit.insert(h);
    }
  }
  vector<int> _h_right_B;
  while (!H_need2visit.empty()) {
    int b = _h_right_B.size();
    int h_start = *H_need2visit.begin();
    H_need2visit.erase(H_need2visit.begin());
    f_left_H[h_start] = -(b + 1);
    int h = h_next_H[h_start];
    _h_right_B.push_back(h);
    while (h != h_start) {
      H_need2visit.erase(h);
      f_left_H[h] = -(b + 1);
      h = h_next_H[h];
    }
  }
  VectorXi h_right_B =
      Eigen::Map<VectorXi>(_h_right_B.data(), _h_right_B.size());
  return h_right_B;
}

std::tuple<MatrixXd, VectorXi, VectorXi, VectorXi, VectorXi, VectorXi, VectorXi,
           VectorXi>
vf_samples_to_he_samples(const MatrixXd &xyz_coord_V,
                         const MatrixXi &vvv_of_F) {
  int Nfaces = vvv_of_F.rows();
  int Nvertices = xyz_coord_V.rows();
  int _Nhedges = 3 * Nfaces * 2;
  MatrixXi _H = MatrixXi::Zero(_Nhedges, 2);
  VectorXi h_out_V = VectorXi::Constant(Nvertices, -1);
  VectorXi _v_origin_H = VectorXi::Zero(_Nhedges);
  VectorXi _h_next_H = VectorXi::Constant(_Nhedges, -1);
  VectorXi _f_left_H = VectorXi::Zero(_Nhedges);
  VectorXi h_bound_F = VectorXi::Zero(Nfaces);

  for (int f = 0; f < Nfaces; ++f) {
    h_bound_F[f] = 3 * f;
    for (int i = 0; i < 3; ++i) {
      int h = 3 * f + i;
      int h_next = 3 * f + (i + 1) % 3;
      int v0 = vvv_of_F(f, i);
      int v1 = vvv_of_F(f, (i + 1) % 3);
      _H.row(h) << v0, v1;
      _v_origin_H[h] = v0;
      _f_left_H[h] = f;
      _h_next_H[h] = h_next;
      if (h_out_V[v0] == -1) {
        h_out_V[v0] = h;
      }
    }
  }

  int h_count = 3 * Nfaces;
  set<int> need_twins;
  for (int i = 0; i < h_count; ++i) {
    need_twins.insert(i);
  }
  set<int> need_next;
  VectorXi _h_twin_H = VectorXi::Constant(_Nhedges, -2);

  while (!need_twins.empty()) {
    int h = *need_twins.begin();
    need_twins.erase(need_twins.begin());
    if (_h_twin_H[h] == -2) {
      int h_twin = find_halfedge_index_of_twin(_H, h);
      if (h_twin == -1) {
        h_twin = h_count;
        h_count += 1;
        int v0 = _H(h, 0);
        int v1 = _H(h, 1);
        _H.row(h_twin) << v1, v0;
        _v_origin_H[h_twin] = v1;
        need_next.insert(h_twin);
        _h_twin_H[h] = h_twin;
        _h_twin_H[h_twin] = h;
        _f_left_H[h_twin] = -1;
      } else {
        _h_twin_H[h] = h_twin;
        _h_twin_H[h_twin] = h;
        need_twins.erase(h_twin);
      }
    }
  }

  int Nhedges = h_count;
  VectorXi v_origin_H = _v_origin_H.head(Nhedges);
  VectorXi h_next_H = _h_next_H.head(Nhedges);
  VectorXi f_left_H = _f_left_H.head(Nhedges);
  VectorXi h_twin_H = _h_twin_H.head(Nhedges);

  while (!need_next.empty()) {
    int h = *need_next.begin();
    need_next.erase(need_next.begin());
    int h_next = h_twin_H[h];
    while (f_left_H[h_next] != -1) {
      h_next = h_twin_H[h_next_H[h_next_H[h_next]]];
    }
    h_next_H[h] = h_next;
  }

  set<int> H_need2visit;
  for (int h = 0; h < Nhedges; ++h) {
    if (f_left_H[h] == -1) {
      H_need2visit.insert(h);
    }
  }
  vector<int> _h_right_B;
  while (!H_need2visit.empty()) {
    int b = _h_right_B.size();
    int h_start = *H_need2visit.begin();
    H_need2visit.erase(H_need2visit.begin());
    f_left_H[h_start] = -(b + 1);
    int h = h_next_H[h_start];
    _h_right_B.push_back(h);
    while (h != h_start) {
      H_need2visit.erase(h);
      f_left_H[h] = -(b + 1);
      h = h_next_H[h];
    }
  }
  VectorXi h_right_B =
      Eigen::Map<VectorXi>(_h_right_B.data(), _h_right_B.size());

  return std::make_tuple(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
                         f_left_H, h_bound_F, h_right_B);
}