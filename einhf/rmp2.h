/*
 * @BEGIN LICENSE
 *
 * einhf by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2023 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Psi4; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#pragma once

#include "einsums.hpp"
#include <deque>
#include <vector>

#include "psi4/libmints/wavefunction.h"
#include "psi4/psi4-dec.h"
#include "rhf.h"

namespace psi {
// Forward declare several variables
class Options;
class JK;

namespace einhf {

struct MP2ScaleFunction
    : public virtual einsums::tensor_props::FunctionTensorBase<double, 4>,
      virtual einsums::tensor_props::CoreTensorBase {
private:
  einsums::Tensor<double, 1> _evalsi, _evalsa, _evalsj, _evalsb;

public:
  MP2ScaleFunction(const MP2ScaleFunction &) = default;

  MP2ScaleFunction(std::string name,
                   const einsums::TensorView<double, 1> &evalsi,
                   const einsums::TensorView<double, 1> &evalsa,
                   const einsums::TensorView<double, 1> &evalsj,
                   const einsums::TensorView<double, 1> &evalsb)
      : einsums::tensor_props::FunctionTensorBase<double, 4>(
            name, evalsi.dim(0), evalsa.dim(0), evalsj.dim(0), evalsb.dim(0)),
        _evalsi(evalsi), _evalsa(evalsa), _evalsj(evalsj), _evalsb(evalsb) {}

  double call(const std::array<int, 4> &inds) const override {
    return 1.0 / ((_evalsi)(inds[0]) + (_evalsj)(inds[2]) - (_evalsa)(inds[1]) -
                  (_evalsb)(inds[3]));
  }
};

struct RMP2ScaleTensor final
    : public virtual einsums::tensor_props::TiledTensorBase<double, 4,
                                                            MP2ScaleFunction>,
      virtual einsums::tensor_props::CoreTensorBase {
private:
  const einsums::Tensor<double, 1> *_evals;
  std::vector<int> _irrep_offsets, _irrep_sizes;
  std::vector<std::string> _irrep_names;

  virtual void add_tile(std::array<int, 4> pos) override {
    std::string tile_name = name() + " - (";
    einsums::Dim<4> dims{};

    for (int i = 0; i < 4; i++) {
      tile_name += _irrep_names[pos[i]];
      dims[i] = this->_tile_sizes[i][pos[i]];
      if (i != 3) {
        tile_name += ", ";
      }
    }
    tile_name += ")";

    auto viewi = (*_evals)(einsums::Range{_irrep_offsets[pos[0]],
                                          _irrep_offsets[pos[0]] + dims[0]});
    auto viewa = (*_evals)(
        einsums::Range{_irrep_offsets[pos[1]] + _irrep_sizes[pos[1]] - dims[1],
                       _irrep_offsets[pos[1]] + _irrep_sizes[pos[1]]});
    auto viewj = (*_evals)(einsums::Range{_irrep_offsets[pos[2]],
                                          _irrep_offsets[pos[2]] + dims[2]});
    auto viewb = (*_evals)(
        einsums::Range{_irrep_offsets[pos[3]] + _irrep_sizes[pos[3]] - dims[3],
                       _irrep_offsets[pos[3]] + _irrep_sizes[pos[3]]});

    if (viewi.dim(0) != 0 && viewa.dim(0) != 0 && viewj.dim(0) != 0 &&
        viewb.dim(0) != 0) {
      auto piece = MP2ScaleFunction(tile_name, viewi, viewa, viewj, viewb);

      this->_tiles.emplace(pos, piece);
    }
  }

public:
  RMP2ScaleTensor() = default;

  // RMP2ScaleTensor(const RMP2ScaleTensor &) = default;

  RMP2ScaleTensor(std::string name, std::vector<int> occupied,
                  std::vector<int> unoccupied, std::vector<int> irrep_offsets,
                  std::vector<int> irrep_sizes,
                  std::vector<std::string> irrep_names,
                  const einsums::Tensor<double, 1> *evals)
      : einsums::tensor_props::TiledTensorBase<double, 4, MP2ScaleFunction>(
            name, occupied, unoccupied, occupied, unoccupied),
        _irrep_offsets(irrep_offsets), _irrep_sizes(irrep_sizes),
        _irrep_names(irrep_names), _evals{evals} {}

  const einsums::Tensor<double, 1> *get_evals() const { return _evals; }
  std::vector<int> get_irrep_offsets() const { return _irrep_offsets; }
  std::vector<int> get_irrep_sizes() const { return _irrep_sizes; }
  std::vector<std::string> get_irrep_names() const { return _irrep_names; }
};

class EinsumsRMP2 : public Wavefunction {
public:
  /// The constuctor
  EinsumsRMP2(std::shared_ptr<EinsumsRHF> ref_wfn, Options &options);
  /// The destuctor
  ~EinsumsRMP2();
  /// Computes the SCF energy, and returns it
  virtual double compute_energy() override;

  virtual void print_header();

  const einsums::BlockTensor<double, 2> &getS() const { return S_; }
  const einsums::BlockTensor<double, 2> &getF() const { return F_; }
  const einsums::BlockTensor<double, 2> &getFt() const { return Ft_; }
  const einsums::BlockTensor<double, 2> &getC() const { return C_; }
  const einsums::BlockTensor<double, 2> &getCocc() const { return Cocc_; }
  const einsums::BlockTensor<double, 2> &getD() const { return D_; }
  const einsums::Tensor<double, 1> &getEvals() const { return evals_; }
  const einsums::TiledTensor<double, 4> &getTei() const { return tei_; }
  const einsums::TiledTensor<double, 4> &getTeiTrans() const { return teit_; }
  const einsums::TiledTensor<double, 4> &getMP2Amps() const {
    return MP2_amps_;
  }
  const RMP2ScaleTensor &getDenominator() const { return denominator_; }

  einsums::BlockTensor<double, 2> &getS() { return S_; }
  einsums::BlockTensor<double, 2> &getF() { return F_; }
  einsums::BlockTensor<double, 2> &getFt() { return Ft_; }
  einsums::BlockTensor<double, 2> &getC() { return C_; }
  einsums::BlockTensor<double, 2> &getCocc() { return Cocc_; }
  einsums::BlockTensor<double, 2> &getD() { return D_; }
  einsums::Tensor<double, 1> &getEvals() { return evals_; }
  einsums::TiledTensor<double, 4> &getTei() { return tei_; }
  einsums::TiledTensor<double, 4> &getTeiTrans() { return teit_; }
  einsums::TiledTensor<double, 4> &getMP2Amps() { return MP2_amps_; }
  RMP2ScaleTensor &getDenominator() { return denominator_; }

  double getSCFEnergy() { return scf_energy_; }

protected:
  /// The amount of information to print to the output file
  int print_;
  /// The number of doubly occupied orbitals
  int ndocc_;
  /// The SCF energy.
  double scf_energy_;

  /// The occupation per irrep.
  std::vector<int> occ_per_irrep_, unocc_per_irrep_;
  /// The sizes of each irrep.
  std::vector<int> irrep_sizes_, irrep_offsets_;
  /// The names of the irreps.
  std::vector<std::string> irrep_names_;

  /// The number of symmetrized spin orbitals
  int nso_;
  /// The overlap matrix
  einsums::BlockTensor<double, 2> S_;
  /// The Fock Matrix
  einsums::BlockTensor<double, 2> F_;
  /// The transformed Fock matrix
  einsums::BlockTensor<double, 2> Ft_;
  /// The MO coefficients
  einsums::BlockTensor<double, 2> C_;
  /// The occupied MO coefficients
  einsums::BlockTensor<double, 2> Cocc_;
  /// The density matrix
  einsums::BlockTensor<double, 2> D_;
  /// The orbital energies.
  einsums::Tensor<double, 1> evals_;
  /// The MO-basis two-electron integrals
  einsums::TiledTensor<double, 4> tei_;
  /// Only the (ia|jb) part of the two-electron integrals
  einsums::TiledTensor<double, 4> teit_;
  /// MP2 amplitudes.
  einsums::TiledTensor<double, 4> MP2_amps_;
  /// Function tensor for the MP2 denominator.
  RMP2ScaleTensor denominator_;
  /// Sets up the integrals object
  void init_integrals();

  void set_tile(const einsums::TiledTensor<double, 4> &temp, int i, int a,
                int j, int b);
};

} // namespace einhf
} // namespace psi
