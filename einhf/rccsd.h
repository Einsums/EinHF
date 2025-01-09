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

#include "Einsums/Einsums.hpp"
#include <deque>
#include <vector>

#include "psi4/libmints/wavefunction.h"
#include "psi4/psi4-dec.h"
#include "rhf.h"
#include "rmp2.h"

namespace psi {
// Forward declare several variables
class Options;
class JK;

namespace einhf {

struct CCSScaleFunction
    : public virtual einsums::tensor_base::FunctionTensorBase<double, 2>,
      virtual einsums::tensor_base::CoreTensorBase {
private:
  einsums::Tensor<double, 1> _evalsi, _evalsa;

public:
  CCSScaleFunction(const MP2ScaleFunction &) = default;

  CCSScaleFunction(std::string name,
                   const einsums::TensorView<double, 1> &evalsi,
                   const einsums::TensorView<double, 1> &evalsa)
      : einsums::tensor_props::FunctionTensorBase<double, 2>(
            name, evalsi.dim(0), evalsa.dim(0)),
        _evalsi(evalsi), _evalsa(evalsa) {}

  double call(const std::array<int, 2> &inds) const override {
    return 1.0 / ((_evalsi)(inds[0]) - (_evalsa)(inds[1]));
  }
};

struct RCCSScaleTensor final
    : public virtual einsums::tensor_props::TiledTensorBase<double, 2,
                                                            CCSScaleFunction>,
      virtual einsums::tensor_props::CoreTensorBase {
private:
  const einsums::Tensor<double, 1> *_evals;
  std::vector<int> _irrep_offsets, _irrep_sizes;
  std::vector<std::string> _irrep_names;

  virtual void add_tile(std::array<int, 2> pos) override {
    std::string tile_name = name() + " - (";
    einsums::Dim<2> dims{};

    for (int i = 0; i < 2; i++) {
      tile_name += _irrep_names[pos[i]];
      dims[i] = this->_tile_sizes[i][pos[i]];
      if (i != 1) {
        tile_name += ", ";
      }
    }
    tile_name += ")";

    auto viewi = (*_evals)(einsums::Range{_irrep_offsets[pos[0]],
                                          _irrep_offsets[pos[0]] + dims[0]});
    auto viewa = (*_evals)(
        einsums::Range{_irrep_offsets[pos[1]] + _irrep_sizes[pos[1]] - dims[1],
                       _irrep_offsets[pos[1]] + _irrep_sizes[pos[1]]});

    if (viewi.dim(0) != 0 && viewa.dim(0) != 0) {
      auto piece = CCSScaleFunction(tile_name, viewi, viewa);

      this->_tiles.emplace(pos, piece);
    }
  }

public:
  RCCSScaleTensor() = default;

  RCCSScaleTensor(std::string name, std::vector<int> occupied,
                  std::vector<int> unoccupied, std::vector<int> irrep_offsets,
                  std::vector<int> irrep_sizes,
                  std::vector<std::string> irrep_names,
                  const einsums::Tensor<double, 1> *evals)
      : einsums::tensor_props::TiledTensorBase<double, 2, CCSScaleFunction>(
            name, occupied, unoccupied),
        _irrep_offsets(irrep_offsets), _irrep_sizes(irrep_sizes),
        _irrep_names(irrep_names), _evals{evals} {}

  const einsums::Tensor<double, 1> *get_evals() const { return _evals; }
  std::vector<int> get_irrep_offsets() const { return _irrep_offsets; }
  std::vector<int> get_irrep_sizes() const { return _irrep_sizes; }
  std::vector<std::string> get_irrep_names() const { return _irrep_names; }
};

class EinsumsRCCSD : public Wavefunction {
public:
  /// The constuctor
  EinsumsRCCSD(std::shared_ptr<EinsumsRMP2> ref_wfn, Options &options);
  /// The destuctor
  ~EinsumsRCCSD();
  /// Computes the SCF energy, and returns it
  virtual double compute_energy() override;

  virtual void print_header();

  double T1_diagnostic() const;

  const einsums::BlockTensor<double, 2> &getS() const {
    return mp2_wfn_->getS();
  }
  const einsums::BlockTensor<double, 2> &getF() const {
    return mp2_wfn_->getF();
  }
  const einsums::BlockTensor<double, 2> &getFt() const {
    return mp2_wfn_->getFt();
  }
  const einsums::BlockTensor<double, 2> &getC() const {
    return mp2_wfn_->getC();
  }
  const einsums::BlockTensor<double, 2> &getCocc() const {
    return mp2_wfn_->getCocc();
  }
  const einsums::BlockTensor<double, 2> &getD() const {
    return mp2_wfn_->getD();
  }
  const einsums::Tensor<double, 1> &getEvals() const {
    return mp2_wfn_->getEvals();
  }
  const einsums::TiledTensor<double, 4> &getTeiAnti() const {
    return tei_anti_;
  }
  const einsums::TiledTensor<double, 4> &getTeiTrans() const {
    return mp2_wfn_->getTeiTrans();
  }
  const einsums::TiledTensor<double, 4> &getMP2Amps() const {
    return mp2_wfn_->getMP2Amps();
  }
  const RMP2ScaleTensor &getDenominator() const {
    return mp2_wfn_->getDenominator();
  }
  const RCCSScaleTensor &getT1Den() const { return t1_den_; }
  const einsums::TiledTensor<double, 2> &getT1Amps() const { return t1_amps_; }
  const einsums::TiledTensor<double, 4> &getT2Amps() const { return t2_amps_; }

  einsums::BlockTensor<double, 2> &getS() { return mp2_wfn_->getS(); }
  einsums::BlockTensor<double, 2> &getF() { return mp2_wfn_->getF(); }
  einsums::BlockTensor<double, 2> &getFt() { return mp2_wfn_->getFt(); }
  einsums::BlockTensor<double, 2> &getC() { return mp2_wfn_->getC(); }
  einsums::BlockTensor<double, 2> &getCocc() { return mp2_wfn_->getCocc(); }
  einsums::BlockTensor<double, 2> &getD() { return mp2_wfn_->getD(); }
  einsums::Tensor<double, 1> &getEvals() { return mp2_wfn_->getEvals(); }
  einsums::TiledTensor<double, 4> &getTeiAnti() { return tei_anti; }
  einsums::TiledTensor<double, 4> &getTeiTrans() {
    return mp2_wfn_->getTeiTrans();
  }
  einsums::TiledTensor<double, 4> &getMP2Amps() {
    return mp2_wfn_->getMP2Amps();
  }
  RMP2ScaleTensor &getDenominator() { return mp2_wfn_->getDenominator(); }
  RCCSScaleTensor &getT1Den() { return t1_den_; }

  einsums::TiledTensor<double, 2> &getT1Amps() { return t1_amps_; }
  einsums::TiledTensor<double, 4> &getT2Amps() { return t2_amps_; }

protected:
  ///
  std::shared_ptr<EinsumsRMP2> mp2_wfn_;
  /// The amount of information to print to the output file
  int print_;
  /// The number of electrons. Used for calculating the T1 diagnostic.
  int nelec_;

  double e_convergence_, d_convergence_;
  int maxiter_;

  /// The occupation per irrep.
  std::vector<int> occ_per_irrep_, unocc_per_irrep_;
  /// The sizes of each irrep.
  std::vector<int> irrep_sizes_, irrep_offsets_;
  /// The names of the irreps.
  std::vector<std::string> irrep_names_;

  /// The number of symmetrized spin orbitals
  int nso_;

  /// The T1 amplitudes.
  einsums::TiledTensor<double, 2> t1_amps_;
  /// The T2 amplitudes.
  einsums::TiledTensor<double, 4> t2_amps_;
  /// The anti-symmetrized two-electron integrals. In physicists' notation.
  einsums::TiledTensor<double, 4> tei_anti_oooo, tei_anti_ooov, tei_anti_oovv, tei_anti_ovov, tei_anti_ovvv, tei_anti_vvvv;
  /// The T1 denominator. T2 is handled by the stored EinsumsRMP2 wavefunction
  /// object.
  RCCSScaleTensor t1_den_;

  void init_integrals();
};

} // namespace einhf
} // namespace psi
