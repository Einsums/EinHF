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

#include <Einsums/TensorBase.hpp>
#include <Einsums/Tensor.hpp>
#include <vector>

namespace psi {

namespace einhf {

template <size_t OccUnocc>
struct ScaleFunction
    : public virtual einsums::tensor_base::FunctionTensor<double,
                                                               2 * OccUnocc>,
      virtual einsums::tensor_base::CoreTensor {
protected:
  std::array<einsums::TensorView<double, 1>, OccUnocc> _evals_occ, _evals_unocc;

public:
  ScaleFunction() = default;
  ScaleFunction(const ScaleFunction &) = default;

  ScaleFunction(
      std::string name,
      const std::array<einsums::TensorView<double, 1>, OccUnocc> &evals_occ,
      const std::array<einsums::TensorView<double, 1>, OccUnocc> &evals_unocc)
      : _evals_occ(evals_occ), _evals_unocc(evals_unocc) {
    std::array<size_t, 2 * OccUnocc> dims;

    for (int i = 0; i < OccUnocc; i++) {
      dims[i] = evals_occ[i].dim(0);
      dims[i + OccUnocc] = evals_unocc[i].dim(0);
    }

    einsums::tensor_base::FunctionTensor<double, 2 * OccUnocc>(name, dims);
  }

  double call(const std::array<int, 2 * OccUnocc> &ind) {
    double sum = 0.0;

    for(int i = 0; i < OccUnocc; i++) {
        sum += _evals_occ[i](ind[i]) - _evals_unocc[i](ind[OccUnocc + i]);
    }
    return 1.0 / sum;
  }
};

template<size_t OccUnocc>


} // namespace einhf
} // namespace psi