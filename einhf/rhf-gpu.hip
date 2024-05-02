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

#include "rhf-gpu.h"

#include "einsums.hpp"

#include "psi4/libfock/jk.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/sobasis.h"
#include "psi4/libmints/vector.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"
#include "psi4/psi4-dec.h"
#include <LinearAlgebra.hpp>
#include <_Common.hpp>
#include <_Index.hpp>
#include <cmath>

static std::string to_lower(const std::string &str) {
  std::string out(str);
  std::transform(str.begin(), str.end(), out.begin(),
                 [](char c) { return std::tolower(c); });
  return out;
}

namespace psi {
namespace einhf {

GPUEinsumsSCF::GPUEinsumsSCF(SharedWavefunction ref_wfn,
                             const std::shared_ptr<SuperFunctional> &functional,
                             Options &options)
    : Wavefunction(options) {

  timer_on("EinHF: Setup wavefunction");

  // Shallow copy useful objects from the passed in wavefunction
  shallow_copy(ref_wfn);

  print_ = options_.get_int("PRINT");

  nirrep_ = sobasisset_->nirrep();
  nso_ = basisset_->nbf();

  maxiter_ = options_.get_int("SCF_MAXITER");
  e_convergence_ = options_.get_double("E_CONVERGENCE");
  d_convergence_ = options_.get_double("D_CONVERGENCE");
  func_ = functional;

  print_header();
  if (options_.get_bool("DIIS")) {
    diis_max_iters_ = options_.get_int("DIIS_MAX_VECS");
  } else {
    diis_max_iters_ = 0;
    outfile->Printf("Turning DIIS off.\n");
  }

  init_integrals();

  // Set Wavefunction matrices
  X_.set_name("S^1/2");
  F_.set_name("Fock Matrix");
  Ft_.set_name("Transformed Fock Matrix");
  C_.set_name("MO Coefficients");
  Cocc_.set_name("Occupied MO Coefficients");
  D_.set_name("Density Matrix");

  for (int i = 0; i < this->nirrep_; i++) {
    X_.push_block(einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    F_.push_block(einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    Ft_.push_block(einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    auto block = einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]);

    C_.push_block(block);
  }

  for (int i = 0; i < this->nirrep_; i++) {
    auto block = einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]);

    Cocc_.push_block(block);
  }

  for (int i = 0; i < this->nirrep_; i++) {
    auto block = einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]);

    D_.push_block(block);
  }

  for (int i = 0; i < this->nirrep_; i++) {
    JKwK_.push_block(einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]));
  }

  timer_off("EinHF: Setup wavefunction");
}

GPUEinsumsSCF::~GPUEinsumsSCF() {}

void GPUEinsumsSCF::init_integrals() {
  // The basisset object contains all of the basis information and is formed in
  // the new_wavefunction call The integral factory oversees the creation of
  // integral objects
  auto integral = std::make_shared<IntegralFactory>(basisset_, basisset_,
                                                    basisset_, basisset_);

  // Determine the number of electrons in the system
  // The molecule object is built into all wavefunctions
  int charge = molecule_->molecular_charge();
  int nelec = 0;
  for (int i = 0; i < molecule_->natom(); ++i) {
    nelec += (int)molecule_->Z(i);
  }
  nelec -= charge;
  if (nelec % 2) {
    throw PSIEXCEPTION("This is only an RHF code, but you gave it an odd "
                       "number of electrons.  Try again!");
  }
  ndocc_ = nelec / 2;

  outfile->Printf("    There are %d doubly occupied orbitals\n", ndocc_);
  molecule_->print();
  if (print_ > 1) {
    basisset_->print_detail();
    options_.print();
  }

  // nso_ = basisset_->nso();

  // Nuclear repulsion without a field
  e_nuc_ = molecule_->nuclear_repulsion_energy({0, 0, 0});
  outfile->Printf("\n    Nuclear repulsion energy: %16.8f\n\n", e_nuc_);

  // Make a MintsHelper object to help
  auto mints = std::make_shared<MintsHelper>(basisset_);

  // These don't need to be declared, because they belong to the class
  auto S_mat = mints->so_overlap();

  // Core Hamiltonian is Kinetic + Potential
  auto H_mat = mints->so_kinetic();

  this->nirrep_ = (int)S_mat->nirrep();

  occ_per_irrep_.resize(nirrep_);

  H_mat->add(mints->so_potential());

  H_.set_name("Hamiltonian");
  S_.set_name("Overlap");

  for (int i = 0; i < S_mat->nirrep(); i++) {
    irrep_sizes_.push_back(S_mat->rowdim(i));
    auto S_block = einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        S_mat->rowdim(i), S_mat->coldim(i));
    S_block.read(S_mat->get_pointer(i));
    S_.push_block(S_block);
  }

  for (int i = 0; i < S_mat->nirrep(); i++) {
    auto H_block = einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        H_mat->rowdim(i), H_mat->coldim(i));

    H_block.read(H_mat->get_pointer(i));

    H_.push_block(H_block);
  }

  if (print_ > 3) {
    fprintln(*outfile->stream(), S_);
    fprintln(*outfile->stream(), H_);
    outfile->stream()->flush();
  }

  outfile->Printf("    Forming JK object\n\n");

  size_t total_memory = Process::environment.get_memory() / 8 *
                        options_.get_double("SCF_MEM_SAFETY_FACTOR");

  // Construct a JK object that compute J and K SCF matrices very efficiently
  jk_ = JK::build_JK(basisset_, mintshelper_->get_basisset("DF_BASIS_SCF"),
                     options_, false, total_memory);

  jk_->set_do_K(func_->is_x_hybrid());
  jk_->set_do_wK(func_->is_x_lrc());

  jk_->set_omega(func_->x_omega());
  jk_->set_omega_alpha(func_->x_alpha());
  jk_->set_omega_beta(func_->x_beta());

  size_t jk_size = jk_->memory_estimate();

  if (jk_size < total_memory) {
    jk_->set_memory(jk_size);
  } else {
    jk_->set_memory(total_memory * 0.9);
  }

  // This is a very heavy compute object, lets give it 80% of our total memory
  // jk_->set_memory(Process::environment.get_memory() * 0.8);
  jk_->initialize();
  jk_->print_header();

  if (func_->needs_xc()) {
    v_ = VBase::build_V(basisset_, func_, options_);
    v_->initialize();
    v_->print_header();
  }
}

double GPUEinsumsSCF::compute_electronic_energy() {
  // Compute the electronic energy: (H + F)_pq * D_pq -> energy

  einsums::DeviceTensor<double, 0> e_tens;
  auto temp = einsums::BlockDeviceTensor<double, 2>(
      "temp", einsums::detail::DEV_ONLY, H_.vector_dims());

  temp = H_;
  temp *= 2;
  temp += JKwK_;

  einsums::tensor_algebra::einsum(
      0.0, einsums::tensor_algebra::Indices{}, &e_tens, 1.0,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      D_,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      temp);

  double out = (double)e_tens;

  if (func_->needs_xc()) {
    out += v_->quadrature_values()["FUNCTIONAL"];
  }

  if (func_->needs_vv10()) {
    out += v_->quadrature_values()["VV10"];
  }

  return out + scalar_variable("-D ENERGY");
}

void GPUEinsumsSCF::update_Cocc(
    const einsums::DeviceTensor<double, 1> &energies) {
  // Update occupation.

  for (int i = 0; i < nirrep_; i++) {
    occ_per_irrep_[i] = 0;
  }

  for (int i = 0; i < ndocc_; i++) {
    double curr_min = INFINITY;
    int irrep_occ = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (occ_per_irrep_[j] >= irrep_sizes_[j]) {
        continue;
      }

      double energy = energies(S_.block_range(j)[0] + occ_per_irrep_[j]);

      if (energy < curr_min) {
        curr_min = energy;
        irrep_occ = j;
      }
    }

    occ_per_irrep_[irrep_occ]++;
  }

  Cocc_.zero();
#pragma omp parallel for
  for (int i = 0; i < nirrep_; i++) {
    Cocc_[i](einsums::AllT{}, einsums::Range(0, occ_per_irrep_[i])) =
        C_[i](einsums::AllT{}, einsums::Range(0, occ_per_irrep_[i]));
  }
}

void GPUEinsumsSCF::compute_diis_coefs(
    const std::deque<einsums::BlockDeviceTensor<double, 2>> &errors,
    std::vector<double> *out) const {
  einsums::Tensor<double, 2> *B_mat = new einsums::Tensor<double, 2>(
      "DIIS error matrix", errors.size() + 1, errors.size() + 1);

  B_mat->zero();
  (*B_mat)(einsums::Range{errors.size(), errors.size() + 1},
           einsums::Range{0, errors.size()}) = 1.0;
  (*B_mat)(einsums::Range{0, errors.size()},
           einsums::Range{errors.size(), errors.size() + 1}) = 1.0;

#pragma omp parallel for
  for (int i = 0; i < errors.size(); i++) {
#pragma omp parallel for
    for (int j = 0; j <= i; j++) {
      (*B_mat)(i, j) = einsums::linear_algebra::dot(errors[i], errors[j]);
      (*B_mat)(j, i) = (*B_mat)(i, j);
    }
  }

  einsums::Tensor<double, 2> res_mat("DIIS result matrix", 1,
                                     errors.size() + 1);

  res_mat.zero();
  res_mat(0, errors.size()) = 1.0;

  einsums::linear_algebra::gesv(B_mat, &res_mat);

  out->resize(errors.size());

  for (int i = 0; i < errors.size(); i++) {
    out->at(i) = res_mat(0, i);
  }

  delete B_mat;
}

void GPUEinsumsSCF::compute_diis_fock(
    const std::vector<double> &coefs,
    const std::deque<einsums::BlockDeviceTensor<double, 2>> &focks,
    einsums::BlockDeviceTensor<double, 2> *out) const {

  out->zero();

  for (int i = 0; i < coefs.size(); i++) {
    einsums::linear_algebra::axpy(coefs[i], focks[i], out);
  }
}

double GPUEinsumsSCF::compute_energy() {
  timer_on("EinHF: Computing energy");
  if (diis_max_iters_ != 0) {
    outfile->Printf("Performing DIIS with %d vectors.\n", diis_max_iters_);
  } else {
    outfile->Printf("Turning DIIS off.\n");
  }

  // Allocate a few temporary matrices
  auto Temp1 = new einsums::BlockDeviceTensor<double, 2>(
      "Temporary Array 1", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto Temp2 = new einsums::BlockDeviceTensor<double, 2>(
      "Temporary Array 2", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto FDS = new einsums::BlockDeviceTensor<double, 2>(
      "FDS", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto SDF = new einsums::BlockDeviceTensor<double, 2>(
      "SDF", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto Evecs = new einsums::BlockDeviceTensor<double, 2>(
      "Eigenvectors", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto Evals = new einsums::DeviceTensor<double, 1>(
      "Eigenvalues", einsums::detail::DEV_ONLY, nso_);

  auto J = new einsums::BlockDeviceTensor<double, 2>(
      "J matrix", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto K = new einsums::BlockDeviceTensor<double, 2>(
      "K matrix", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto wK = new einsums::BlockDeviceTensor<double, 2>(
      "wK matrix", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto V = new einsums::BlockDeviceTensor<double, 2>(
      "V matrix", einsums::detail::DEV_ONLY, irrep_sizes_);

  std::deque<einsums::BlockDeviceTensor<double, 2>>
      *errors = new std::deque<einsums::BlockDeviceTensor<double, 2>>(0),
      *focks = new std::deque<einsums::BlockDeviceTensor<double, 2>>(0);
  std::vector<double> *coefs = new std::vector<double>(0),
                      *error_vals = new std::vector<double>(0);

  std::vector<int> old_occs;

  einsums::DeviceTensor<double, 0> *dRMS_tens =
      new einsums::DeviceTensor<double, 0>();

#pragma omp task depend(in : this->S_) depend(out : this -> X_)
  {
    timer_on("Form X");
    // Form the X_ matrix (S^-1/2)
    X_ = einsums::linear_algebra::pow(S_, -0.5);
    timer_off("Form X");
  }

#pragma omp task depend(in : this->H_) depend(out : this -> F_)
  {
    timer_on("Set guess");
    F_ = H_;
    timer_off("Set guess");
  }
#pragma omp task depend(inout : this->JKwK_)
  { JKwK_.zero(); }
#pragma omp taskwait depend(in : this->F_, this->X_)

  timer_on("Form C");
  einsums::linear_algebra::gemm<false, false>(1.0, F_, X_, 0.0, Temp1);
  einsums::linear_algebra::gemm<true, false>(1.0, X_, *Temp1, 0.0, &Ft_);
  *Evecs = Ft_;

  einsums::linear_algebra::syev(Evecs, Evals);

  einsums::linear_algebra::gemm<false, true>(1.0, X_, *Evecs, 0.0, &C_);
  timer_off("Form C");

  update_Cocc(*Evals);

  old_occs = occ_per_irrep_;

  timer_on("Form D");
  einsums::tensor_algebra::einsum(
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      &D_,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::m},
      Cocc_,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j,
                                       einsums::tensor_algebra::index::m},
      Cocc_);
  timer_off("Form D");

  if (print_ > 3) {
    outfile->Printf(
        "MO Coefficients and density from Core Hamiltonian guess:\n");
    fprintln(*outfile->stream(), X_);
    fprintln(*outfile->stream(), C_);
    fprintln(*outfile->stream(), D_);
    fprintln(*outfile->stream(), *Evals);
    fprintln(*outfile->stream(), Cocc_);
  }

  int iter = 1;
  bool converged = false;
  double e_old;
  double e_new = e_nuc_ + compute_electronic_energy();

  outfile->Printf("    Energy from core Hamiltonian guess: %20.16f\n\n", e_new);

  outfile->Printf("    Initial Occupation:\n         \t");

  for (int i = 0; i < S_.num_blocks(); i++) {
    outfile->Printf("%4s\t", S_.name(i).c_str());
  }

  outfile->Printf("\n    DOCC \t");
  for (int i = 0; i < occ_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", occ_per_irrep_[i]);
  }

  outfile->Printf("\n    VIRT \t");

  for (int i = 0; i < occ_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", irrep_sizes_[i] - occ_per_irrep_[i]);
  }

  outfile->Printf("\n    Total\t");

  for (int i = 0; i < occ_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", irrep_sizes_[i]);
  }

  outfile->Printf("\n");

  outfile->Printf(
      "    *===========================================================*\n");
  outfile->Printf(
      "    * Iter       Energy            delta E    ||gradient||      *\n");
  outfile->Printf(
      "    *-----------------------------------------------------------*\n");

  while (!converged && iter < maxiter_) {
    e_old = e_new;

    timer_on("Form F");
// Add the core Hamiltonian term to the Fock operator
#pragma omp task depend(in : this->H_) depend(out : this -> F_)
    { F_ = H_; }

#pragma omp task depend(inout : this->JKwK_)
    { JKwK_.zero(); }

    // The JK object handles all of the two electron integrals
    // To enhance efficiency it does use the density, but the orbitals
    // themselves D_uv = C_ui C_vj J_uv = I_uvrs D_rs K_uv = I_urvs D_rs

    // Here we clear the old Cocc and push_back our new one
    std::vector<SharedMatrix> &Cl = jk_->C_left();
    Cl.clear();

    SharedMatrix CTemp = std::make_shared<Matrix>(nirrep_, irrep_sizes_.data(),
                                                  occ_per_irrep_.data());

#pragma omp parallel for
    for (int i = 0; i < nirrep_; i++) {
      if (irrep_sizes_[i] == 0) {
        continue;
      }
#pragma omp parallel for
      for (int j = 0; j < irrep_sizes_[i]; j++) {
#pragma omp parallel for
        for (int k = 0; k < occ_per_irrep_[i]; k++) {
          (*CTemp.get())(i, j, k) = Cocc_[i](j, k);
        }
      }
    }

    Cl.push_back(CTemp);
    jk_->compute();

#pragma omp task depend(out : *J)
    {
      const std::vector<SharedMatrix> &J_mat = jk_->J();
      timer_on("Form J");
#pragma omp parallel for
      for (int i = 0; i < nirrep_; i++) {
        if (irrep_sizes_[i] == 0) {
          continue;
        }
        J->block(i).read(J_mat[0]->get_pointer(i));
      }
      *J *= 2.0;
      timer_off("Form J");
    }

    if (func_->is_x_hybrid() && !(func_->is_x_lrc() && jk_->get_wcombine())) {
#pragma omp task depend(out : *K)
      {
        const std::vector<SharedMatrix> &K_mat = jk_->K();
        double alpha = func_->x_alpha();
        timer_on("Form K");
#pragma omp parallel for
        for (int i = 0; i < nirrep_; i++) {
          if (irrep_sizes_[i] == 0) {
            continue;
          }
          K->block(i).read(K_mat[0]->get_pointer(i));
        }
        *K *= alpha;
        timer_off("Form K");
      }
    }

    if (func_->is_x_lrc()) {
#pragma omp task depend(out : *wK)
      {
        const std::vector<SharedMatrix> &wK_mat = jk_->wK();
        double beta = func_->x_beta();
        timer_on("Form wK");
#pragma omp parallel for
        for (int i = 0; i < nirrep_; i++) {
          if (irrep_sizes_[i] == 0) {
            continue;
          }
          wK->block(i).read(wK_mat[0]->get_pointer(i));
        }
        *wK *= beta;
        timer_off("Form wK");
      }
    }

    if (func_->needs_xc()) {
#pragma omp task depend(in : this->D_) depend(out : *V)
      {
        SharedMatrix D_mat = std::make_shared<Matrix>(
                         nirrep_, irrep_sizes_.data(), irrep_sizes_.data()),
                     V_mat = std::make_shared<Matrix>(
                         nirrep_, irrep_sizes_.data(), irrep_sizes_.data());

        for (int i = 0; i < nirrep_; i++) {
          if (irrep_sizes_[i] == 0) {
            continue;
          }
          for (int j = 0; j < irrep_sizes_[i]; j++) {
            for (int k = 0; k < irrep_sizes_[i]; k++) {
              (*D_mat.get())(i, j, k) = D_[i](j, k);
            }
          }
        }

        std::vector<SharedMatrix> D_vec{D_mat};
        std::vector<SharedMatrix> V_vec{V_mat};

        v_->set_D(D_vec);
        v_->compute_V(V_vec);

#pragma omp parallel for
        for (int i = 0; i < nirrep_; i++) {
          if (irrep_sizes_[i] == 0) {
            continue;
          }
          V->block(i).read(V_mat->get_pointer(i));
        }
      }
    }

#pragma omp task depend(in : *J) depend(out : this -> F_, this->JKwK_)
    {
      F_ += *J;
      JKwK_ += *J;
    }
    if (func_->is_x_hybrid() && !(func_->is_x_lrc() && jk_->get_wcombine())) {
#pragma omp task depend(in : *K) depend(out : this -> F_, this->JKwK_)
      {
        F_ -= *K;
        JKwK_ -= *K;
      }
    }

    if (func_->is_x_lrc()) {
#pragma omp task depend(in : *wK) depend(out : this -> F_, this->JKwK_)
      {
        F_ -= *wK;
        JKwK_ -= *wK;
      }
    }

    if (func_->needs_xc()) {
#pragma omp task depend(in : *V) depend(out : this -> F_)
      { F_ += *V; }
    }

// I don't think this will cause a slowdown since the next part will do the same
// thing.
#pragma omp taskwait depend(in : this->F_)
    timer_off("Form F");

// Compute the orbital gradient, FDS-SDF
#pragma omp task depend(in : this->D_, this->S_, this->F_) depend(out : *FDS)
    {
      einsums::linear_algebra::gemm<false, false>(1.0, D_, S_, 0.0, Temp1);
      einsums::linear_algebra::gemm<false, false>(1.0, F_, *Temp1, 0.0, FDS);
    }
#pragma omp task depend(in : this->D_, this->S_, this->F_) depend(out : *SDF)
    {
      einsums::linear_algebra::gemm<false, false>(1.0, D_, F_, 0.0, Temp2);
      einsums::linear_algebra::gemm<false, false>(1.0, S_, *Temp2, 0.0, SDF);
    }

#pragma omp task depend(in : *FDS, *SDF) depend(out : *Temp1)
    {
      *Temp1 = *FDS;
      *Temp1 -= *SDF;
    }

    // Density RMS
    *dRMS_tens = 0;

    if (diis_max_iters_ > 0) {
#pragma omp task depend(in : *Temp1)                                           \
    depend(inout : *errors, *focks, this -> F_, *coefs, *error_vals)
      {
        timer_on("Perform DIIS");

        if (errors->size() == diis_max_iters_) {
          double max_error = -INFINITY;
          int max_ind = -1;

          for (int i = 0; i < diis_max_iters_; i++) {
            if (error_vals->at(i) > max_error) {
              max_error = error_vals->at(i);
              max_ind = i;
            }
          }

          focks->at(max_ind) = F_;
          errors->at(max_ind) = *Temp1;
          error_vals->at(max_ind) =
              einsums::linear_algebra::dot(*Temp1, *Temp1);
        } else {
          errors->push_back(*Temp1);
          focks->push_back(F_);
          error_vals->push_back(einsums::linear_algebra::dot(*Temp1, *Temp1));
        }

        compute_diis_coefs(*errors, coefs);

        compute_diis_fock(*coefs, *focks, &F_);
        timer_off("Perform DIIS");
      }
    }

#pragma omp task depend(in : *Temp1) depend(out : *dRMS_tens)
    {
      einsums::tensor_algebra::einsum(
          0.0, einsums::tensor_algebra::Indices{}, dRMS_tens,
          1.0 / (nso_ * nso_),
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::j},
          *Temp1,
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::j},
          *Temp1);
    }

// Compute the energy
#pragma omp taskwait depend(in : this->JKwK_, this->D_, this->H_)
    timer_on("Compute electronic energy");
    e_new = e_nuc_ + compute_electronic_energy();
    timer_off("Compute electronic energy");

    double dE = e_new - e_old;

#pragma omp taskwait depend(in : *dRMS_tens)
    double dRMS = std::sqrt((double)*dRMS_tens);

    converged = (fabs(dE) < e_convergence_) && (dRMS < d_convergence_);

    outfile->Printf("    * %3d %20.14f    %9.2e    %9.2e    ", iter, e_new, dE,
                    dRMS);
    if (focks->size() > 0) {
      outfile->Printf("DIIS*\n");
    } else {
      outfile->Printf("    *\n");
    }
    timer_on("Form C");
    einsums::linear_algebra::gemm<false, false>(1.0, F_, X_, 0.0, Temp1);
    einsums::linear_algebra::gemm<true, false>(1.0, X_, *Temp1, 0.0, &Ft_);

    *Evecs = Ft_;
    einsums::linear_algebra::syev(Evecs, Evals);

    einsums::linear_algebra::gemm<false, true>(1.0, X_, *Evecs, 0.0, &C_);
    timer_off("Form C");

    update_Cocc(*Evals);

    timer_on("Form D");
    einsums::tensor_algebra::einsum(
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::j},
        &D_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::m},
        Cocc_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j,
                                         einsums::tensor_algebra::index::m},
        Cocc_);
    timer_off("Form D");

    if (occ_per_irrep_ != old_occs) {
      outfile->Printf("    Occupation Changed:\n         \t");

      for (int i = 0; i < S_.num_blocks(); i++) {
        outfile->Printf("%4s\t", S_.name(i).c_str());
      }

      outfile->Printf("\n    DOCC \t");
      for (int i = 0; i < occ_per_irrep_.size(); i++) {
        outfile->Printf("%4d\t", occ_per_irrep_[i]);
      }

      outfile->Printf("\n    VIRT \t");

      for (int i = 0; i < occ_per_irrep_.size(); i++) {
        outfile->Printf("%4d\t", irrep_sizes_[i] - occ_per_irrep_[i]);
      }

      outfile->Printf("\n    Total\t");

      for (int i = 0; i < occ_per_irrep_.size(); i++) {
        outfile->Printf("%4d\t", irrep_sizes_[i]);
      }

      outfile->Printf("\n");
    }

    old_occs = occ_per_irrep_;

    // Optional printing
    if (print_ > 3) {
#pragma omp taskwait depend(in : this->Ft_, this->F_, *Evecs, *Evals,          \
                                this->C_, this->D_, *FDS, *SDF, *Temp1,        \
                                *errors, *focks, *coefs)
      fprintln(*outfile->stream(), Ft_);
      fprintln(*outfile->stream(), F_);
      fprintln(*outfile->stream(), *Evecs);
      fprintln(*outfile->stream(), *Evals);
      fprintln(*outfile->stream(), C_);
      fprintln(*outfile->stream(), D_);
      fprintln(*outfile->stream(), *FDS);
      fprintln(*outfile->stream(), *SDF);
      Temp1->set_name("Orbital Gradient");
      fprintln(*outfile->stream(), *Temp1);

      outfile->Printf("DIIS error size: %d\nDIIS Focks size: %d\n",
                      errors->size(), focks->size());
      outfile->Printf("DIIS coefs: ");

      for (int i = 0; i < coefs->size(); i++) {
        outfile->Printf("%lf ", coefs->at(i));
      }
      outfile->Printf("\n");
    }

    iter++;
  }
  outfile->Printf(
      "    *===========================================================*\n");

  if (!converged)
    throw PSIEXCEPTION("The SCF iterations did not converge.");

  Evals->set_name("Orbital Energies");
  outfile->Printf("\nOccupied:\n");

  std::vector<int> inds(nirrep_);
  for (int i = 0; i < nirrep_; i++) {
    inds[i] = 0;
  }

  for (int i = 0; i < ndocc_; i++) {
    double curr_min = INFINITY;
    int min_ind = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (inds[j] >= occ_per_irrep_[j]) {
        continue;
      }
      if ((*Evals)(S_.block_range(j)[0] + inds[j]) < curr_min) {
        curr_min = (*Evals)(S_.block_range(j)[0] + inds[j]);
        min_ind = j;
      }
    }

    inds[min_ind]++;
    outfile->Printf("%3d %s: %lf\n", inds[min_ind],
                    to_lower(S_[min_ind].name()).c_str(), curr_min);
  }

  outfile->Printf("Unoccupied:\n");

  for (int i = ndocc_; i < nso_; i++) {
    double curr_min = INFINITY;
    int min_ind = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (inds[j] >= irrep_sizes_[j]) {
        continue;
      }
      if ((*Evals)(S_.block_range(j)[0] + inds[j]) < curr_min) {
        curr_min = (*Evals)(S_.block_range(j)[0] + inds[j]);
        min_ind = j;
      }
    }

    inds[min_ind]++;
    outfile->Printf("%3d %s: %lf\n", inds[min_ind],
                    to_lower(S_[min_ind].name()).c_str(), curr_min);
  }
  energy_ = e_new;

  delete coefs;
  delete focks;
  delete errors;
  delete error_vals;

  delete Temp1;
  delete Temp2;
  delete Evecs;
  delete Evals;
  delete FDS;
  delete SDF;

  delete J;
  delete K;
  delete wK;
  delete V;
  delete dRMS_tens;

  timer_off("EinHF: Computing energy");

  return e_new;
}

void GPUEinsumsSCF::print_header() {
  int nthread = Process::environment.get_n_threads();

  outfile->Printf("\n");
  outfile->Printf(
      "         ---------------------------------------------------------\n");
  outfile->Printf("                                   Einsums SCF\n");
  outfile->Printf("                                by Connor Briggs\n");
  outfile->Printf("                                 %4s Reference\n",
                  options_.get_str("REFERENCE").c_str());
  outfile->Printf("                               Running on the GPU\n");
  outfile->Printf("                      %3d Threads, %6ld MiB Core\n", nthread,
                  memory_ / 1048576L);
  outfile->Printf(
      "         ---------------------------------------------------------\n");
  outfile->Printf("\n");
  outfile->Printf("  ==> Geometry <==\n\n");

  molecule_->print();

  outfile->Printf("  Running in %s symmetry.\n\n",
                  molecule_->point_group()->symbol().c_str());

  molecule_->print_rotational_constants();

  outfile->Printf("  Nuclear repulsion = %20.15f\n\n",
                  molecule_->nuclear_repulsion_energy({0, 0, 0}));
  outfile->Printf("  Charge       = %d\n", molecule_->molecular_charge());
  outfile->Printf("  Multiplicity = %d\n", molecule_->multiplicity());
  outfile->Printf("  Nalpha       = %d\n", nalpha_);
  outfile->Printf("  Nbeta        = %d\n\n", nbeta_);

  outfile->Printf("  ==> Algorithm <==\n\n");
  outfile->Printf("  SCF Algorithm Type is %s.\n",
                  options_.get_str("SCF_TYPE").c_str());
  outfile->Printf("  DIIS %s.\n",
                  options_.get_bool("DIIS") ? "enabled" : "disabled");
  outfile->Printf("  Energy threshold   = %3.2e\n",
                  options_.get_double("E_CONVERGENCE"));
  outfile->Printf("  Density threshold  = %3.2e\n",
                  options_.get_double("D_CONVERGENCE"));

  outfile->Printf("  ==> Primary Basis <==\n\n");

  basisset_->print_by_level("outfile", print_);

  outfile->Printf("  ==> Functional <==\n\n");
  func_->print("outfile", print_);
}
} // namespace einhf
} // namespace psi
