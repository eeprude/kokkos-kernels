//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

// AquiEEP

#ifndef KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class AMatrix, class BXMV>
struct gesv_tpl_spec_avail {
  enum : bool { value = false };
};

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS

#define KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_BLAS(SCALAR, LAYOUT, MEMSPACE)     \
  template <class ExecSpace>                                              \
  struct gesv_tpl_spec_avail<                                             \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {          \
    enum : bool { value = true };                                         \
  };

KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutLeft,
                                    Kokkos::HostSpace)
KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutLeft,
                                    Kokkos::HostSpace)
KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutLeft,
                                    Kokkos::HostSpace)
KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutLeft,
                                    Kokkos::HostSpace)
/*
#if defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
 KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_BLAS( double, Kokkos::LayoutRight,
Kokkos::HostSpace) #endif
#if defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
 KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_BLAS( float, Kokkos::LayoutRight,
Kokkos::HostSpace) #endif
#if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
 KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_BLAS( Kokkos::complex<double>,
Kokkos::LayoutRight, Kokkos::HostSpace) #endif
#if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
 KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_BLAS( Kokkos::complex<float>,
Kokkos::LayoutRight, Kokkos::HostSpace) #endif
*/
#endif

// MAGMA
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA

#define KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_MAGMA(SCALAR, LAYOUT, MEMSPACE)    \
  template <class ExecSpace>                                              \
  struct gesv_tpl_spec_avail<                                             \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {          \
    enum : bool { value = true };                                         \
  };

KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_MAGMA(double, Kokkos::LayoutLeft,
                                     Kokkos::CudaSpace)
KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_MAGMA(float, Kokkos::LayoutLeft,
                                     Kokkos::CudaSpace)
KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_MAGMA(Kokkos::complex<double>,
                                     Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_MAGMA(Kokkos::complex<float>, Kokkos::LayoutLeft,
                                     Kokkos::CudaSpace)

/*
#if defined (KOKKOSKERNELS_INST_DOUBLE) \
 && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
 KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_MAGMA( double, Kokkos::LayoutRight,
Kokkos::CudaSpace) #endif
#if defined (KOKKOSKERNELS_INST_FLOAT) \
 && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
 KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_MAGMA( float, Kokkos::LayoutRight,
Kokkos::CudaSpace) #endif
#if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
 && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
 KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_MAGMA(
Kokkos::complex<double>,Kokkos::LayoutRight, Kokkos::CudaSpace) #endif
#if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
 && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
 KOKKOSBLAS_GESV_TPL_SPEC_AVAIL_MAGMA( Kokkos::complex<float>,
Kokkos::LayoutRight, Kokkos::CudaSpace) #endif
*/
#endif

}  // namespace Impl
}  // namespace KokkosBlas

#endif
