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

// only enable this test where KokkosBlas supports getrf:
// CUDA+MAGMA and HOST+BLAS
#if (defined(TEST_CUDA_BLAS_CPP) &&                                           \
     defined(KOKKOSKERNELS_ENABLE_TPL_MAGMA)) ||                              \
    (defined(KOKKOSKERNELS_ENABLE_TPL_BLAS) &&                                \
     (defined(TEST_OPENMP_BLAS_CPP) || defined(TEST_OPENMPTARGET_BLAS_CPP) || \
      defined(TEST_SERIAL_BLAS_CPP) || defined(TEST_THREADS_BLAS_CPP)))

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <KokkosBlas_getrf.hpp>
//#include <KokkosBlas3_gemm.hpp>
#include <KokkosKernels_TestUtils.hpp>

namespace Test {

template <class ViewTypeA, class Device>
void impl_test_getrf(int N) {
  typedef typename Device::execution_space execution_space;
  typedef typename ViewTypeA::value_type ScalarA;
  //typedef Kokkos::ArithTraits<ScalarA> ats; // Aqui

  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);

  // Create device views
  ViewTypeA A("A", N, N);

  // Create host mirrors of device views.
#if 0 // Aqui
  typename ViewTypeB::HostMirror h_X0 = Kokkos::create_mirror_view(X0);
  typename ViewTypeB::HostMirror h_B  = Kokkos::create_mirror(B);
#endif
  
  // Initialize data.
  Kokkos::fill_random(
      A, rand_pool,
      Kokkos::rand<Kokkos::Random_XorShift64<execution_space>, ScalarA>::max());

#if 0 // Aqui
  // Generate RHS B = A*X0.
  ScalarA alpha = 1.0;
  ScalarA beta  = 0.0;

  KokkosBlas::gemv("N", alpha, A, X0, beta, B);
  Kokkos::fence();

  // Deep copy device view to host view.
  Kokkos::deep_copy(h_X0, X0);
#endif

  // Allocate IPIV view on host
  typedef Kokkos::View<int*, Kokkos::LayoutLeft, Kokkos::HostSpace> ViewTypeP; // AquiLuc
  ViewTypeP ipiv("IPIV", N);

  // Solve.
  try {
    KokkosBlas::getrf(A, ipiv);
  } catch (const std::runtime_error& error) {
    // Check for expected runtime errors due to:
    // no-pivoting case (note: only MAGMA supports no-pivoting interface)
    // and no-tpl case
    bool nopivot_runtime_err = false;
    bool notpl_runtime_err   = false;
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA  // have MAGMA TPL
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS   // and have BLAS TPL
    nopivot_runtime_err = (!std::is_same<typename Device::memory_space,
                                         Kokkos::CudaSpace>::value) &&
                          (ipiv.extent(0) == 0) && (ipiv.data() == nullptr);
    notpl_runtime_err = false;
#else
    notpl_runtime_err = true;
#endif
#else                                 // not have MAGMA TPL
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS  // but have BLAS TPL
    nopivot_runtime_err = (ipiv.extent(0) == 0) && (ipiv.data() == nullptr);
    notpl_runtime_err   = false;
#else
    notpl_runtime_err = true;
#endif
#endif
    if (!nopivot_runtime_err && !notpl_runtime_err) FAIL();
    return;
  }
  Kokkos::fence();

#if 0 // Aqui  
  // Get the solution vector.
  Kokkos::deep_copy(h_B, B);

  // Checking vs ref on CPU, this eps is about 10^-9
  typedef typename ats::mag_type mag_type;
  const mag_type eps = 1.0e7 * ats::epsilon();
  bool test_flag     = true;
  for (int i = 0; i < N; i++) {
    if (ats::abs(h_B(i) - h_X0(i)) > eps) {
      test_flag = false;
      // printf( "    Error %d: result( %.15lf ) !=
      // solution( %.15lf ) at (%d)\n", N,
      // ats::abs(h_B(i)), ats::abs(h_X0(i)), int(i) );
      // break;
    }
  }
  ASSERT_EQ(test_flag, true);
#endif
}

}  // namespace Test

template <class Scalar, class Device>
int test_getrf() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&      \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<Scalar**, Kokkos::LayoutLeft, Device> view_type_a_ll;
  Test::impl_test_getrf<view_type_a_ll, Device>(
      2);
  Test::impl_test_getrf<view_type_a_ll, Device>(
      13);
  Test::impl_test_getrf<view_type_a_ll, Device>(
      179);
  Test::impl_test_getrf<view_type_a_ll, Device>(
      64);
  Test::impl_test_getrf<view_type_a_ll, Device>(
      1024);
  Test::impl_test_getrf<view_type_a_ll, Device>(
      13);
  Test::impl_test_getrf<view_type_a_ll, Device>(
      179);
#endif

  return 1;
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) && \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, getrf_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::getrf_float");
  test_getrf<float, TestExecSpace>();
  Kokkos::Profiling::popRegion();
}

#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&  \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, getrf_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::getrf_double");
  test_getrf<double, TestExecSpace>();
  Kokkos::Profiling::popRegion();
}

#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&          \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, getrf_complex_double) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::getrf_complex_double");
  test_getrf<Kokkos::complex<double>, TestExecSpace>();
  Kokkos::Profiling::popRegion();
}

#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&         \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F(TestCategory, getrf_complex_float) {
  Kokkos::Profiling::pushRegion("KokkosBlas::Test::getrf_complex_float");
  test_getrf<Kokkos::complex<float>, TestExecSpace>();
  Kokkos::Profiling::popRegion();
}

#endif

#endif  // CUDA+MAGMA or BLAS+HOST
