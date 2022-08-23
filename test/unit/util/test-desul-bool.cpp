#include "RAJA/RAJA.hpp"
#include "desul/atomics.hpp"

#include "RAJA_gtest.hpp"
#include "RAJA_test-base.hpp"

/// A mutex lock implemented using a DESUL atomic bool.
class desul_mutex_bool
{
  bool lock;

public:
  RAJA_DEVICE desul_mutex_bool() : lock(false) {}

  /// Acquire the mutex. If locked, wait until unlocked.
  RAJA_DEVICE void acquire()
  {
    bool exchanged_value = true;
    do {
      // Try in a loop to exchange it from false to true until successful.
      exchanged_value = !atomic_exchange(&lock,
                                         true,
                                         desul::MemoryOrderAcquire(),
                                         desul::MemoryScopeDevice());
    } while (exchanged_value == true);
  }

  /// Release the mutex.
  RAJA_DEVICE void release()
  {
    atomic_store(&lock,
                 false,
                 desul::MemoryOrderRelease(),
                 desul::MemoryScopeDevice());
  }
};

/// From host, invoke a device kernel to acquire a desul_mutex_bool that lives
/// in device memory.
void acquire(desul_mutex_bool *lock)
{
  using policy = RAJA::cuda_exec<1>;
  auto range = RAJA::RangeSegment(0, 1);
  RAJA::forall<policy>(range, [=] RAJA_DEVICE(int) { lock->acquire(); });
}

/// Entry point for the test.
TEST(DesulMutexBoolTest, AcquireTest)
{
  desul_mutex_bool *lock;
  cudaErrchk(cudaMallocManaged((void **)&lock,
                               sizeof(desul_mutex_bool),
                               cudaMemAttachGlobal));
  acquire(lock);
  cudaErrchk(cudaFree(lock));
}
