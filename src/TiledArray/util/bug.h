//
// bug.h
//
// Copyright (C) 1996 Limit Point Systems, Inc.
//
// Author: Curtis Janssen <cljanss@limitpt.com>
// Maintainer: LPS
//
// This file is part of the SC Toolkit.
//
// The SC Toolkit is free software; you can redistribute it and/or modify
// it under the terms of the GNU Library General Public License as published by
// the Free Software Foundation; either version 2, or (at your option)
// any later version.
//
// The SC Toolkit is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Library General Public License for more details.
//
// You should have received a copy of the GNU Library General Public License
// along with the SC Toolkit; see the file COPYING.LIB.  If not, write to
// the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
//
// The U.S. Government is granted a limited license as per AL 91-7.
//

#ifndef TILEDARRAY_UTIL_BUG_H_
#define TILEDARRAY_UTIL_BUG_H_

#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <pthread.h>
#if defined(__APPLE__) && defined(__x86_64__)
#include <mach/mach.h>
#include <stdexcept>
#endif

namespace TiledArray {
namespace detail {

/// @brief MemoryWatchpoint represents a hardware watchpoint for a memory
/// location Implements a memory watchpoint on x86 ... only implemented for
/// macOS so far this is a slightly tweaked version of
/// https://m.habrahabr.ru/post/103073/ see also
/// http://www.sandpile.org/x86/drx.htm for the x86 debugging register map
class MemoryWatchpoint_x86_64 {
 public:
  // x86 debugging registers are described in see
  // https://www.intel.com/content/www/us/en/architecture-and-technology/64-ia-32-architectures-software-developer-system-programming-manual-325384.html
  enum DebugRegister { kDR0 = 0, kDR1 = 1, kDR2 = 2, kDR3 = 3 };

  enum Size {
    kByte = 0,      // 1 byte  - 00
    kHalfWord = 1,  // 2 bytes - 01
    kWord = 3,      // 4 bytes - 11
                //        kDoubleWord = 2,  // 8 bytes - 10  NOT SUPPORTED BY
                //        SOME CHIPS!
    kSizeMask = 3  // mask      11
  };

  enum BreakState {
    kDisabled = 0,         // disabled   - 00
    kEnabledLocally = 1,   // task local - 01
    kEnabledGlobally = 2,  // global     - 10
    kBreakStateMask = 3    // mask         11
  };

  enum Condition {
    kWhenExecuted = 0,       // on execution     - 00
    kWhenWritten = 1,        // on write         - 01
    kWhenWrittenOrRead = 3,  // on read or write - 11
    kConditionMask = 3       // mask               11
  };

  /// a singleton pool of MemoryWatchpoint objects
  class Pool {
   public:
    static constexpr const size_t nwatchpoints_per_thread = 4;

    ~Pool() = default;

    /// Constructs the unique pool from a set of thread IDs
    /// @param threads a vector of pthread_t obtained via pthread_self (for the
    /// main thread) and pthread_create for other threads; only pointers to
    /// these unique objects can be used in subsequent calls.
    static void initialize_instance(
        const std::vector<const pthread_t *> &threads) {
      get_instance() = std::shared_ptr<Pool>(new Pool(threads));
    }

    /// accesses the unique pool; asserts that the default instance has been
    /// initialized by calling initialize()
    static std::shared_ptr<Pool> instance() {
      auto result = get_instance();
      assert(result &&
             "Pool::instance() called but Pool::initialize_instance() had not "
             "been called");
      return result;
    }

    /// Sets a watchpoint in thread @c thread and pointing to @c addr
    /// @param thread pthread_t object
    /// @throws std::runtime_error if setting memory watchpoint failed
    /// @return reference to this
    Pool &set(void *addr, Size size, Condition cond, const pthread_t *thread) {
      const auto it = pool_.find(thread);
      assert(it != pool_.end());
      // make sure there is no watchpoint for this address already
      for (auto &watchpt_ptr : it->second) {
        if (watchpt_ptr && watchpt_ptr->address() == addr) return *this;
      }
      // now create a watchpoint
      for (auto dr = 0; dr != nwatchpoints_per_thread; ++dr) {
        auto &watchpt_ptr = it->second[dr];
        if (!watchpt_ptr) {
          watchpt_ptr = std::make_shared<MemoryWatchpoint_x86_64>(
              addr, size, cond, static_cast<DebugRegister>(dr), thread);
          return *this;
        }
      }
      return *this;
    }

    /// Find watchpoint in thread @c thread and pointing to @c addr
    /// @param thread pointer to the pthread_t
    /// @return the pointer to the MemoryWatchpoint object; nullptr if such a
    /// watchpoint does not exist
    MemoryWatchpoint_x86_64 *find(void *addr, const pthread_t *thread) {
      const auto it = pool_.find(thread);
      assert(it != pool_.end());
      for (auto &watchpt_ptr : it->second) {
        if (watchpt_ptr && watchpt_ptr->address() == addr)
          return watchpt_ptr.get();
      }
      return nullptr;
    }

    /// @param thread pointer to the pthread_t
    Pool &clear(void *addr, const pthread_t *thread) {
      const auto it = pool_.find(thread);
      assert(it != pool_.end());
      for (auto &watchpt_ptr : it->second) {
        if (watchpt_ptr && watchpt_ptr->address() == addr) {
          watchpt_ptr.reset();
          return *this;
        }
      }
      return *this;
    }

   private:
    std::unordered_map<const pthread_t *,
                       std::vector<std::shared_ptr<MemoryWatchpoint_x86_64>>>
        pool_;

    /// Constructs a pool from a set of thread IDs
    /// N.B. The rationale for this is opaqueness of pthread_t and its lack of
    /// functionality. To use pthread_t portably you must store pthread_t
    /// obtained from pthread_self (used to obtain pthread_t for the main
    /// thread) and pthread_create; then pass pointers to the unique objects to
    /// this constructor. The pointers are used to index threads.
    explicit Pool(const std::vector<const pthread_t *> &threads) {
      for (const auto &thread : threads) {
        assert(thread != nullptr);
        pool_[thread].resize(nwatchpoints_per_thread);
      }
    }

    static std::shared_ptr<Pool> &get_instance() {
      static std::shared_ptr<Pool> instance_;
      return instance_;
    }
  };

  /// @brief creates a MemoryWatchpoint watching memory window
  /// `[addr,addr+size)` for condition @p cond from thread @p thread
  /// @param[in] addr the beginning of the memory window
  /// @param[in] size the size of the memory window
  /// @param[in] cond the condition to watch for
  /// @param[in] dr the debugging register to use
  /// @param[in] thread the thread to watch
  /// @throw std::runtime_error if setting the watchpoint fails (either due to
  /// the lack of available registers or another reason)
  MemoryWatchpoint_x86_64(void *addr, Size size, Condition cond,
                          DebugRegister dr, const pthread_t *thread)
      : addr_(addr), size_(size), cond_(cond), dr_(dr), thread_(thread) {
    init(true);
  }

  ~MemoryWatchpoint_x86_64() { init(false); }

  void *address() const { return addr_; }
  Size size() const { return size_; }
  Condition condition() const { return cond_; }
  DebugRegister debug_register() const { return dr_; }

 private:
  void *addr_;
  Size size_;
  Condition cond_;
  DebugRegister dr_;
  const pthread_t *thread_;

  inline uint64_t MakeFlags(DebugRegister reg, BreakState state, Condition cond,
                            Size size) {
    // N.B. each register takes 2 bits in DR7
    return (state | cond << 16 | size << 24) << (2 * reg);
  }

  inline uint64_t MakeMask(DebugRegister reg) {
    return MakeFlags(reg, kBreakStateMask, kConditionMask, kSizeMask);
  }

  friend class MemoryWatchPool;

  void init(bool create) {
#if defined(__APPLE__) && defined(__x86_64__)
    x86_debug_state dr;
    mach_msg_type_number_t dr_count = x86_DEBUG_STATE_COUNT;

    mach_port_t target_mach_thread = pthread_mach_thread_np(*thread_);

    kern_return_t rc =
        thread_get_state(target_mach_thread, x86_DEBUG_STATE,
                         reinterpret_cast<thread_state_t>(&dr), &dr_count);

    if (create && rc != KERN_SUCCESS)
      throw std::runtime_error(
          "MemoryWatchpoint_x86_64::MemoryWatchpoint_x86_64(): "
          "thread_get_state failed");

    switch (dr_) {
      case kDR0:
        dr.uds.ds64.__dr0 = reinterpret_cast<uint64_t>(addr_);
        break;
      case kDR1:
        dr.uds.ds64.__dr1 = reinterpret_cast<uint64_t>(addr_);
        break;
      case kDR2:
        dr.uds.ds64.__dr2 = reinterpret_cast<uint64_t>(addr_);
        break;
      case kDR3:
        dr.uds.ds64.__dr3 = reinterpret_cast<uint64_t>(addr_);
        break;
    }

    dr.uds.ds64.__dr7 &= ~MakeMask(dr_);

    dr.uds.ds64.__dr7 |=
        MakeFlags(dr_, create ? kEnabledLocally : kDisabled, cond_, size_);

    rc = thread_set_state(target_mach_thread, x86_DEBUG_STATE,
                          reinterpret_cast<thread_state_t>(&dr), dr_count);

    if (create && rc != KERN_SUCCESS)
      throw std::runtime_error(
          "MemoryWatchpoint_x86_64::MemoryWatchpoint_x86_64(): "
          "thread_set_state failed");
#endif  // defined(__APPLE__) && defined(__x86_64__)
  }
};

}  // namespace detail
}  // namespace TiledArray

namespace TiledArray {

/**
 * The Debugger class describes what should be done when a catastrophic
 * error causes unexpected program termination.  It can try things such as
 * start a debugger running where the program died or it can attempt to
 * produce a stack traceback showing roughly where the program died.  These
 * attempts will not always succeed.
 */
class Debugger {
 protected:
  std::string prefix_;
  std::string exec_;
  std::string cmd_;
  volatile int debugger_ready_;

  bool debug_;
  bool traceback_;
  bool exit_on_signal_;
  bool sleep_;
  bool wait_for_debugger_;
  bool handle_sigint_;
  int *mysigs_;

  void init();

  static std::shared_ptr<Debugger> default_debugger_;

  /** prints out a backtrace
   *
   * @param prefix this string will be prepended at the beginning of each line
   * of Backtrace
   * @param reason optional string specifying the reason for traceback
   * @return backtrace
   */
  static void __traceback(const std::string &prefix,
                          const char *reason = nullptr);

 public:
  /** @brief Programmatic construction of Debugger
   * @param exec the executable name
   */
  explicit Debugger(const char *exec = nullptr);
  virtual ~Debugger();

  /** The debug member attempts to start a debugger
      running on the current process. */
  virtual void debug(const char *reason);
  /** The traceback member attempts to produce a Backtrace
   for the current process.  A symbol table must be saved for
   the executable if any sense is to be made of the traceback.
   This feature is available on platforms with (1) libunwind,
   (2) backtrace, or (3) certain platforms with hardwired unwinding.
   @param reason optional string specifying the reason for traceback
   */
  virtual void traceback(const char *reason);
  /// Turn on or off debugging on a signel.  The default is on.
  virtual void set_debug_on_signal(int);
  /// Turn on or off traceback on a signel.  The default is on.
  virtual void set_traceback_on_signal(int);
  /// Turn on or off exit after a signel.  The default is on.
  virtual void set_exit_on_signal(int);
  /** Turn on or off running an infinite loop after the debugger is started.
      This loop gives the debugger a chance to attack to the process.
      The default is on. */
  virtual void set_wait_for_debugger(int);

  /// The Debugger will be activated when @c sig is caught.
  virtual void handle(int sig);
  /// Reverts the effect of @c handle(sig) , i.e. the Debugger will not be
  /// activated when @c sig is caught.
  virtual void release(int sig);
  /// This calls handle(int) with all of the major signals.
  virtual void handle_defaults();

  /// This sets a prefix which preceeds all messages printing by Debugger.
  virtual void set_prefix(const char *p);
  /// Set the prefix to the decimal represention of p followed by a ": ".
  virtual void set_prefix(int p);

  /** Sets the command to be exectuted when debug is called.
      The character sequence "$(EXEC)" is replaced by the executable
      name (see set_exec), "$(PID)" is replaced by the
      current process id, and "$(PREFIX)" is replaced by the
      prefix. */
  virtual void set_cmd(const char *);
  /// Calls set_cmd with a hopefully suitable default.
  virtual void default_cmd();
  /** Set the name of the executable for the current process.
      It is up to the programmer to set this, even if the Debugger
      is initialized with the KeyVal constructor. */
  virtual void set_exec(const char *);

  /// Called when signal sig is received.  This is mainly for internal use.
  virtual void got_signal(int sig);

  /// Set the global default debugger.  The initial value is null.
  static void set_default_debugger(const std::shared_ptr<Debugger> &);
  /// Return the global default debugger.
  static std::shared_ptr<Debugger> default_debugger();

 private:
  /// Replaces alias in cmd_ with its full form
  void resolve_cmd_alias();
};

/// Use this to launch GNU debugger in xterm
void launch_gdb_xterm();
/// Use this to launch LLVM debugger in xterm
void launch_lldb_xterm();

}  // namespace TiledArray

#endif // TILEDARRAY_UTIL_BUG_H_

// Local Variables:
// mode: c++
// c-file-style: "CLJ"
// End:
