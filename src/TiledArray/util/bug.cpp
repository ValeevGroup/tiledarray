//
// bug.cpp
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

#include "bug.h"

#include <unistd.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <sstream>

#include "TiledArray/external/madness.h"
#include "TiledArray/util/backtrace.h"

// usually in signal.h, but not always.
#ifndef NSIG
#define NSIG 100
#endif

namespace TiledArray {

//////////////////////////////////////////////////////////////////////
// static variables

static Debugger *signals[NSIG];

//////////////////////////////////////////////////////////////////////
// Debugger class definition

std::shared_ptr<Debugger> Debugger::default_debugger_(nullptr);

Debugger::Debugger(const char *exec) {
  init();

  debug_ = true;
  traceback_ = true;
  exit_on_signal_ = true;
  sleep_ = false;
  wait_for_debugger_ = true;
  default_cmd();
  prefix_ = "";
  handle_sigint_ = false;
  handle_defaults();

  set_exec(exec);
  resolve_cmd_alias();
}

Debugger::~Debugger() {
  for (int i = 0; i < NSIG; i++) {
    if (mysigs_[i]) signals[i] = nullptr;
  }
  delete[] mysigs_;
}

void Debugger::init() {
  exec_.resize(0);
  prefix_.resize(0);
  cmd_.resize(0);
  sleep_ = 0;

  exit_on_signal_ = 1;
  traceback_ = 1;
  debug_ = 1;
  wait_for_debugger_ = 1;

  mysigs_ = new int[NSIG];
  for (int i = 0; i < NSIG; i++) {
    mysigs_[i] = 0;
  }
}

namespace {
static void handler(int sig) {
  if (signals[sig]) signals[sig]->got_signal(sig);
}
}  // namespace

void Debugger::handle(int sig) {
  if (sig >= NSIG) return;
  typedef void (*handler_type)(int);
  signal(sig, (handler_type)handler);
  signals[sig] = this;
  mysigs_[sig] = 1;
}

void Debugger::release(int sig) {
  if (sig >= NSIG) return;
  signal(sig, SIG_DFL);
  signals[sig] = nullptr;
  mysigs_[sig] = 0;
}

void Debugger::handle_defaults() {
#ifdef SIGSEGV
  handle(SIGSEGV);
#endif
#ifdef SIGFPE
  handle(SIGFPE);
#endif
#ifdef SIGQUIT
  handle(SIGQUIT);
#endif
#ifdef SIGIOT
  handle(SIGIOT);
#endif
#ifdef SIGINT
  if (handle_sigint_) handle(SIGINT);
#endif
#ifdef SIGHUP
  handle(SIGHUP);
#endif
#ifdef SIGBUS
  handle(SIGBUS);
#endif
#ifdef SIGABRT
  handle(SIGABRT);
#endif
#ifdef SIGTRAP
  handle(SIGTRAP);
#endif
}

void Debugger::set_exec(const char *exec) {
  if (exec) {
    exec_ = exec;
  } else {
    exec_.resize(0);
  }
}

void Debugger::set_prefix(const char *p) {
  if (p) {
    prefix_ = p;
  } else {
    prefix_.resize(0);
  }
}

void Debugger::set_prefix(int i) {
  char p[128];
  snprintf(p, sizeof(p), "%3d: ", i);
  set_prefix(p);
}

void Debugger::default_cmd() {
  int has_x11_display = (getenv("DISPLAY") != 0);

  if (has_x11_display) {
    set_cmd("gdb_xterm");
  } else {
    set_cmd(0);
  }
}

void Debugger::resolve_cmd_alias() {
  if (cmd_ == "gdb_xterm") {
    cmd_ =
        "xterm -title \"$(PREFIX)$(EXEC)\" -e gdb -ex \"set variable "
        "debugger_ready_=1\" --pid=$(PID) $(EXEC) &";
  } else if (cmd_ == "lldb_xterm") {
    cmd_ =
        "xterm -title \"$(PREFIX)$(EXEC)\" -e lldb -p $(PID) -o \"expr "
        "debugger_ready_=1\" &";
  }
}

void Debugger::set_cmd(const char *cmd) {
  if (cmd) {
    cmd_ = cmd;
    resolve_cmd_alias();
  } else {
    cmd_.resize(0);
  }
}

void Debugger::debug(const char *reason) {
  std::cout << prefix_ << "Debugger::debug: ";
  if (reason)
    std::cout << reason;
  else
    std::cout << "no reason given";
  std::cout << std::endl;

  if (!cmd_.empty()) {
    int pid = getpid();
    // contruct the command name
    std::string cmd = cmd_;
    std::string::size_type pos;
    std::string pidvar("$(PID)");
    while ((pos = cmd.find(pidvar)) != std::string::npos) {
      std::string pidstr;
      pidstr += std::to_string(pid);
      cmd.replace(pos, pidvar.size(), pidstr);
    }
    std::string execvar("$(EXEC)");
    while ((pos = cmd.find(execvar)) != std::string::npos) {
      cmd.replace(pos, execvar.size(), exec_);
    }
    std::string prefixvar("$(PREFIX)");
    while ((pos = cmd.find(prefixvar)) != std::string::npos) {
      cmd.replace(pos, prefixvar.size(), prefix_);
    }

    // start the debugger
    // before starting the debugger de-register signal handler for SIGTRAP to
    // let the debugger take over
    release(SIGTRAP);
    std::cout << prefix_ << "Debugger: starting \"" << cmd << "\"" << std::endl;
    debugger_ready_ = 0;
    const auto system_retvalue = system(cmd.c_str());
    if (system_retvalue != 0) {  // call to system() failed
      std::cout << prefix_
                << "Failed debugger launch: system() did not succeed ..."
                << std::endl;
    } else {  // call to system() succeeded
      // wait until the debugger is ready
      if (sleep_) {
        std::cout << prefix_ << "Sleeping " << sleep_
                  << " seconds to wait for debugger ..." << std::endl;
        sleep(sleep_);
      }
      if (wait_for_debugger_) {
        std::string make_ready_message;
        if (cmd_.find(" gdb ") != std::string::npos ||
            cmd_.find(" lldb ") != std::string::npos) {
          make_ready_message =
              " configure debugging session (set breakpoints/watchpoints, "
              "etc.) then type 'c' to continue running";
        }

        std::cout << prefix_ << ": waiting for the user ..."
                  << make_ready_message << std::endl;
        while (!debugger_ready_)
          ;
      }
    }
  }
}

void Debugger::got_signal(int sig) {
  const char *signame;
  if (sig == SIGSEGV)
    signame = "SIGSEGV";
  else if (sig == SIGFPE)
    signame = "SIGFPE";
  else if (sig == SIGHUP)
    signame = "SIGHUP";
  else if (sig == SIGINT)
    signame = "SIGINT";
  else if (sig == SIGABRT)
    signame = "SIGABRT";
#ifdef SIGBUS
  else if (sig == SIGBUS)
    signame = "SIGBUS";
#endif
  else if (sig == SIGTRAP)
    signame = "SIGTRAP";
  else
    signame = "UNKNOWN SIGNAL";

  if (traceback_) {
    traceback(signame);
  }
  if (debug_) {
    debug(signame);
  }

  if (exit_on_signal_) {
    std::cout << prefix_ << "Debugger: exiting" << std::endl;
    exit(1);
  } else {
    std::cout << prefix_ << "Debugger: continuing" << std::endl;
  }

  // handle(sig);
}

void Debugger::set_debug_on_signal(int v) { debug_ = v; }

void Debugger::set_traceback_on_signal(int v) { traceback_ = v; }

void Debugger::set_wait_for_debugger(int v) { wait_for_debugger_ = v; }

void Debugger::set_exit_on_signal(int v) { exit_on_signal_ = v; }

void Debugger::set_default_debugger(const std::shared_ptr<Debugger> &d) {
  default_debugger_ = d;
}

std::shared_ptr<Debugger> Debugger::default_debugger() {
  return default_debugger_;
}

#define SIMPLE_STACK \
  (defined(linux) && defined(i386)) || (defined(__OSF1__) && defined(i860))

void Debugger::traceback(const char *reason) {
  Debugger::__traceback(prefix_, reason);
}

void Debugger::__traceback(const std::string &prefix, const char *reason) {
  detail::Backtrace result(prefix);
  const size_t nframes_to_skip = 2;
#if defined(HAVE_LIBUNWIND)
  std::cout << prefix << "Debugger::traceback(using libunwind):";
#elif defined(HAVE_BACKTRACE)  // !HAVE_LIBUNWIND
  std::cout << prefix << "Debugger::traceback(using backtrace):";
#else                          // !HAVE_LIBUNWIND && !HAVE_BACKTRACE
#if defined(SIMPLE_STACK)
  std::cout << prefix << "Debugger::traceback:";
#else
  std::cout << prefix << "traceback not available for this arch" << std::endl;
  return;
#endif  // SIMPLE_STACK
#endif  // HAVE_LIBUNWIND, HAVE_BACKTRACE

  if (reason)
    std::cout << reason;
  else
    std::cout << "no reason given";
  std::cout << std::endl;

  if (result.empty())
    std::cout << prefix << "backtrace returned no state information"
              << std::endl;
  else
    std::cout << result.str(nframes_to_skip) << std::endl;
}

void create_debugger(const char *cmd, const char *exec, std::int64_t rank) {
  auto debugger = std::make_shared<TiledArray::Debugger>();
  if (cmd) debugger->set_cmd(cmd);
  if (exec) debugger->set_exec(exec);
  if (rank < 0) rank = TiledArray::get_default_world().rank();
  debugger->set_prefix(rank);
  Debugger::set_default_debugger(debugger);
}

void launch_gdb_xterm(const char *exec, std::int64_t rank) {
  create_debugger("gdb_xterm", exec, rank);
  Debugger::default_debugger()->debug("Starting gdb ...");
}

void launch_lldb_xterm(const char *exec, std::int64_t rank) {
  create_debugger("lldb_xterm", exec, rank);
  Debugger::default_debugger()->debug("Starting lldb ...");
}

}  // namespace TiledArray

/////////////////////////////////////////////////////////////////////////////
// Local Variables:
// mode: c++
// c-file-style: "CLJ"
// End:
