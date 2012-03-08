#ifndef TILEDARRAY_EVENT_LOG_H__INCLUDED
#define TILEDARRAY_EVENT_LOG_H__INCLUDED

#include <TiledArray/error.h>
#include <world/worldtime.h>
#include <vector>
#include <fstream>


namespace TiledArray {
  namespace logging {

    class EventLog : public madness::CallbackInterface {
    private:
      static double start_;
      static double finish_;

      std::string name_;
      std::vector<double> events_;
      madness::AtomicInt count_;
    public:
      EventLog(const std::string& name, std::size_t n) :
          name_(name), events_(n, 0.0)
      {
        count_ = 0;
      }

      EventLog(const EventLog& other) :
          name_(other.name_), events_(other.events_)
      { count_ = other.count_; }

      EventLog& operator=(EventLog& other) {
        name_ = other.name_;
        events_ = other.events_;
        count_ = other.count_;

        return *this;
      }

      const std::string& name() const { return name_; }

      static double set_start() { return start_ = madness::wall_time(); }
      static double get_start() { return start_; }

      static double set_finis() { return finish_ = madness::wall_time(); }
      static double get_finish() { return finish_; }

      virtual void notify() {
        const long count = count_++;
        events_[count] = madness::wall_time();
        TA_ASSERT(count < events_.size());
      }

      friend std::ostream& operator<<(std::ostream& os, const EventLog& el) {
        os << el.name_ << "\n";
        if(el.events_.empty())
          return os;
        const double delta = (finish_ - start_) / 30.0;
        double start = start_;
        double finish = start_ + delta;
        while(start < finish_) {
          std::size_t n = 0ul;
          for(std::vector<double>::const_iterator it = el.events_.begin(); it != el.events_.end(); ++it)
            if((*it >= start) && (*it < finish))
              ++n;
          os << start - start_ << ", " << n << "\n";
          start = finish;
          finish += delta;
        }

        return os;
      }

    }; // class EventLog

#ifdef TILEDARRAY_INSTANTIATE_STATIC_DATA
    double EventLog::start_ = 0.0;
    double EventLog::finish_ = 0.0;
#endif // TILEDARRAY_INSTANTIATE_STATIC_DATA

  }  // namespace logging
} // namespace TiledArray

#endif // TILEDARRAY_TIME_LOG_H__INCLUDED
