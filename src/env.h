#ifndef ENV_H_
#define ENV_H_

namespace TiledArray {

  /// Describes a runtime environment
  class RuntimeEnvironment {
    public:
      static RuntimeEnvironment& Instance();
      virtual ~RuntimeEnvironment();

      /// number of processors
      virtual unsigned int nproc() const =0;
      /// which processor am I?
      virtual unsigned int me() const =0;
      
    protected:
      static RuntimeEnvironment* instance_;
      RuntimeEnvironment();
  };

  /// can be used to test TiledArray
  class TestRuntimeEnvironment : public RuntimeEnvironment {
    public:
      static void CreateInstance(unsigned int nproc, unsigned int me);
      static void DestroyInstance();
      ~TestRuntimeEnvironment();
      
      /// Implementation of RuntimeEnvironment::nproc()
      unsigned int nproc() const;
      /// Implementation of RuntimeEnvironment::me()
      unsigned int me() const;
      
    private:
      TestRuntimeEnvironment(unsigned int nproc, unsigned int me);
      
      unsigned int nproc_;
      unsigned int me_;
  };

}

#endif /*ENV_H_*/
