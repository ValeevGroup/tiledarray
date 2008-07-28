#ifndef PROCESS_H_
#define PROCESS_H_

/** 
 * Forward declaration of the functionality provided by the Default trait.
 */
namespace TILED_ARRAY_NAMESPACE
{

class Process
{ 
public:
	
	virtual void barrier() = 0;
    virtual int nPlaces() = 0;
    virtual int myPlace() = 0;
    virtual int nullPlace() = 0;
    /** home of the root of every HTA */
    virtual int rootPlace() = 0;
    virtual void sync() = 0;
    virtual void async()= 0;
    virtual void init(int argc, char** argv) = 0;
    virtual void finalize() = 0;

    virtual std::ostream& getStdout() = 0;
};


class SerialProcess : public Process
{
	virtual void
	barrier()
	{}

	virtual int
	nPlaces()
	{
		return 1;
	}

	virtual int
	myPlace()
	{
		return 0;
	}

	virtual int 
	nullPlace()
	{
		return -1;
	} 

	virtual int
	rootPlace()
	{
		return 0;
	} 

	virtual void
	sync()
	{}

	virtual void
	async()
	{}

	virtual void
	init(int argc, char** argv)   
	{ 
#ifdef HTA_DEBUG_COUNT
//		Counters::enable();
#endif // HTA_DEBUG_COUNT   
	}

	virtual void
	SerialProcess::finalize ()                    
	{ 
//		Finalizable::finalizeAll();

#ifdef HTA_DEBUG_TIME
//		Timers::printAll();
#endif // HTA_DEBUG_TIME

#ifdef HTA_DEBUG_COUNT
//		Counters::printAll();
#endif // HTA_DEBUG_COUNT

#ifdef HTA_DEBUG_MEM
//		Alloc::undeleted();
#endif // HTA_DEBUG_MEM

	}


};


} // TILED_ARRAY_NAMESPACE


#endif // PROCESS_H_
