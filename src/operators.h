#ifndef OPERATOR_H__INCLUDED
#define OPERATOR_H__INCLUDED


template<typename T>
class MathOp
{
public:
	static inline T
	negate(const T& data)
	{
		 return -data;
	}
	
	static inline T
	sum(const T& data1, const T& data2)
	{
		return (data1 + data2);
	}

	static inline T
	subtract(const T& data1, const T& data2)
	{
		return (data1 - data2);
	}

	static inline T
	multiply(const T& data1, const T& data2)
	{
		return (data1 * data2);
	}
	
	static inline T
	divide(const T& data1, const T& data2)
	{
		return (data1 / data2);
	}

	static inline bool
	equal(const T& data1, const T& data2)
	{
		return (data1 == data2);
	}

	static inline bool
	not_equal(const T& data1, const T& data2)
	{
		return (data1 != data2);
	}

};

#endif /*OPERATOR_H_*/
