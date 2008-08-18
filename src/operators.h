#ifndef OPERATOR_H__INCLUDED
#define OPERATOR_H__INCLUDED


template<typename T>
class MathOp
{
	
	static inline T
	Negate(const T& data)
	{
		 return -data;
	}
	
	static inline T
	Sum(const T& data1, const T& data2)
	{
		return (data1 + data2);
	}

	static inline T
	Subtract(const T& data1, const T& data2)
	{
		return (data1 - data2);
	}

	static inline T
	Multiply(const T& data1, const T& data2)
	{
		return (data1 * data2);
	}
	
	static inline T
	Divide(const T& data1, const T& data2)
	{
		return (data1 / data2);
	}
};

#endif /*OPERATOR_H_*/
