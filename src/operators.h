#ifndef OPERATOR_H__INCLUDED
#define OPERATOR_H__INCLUDED


template<typename T>
class MathOp
{
	
	static inline const T
	Negate(const T& data)
	{
		 return -data;
	}
	
	static inline const T
	Sum(const T& data1, const T& data2)
	{
		return (data1 + data2);
	}

	static inline const T
	Subtract(const T& data1, const T& data2)
	{
		return (data1 - data2);
	}

	static inline const T
	Multiply(const T& data1, const T& data2)
	{
		return (data1 * data2);
	}
	
	static inline const T
	Divide(const T& data1, const T& data2)
	{
		return (data1 / data2);
	}
};

#endif /*OPERATOR_H_*/
