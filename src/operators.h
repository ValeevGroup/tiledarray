#ifndef OPERATOR_H_
#define OPERATOR_H_


template<typename T>
class MathOp
{
	
	const T
	Negate(const T& data)
	{
		 return -data;
	}
	
	const T
	Sum(const T& data1, const T& data2)
	{
		return (data1 + data2);
	}

	const T
	Subtract(const T& data1, const T& data2) const
	{
		return (data1 - data2);
	}

	const T
	Multiply(const T& data1, const T& data2) const
	{
		return (data1 * data2);
	}
	
	const T
	Divide(const T& data1, const T& data2) const
	{
		return (data1 / data2);
	}
};

#endif /*OPERATOR_H_*/
