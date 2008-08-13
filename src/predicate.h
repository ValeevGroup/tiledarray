#ifndef PREDICATE_H__INCLUDED
#define PREDICATE_H__INCLUDED

template<unsigned int DIM>
class AbstractPrdicate
{
public:
	inline bool
	operator ()(const Tuple<DIM>& tup)
		{return this->included(tup);}
	
	inline bool
	included(const Tuple<DIM>& tup) = 0;
};

template<unsigned int DIM>
class DensePredicate
{
public:
	
	inline bool
	included(const Tuple<DIM>& tup)
		{return true;}
};

template<unsigned int DIM>
class DiagonalPredicate
{
public:
	inline bool
	included(const Tuple<DIM>& tup)
	{
		assert(DIM >= 2);
		return (tup[0] == tup[1]);
	}
};

#endif // PREDICATE_H__INCLUDED
