#ifndef PREDICATE_H__INCLUDED
#define PREDICATE_H__INCLUDED

template<unsigned int DIM>
class AbstractPredicate
{
public:
	
	inline bool
	operator ()(const Tuple<DIM>& tup)
		{return this->included(tup);}
	
	virtual inline bool
	included(const Tuple<DIM>& tup) = 0;
};

template<unsigned int DIM>
class DensePredicate
{
public:

	virtual inline bool
	included(const Tuple<DIM>& tup)
		{return true;}
};


#endif // PREDICATE_H__INCLUDED
