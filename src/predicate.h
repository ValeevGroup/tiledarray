#ifndef PREDICATE_H__INCLUDED
#define PREDICATE_H__INCLUDED

template<unsigned int DIM>
class AbstractPrdicate
{
public:
	virtual inline Tuple<DIM>&
	increment(Tuple<DIM>& it, const Tuple<DIM>& high, const Tuple<DIM>& low)
	{
		assert(false);	// Not implemented
		return it;
	}
	
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
	virtual inline Tuple<DIM>&
	increment(Tuple<DIM>& it, const Tuple<DIM>& high, const Tuple<DIM>& low)
	{
		return VectorOps<Tuple<DIM>, DIM>(it, high, low);
	}
	
	virtual inline bool
	included(const Tuple<DIM>& tup)
		{return true;}
};


#endif // PREDICATE_H__INCLUDED
