#ifndef __CLAS_GEN_HPP
#define __CLAS_GEN_HPP

#include "examplegen.hpp"


/*  TClassifier generator constructs examples. For each attribute in examples, values are set by using TVariable's
    firstValue and nextValue methods. The class of example is determined by using a TClassifier object. */
class ORANGE_API TClassifierGenerator : public TExampleGenerator {
public:
  __REGISTER_CLASS

  PClassifier classify; //P classifier

  TClassifierGenerator();
  TClassifierGenerator(PDomain &);
  TClassifierGenerator(PDomain &, PClassifier &clsf);

  virtual TExampleIterator begin();
  virtual bool randomExample(TExample &);

  virtual int numberOfExamples();

  virtual TValue operator()(const TExample &);

protected:
  virtual void increaseIterator(TExampleIterator &);
  virtual bool sameIterators(const TExampleIterator &, const TExampleIterator &);
  virtual void deleteIterator(TExampleIterator &data);
  virtual void copyIterator(const TExampleIterator &src, TExampleIterator &dest);
};


/*  TClassifier generator that constructs examples by choosing random values of attributes and determining the
    class a TClassifier object. The number of examples can be infinite or limited */
class ORANGE_API TClassifierRandomGenerator : public TExampleGenerator {
public:
  __REGISTER_CLASS

  PClassifier classify; //P classifier
  int noOfEx; //P number of examples

  TClassifierRandomGenerator();
  TClassifierRandomGenerator(PDomain &, int =-1);
  TClassifierRandomGenerator(PDomain &, PClassifier &clsf, int =-1);

  virtual TExampleIterator begin();
  virtual bool randomExample(TExample &);

  virtual int numberOfExamples();

protected:
  virtual void increaseIterator(TExampleIterator &);
  virtual bool sameIterators(const TExampleIterator &, const TExampleIterator &);
  virtual void deleteIterator(TExampleIterator &data);
  virtual void copyIterator(const TExampleIterator &src, TExampleIterator &dest);
};

#endif
