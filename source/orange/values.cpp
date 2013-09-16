#include "values.ppp"

void TSomeValue::val2str(string &) const
{}


void TSomeValue::str2val(const string &) const
{}


bool TSomeValue::operator ==(const TSomeValue &v) const
{ return !compare(v); }


bool TSomeValue::operator !=(const TSomeValue &v) const
{ return (compare(v)!=0); }
