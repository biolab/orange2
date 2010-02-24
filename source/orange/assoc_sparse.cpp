/*
    This file is part of Orange.
    
    Copyright 1996-2010 Faculty of Computer and Information Science, University of Ljubljana
    Contact: janez.demsar@fri.uni-lj.si

    Orange is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "assoc.hpp"
#include "examplegen.hpp"
#include "table.hpp"



/****************************************************************************************
TSparseExample
*****************************************************************************************/

TSparseExample::TSparseExample(TExample *example, int weightID){
	weight = WEIGHT(*example);
	length = 0;

  if (example->domain->variables->size()) {
	  // walk through all attributes in example and adding to sparseExample only those having some value
	  PITERATE(TVarList, vi, example->domain->variables)
		  if (   !(*example)[*vi].isSpecial() 
		      && (((*vi)->varType != TValue::FLOATVAR) || (*example)[*vi].floatV > 1e-6))
			  length++;

	  itemset = new long[length];
	  length = 0;

	  PITERATE(TVarList, vi2, example->domain->variables)
		  if (   !(*example)[*vi2].isSpecial() 
		      && (((*vi2)->varType != TValue::FLOATVAR) || (*example)[*vi2].floatV > 1e-6))
			  itemset[length++] = example->domain->getVarNum(*vi2);
  }
  else {
    length = 0;
    itemset = new long[example->meta.size() - (weightID ? 1 : 0)];
    ITERATE(TMetaValues, mi, example->meta)
      if ((*mi).first != weightID)
        itemset[length++] = (*mi).first;
    sort(itemset, itemset+length);
  }
}

/****************************************************************************************
TSparseExamples
*****************************************************************************************/

TSparseExamples::TSparseExamples(PExampleGenerator examples, int weightID){
	fullWeight = 0.0;
	TSparseExample *sparseExm;
	domain = examples->domain;

  const bool sparseExamples = examples->domain->variables->empty();
  set<long> ids;

	// walk through all examples converting them to sparseExample and add them to transaction
	PEITERATE(example, examples) {
			sparseExm = new TSparseExample(&*example, weightID);
      if (sparseExamples) {
        for(long *vi = sparseExm->itemset, le = sparseExm->length; le--; vi++)
          ids.insert(*vi);
      }
			transaction.push_back(sparseExm);
			fullWeight += sparseExm->weight;
  }

	// walk through all existing attributes in example and add them to intDomain
  if (sparseExamples) {
    intDomain.reserve(ids.size());
    ITERATE(set<long>, si, ids)
      intDomain.push_back(*si);
  }
  else {
    for(int i = 0, e = examples->domain->variables->size(); i!=e; i++)
      intDomain.push_back(i);
	}
}

/****************************************************************************************
TSparseItemsetNode
*****************************************************************************************/

TSparseItemsetNode::TSparseItemsetNode(long avalue) {
	weiSupp = 0.0;
	value = avalue;
};

TSparseItemsetNode *TSparseItemsetNode::operator[] (long avalue) {
	return subNode[avalue];
};

//if no subNode with that key exists add new
TSparseItemsetNode* TSparseItemsetNode::addNode(long avalue) {
	if (subNode.find(avalue)==subNode.end()) {
		subNode[avalue] = new TSparseItemsetNode(avalue);
		subNode[avalue]->parent = this;
	}
	//returns new node
	return subNode[avalue];
};

bool TSparseItemsetNode::hasNode(long avalue) {
	return (subNode.find(avalue)!=subNode.end());
};

/****************************************************************************************
TSparseItemsetTree
*****************************************************************************************/

// constructor
TSparseItemsetTree::TSparseItemsetTree(TSparseExamples examples){
	root = new TSparseItemsetNode();
	domain = examples.domain;
};

// generates all itemsets with one element
int TSparseItemsetTree::buildLevelOne(vector<long> intDomain) {
	int count = 0;

	ITERATE(vector<long>,idi,intDomain) {
		root->addNode(*idi);
		count++;
	}

	return count;
};

// generates candiate itemsets of size k from large itemsets of size k-1
long TSparseItemsetTree::extendNextLevel(int maxDepth, long maxCount) {
	typedef pair<TSparseItemsetNode *,int> NodeDepth; //<node,depth>

	long count = 0;
	vector<NodeDepth> nodeQue;

	long *cItemset = new long[maxDepth +1];
	int currDepth;
	TSparseItemsetNode *currNode;

	nodeQue.push_back(NodeDepth(root,0)); // put root in que

	while (!nodeQue.empty()) {			//repeats until que is empty
		currNode = nodeQue.back().first;			// node
		currDepth = nodeQue.back().second;			// depth

		nodeQue.pop_back();

		if (currDepth) cItemset[currDepth - 1] = currNode->value;		// generates candidate itemset

		if (currDepth == maxDepth) 										// we found an instance that can be extended
			for(TSparseISubNodes::iterator iter(++(root->subNode.find(currNode->value))), \
											   iter_end(root->subNode.end()); \
				iter!=iter_end; \
				iter++) {
					cItemset[currDepth] = iter->second->value;

					if (allowExtend(cItemset, currDepth + 1)) {
						currNode->addNode(cItemset[currDepth]);
						count++;
						if (count>maxCount) return count;
					}

				}
		else RITERATE(TSparseISubNodes,sni,currNode->subNode)		//adds subnodes to list
			nodeQue.push_back(NodeDepth(sni->second, currDepth + 1));
	}
	return count;
};


// tests if some candidate itemset can be extended to large itemset
bool TSparseItemsetTree::allowExtend(long itemset[], int iLength) {
	typedef pair<int,int> IntPair; // <parent node index, depth>
	typedef pair<TSparseItemsetNode *,IntPair> NodeDepth;

	vector<NodeDepth> nodeQue;

	int currDepth;
	int currPrIndex;								//parent index
	TSparseItemsetNode *currNode;
	int i;

	nodeQue.push_back(NodeDepth(root,IntPair(-1,1))); // put root in que

	while (!nodeQue.empty()) {						//repeats until que is empty
		currNode = nodeQue.back().first;			// node
		currPrIndex = nodeQue.back().second.first;	// parentIndex
		currDepth = nodeQue.back().second.second;	// depth

		nodeQue.pop_back();

		if (currDepth == iLength) continue;			// we found an instance

		for (i = currDepth; i!=currPrIndex; i--)		//go through all posible successors of this node
			if (currNode->hasNode(itemset[i]))
				nodeQue.push_back(NodeDepth((*currNode)[itemset[i]],IntPair(i,currDepth + 1)));
			else return 0;
	}
	return 1;
}


// counts number of leaf nodes not using any recursion
long TSparseItemsetTree::countLeafNodes() {
	long countLeaf = 0;
	vector<TSparseItemsetNode *> nodeQue;
	TSparseItemsetNode *currNode;

	nodeQue.push_back(root);

	while (!nodeQue.empty()) {					//repeats until que is empty
		currNode = nodeQue.back();
		nodeQue.pop_back();

		if (!currNode->subNode.empty()) 		//if node is leaf count++ else count children
			RITERATE(TSparseISubNodes,sni,currNode->subNode)
				nodeQue.push_back(sni->second);
		else countLeaf++;						// node is leaf
	}

	return countLeaf;
};


// counts supports of all aimLength long branches in tree using one example (itemset) data
void TSparseItemsetTree::considerItemset(long itemset[], int iLength, float weight, int aimLength) {
	typedef pair<int,int> IntPair; // <parent node index, depth>
	typedef pair<TSparseItemsetNode *,IntPair> NodeDepth;

	vector<NodeDepth> nodeQue;

	int currDepth;
	int currPrIndex;								//parent index
	TSparseItemsetNode *currNode;
	int i, end = iLength - aimLength;

	nodeQue.push_back(NodeDepth(root,IntPair(-1,0))); // put root in que

	while (!nodeQue.empty()) {						//repeats until que is empty
		currNode = nodeQue.back().first;			// node
		currPrIndex = nodeQue.back().second.first;	// parentIndex
		currDepth = nodeQue.back().second.second;	// depth

		nodeQue.pop_back();

		if (currDepth == aimLength) { currNode->weiSupp += weight; continue;}	// we found an instance
		if (currNode->subNode.empty()) continue;	// if node does't have successors

		for (i = currDepth + end; i!=currPrIndex; i--)		//go through all posible successors of this node
			if (currNode->hasNode(itemset[i]))
				nodeQue.push_back(NodeDepth((*currNode)[itemset[i]],IntPair(i,currDepth + 1)));
	}
};

// counts supports of all aimLength long branches in tree using examples data
void TSparseItemsetTree::considerExamples(TSparseExamples *examples, int aimLength) {
		ITERATE(vector<TSparseExample*>,ei,examples->transaction)
			if (aimLength <= (*ei)->length)
				considerItemset((*ei)->itemset, (*ei)->length, (*ei)->weight, aimLength);
}

void TSparseItemsetTree::assignExamples(TSparseItemsetNode *node, long *itemset, long *itemsetend, const int exampleId)
{
  node->exampleIds.push_back(exampleId);
  if (!node->subNode.empty())
    for(; itemset != itemsetend; itemset++)
      if (node->hasNode(*itemset))
        assignExamples((*node)[*itemset], itemset+1, itemsetend, exampleId);
}

void TSparseItemsetTree::assignExamples(TSparseExamples &examples)
{
  int exampleId = 0;
  ITERATE(vector<TSparseExample*>,ei,examples.transaction)
    assignExamples(root, (*ei)->itemset, (*ei)->itemset+(*ei)->length, exampleId++);
}



// deletes all leaves that have weiSupp smaler than given minSupp;
void TSparseItemsetTree::delLeafSmall(float minSupp) {
	long countLeaf = 0;
	vector<TSparseItemsetNode *> nodeQue;
	TSparseItemsetNode *currNode;

	nodeQue.push_back(root);

	while (!nodeQue.empty()) {			//repeats until que is empty
		currNode = nodeQue.back();
		nodeQue.pop_back();

		if (!currNode->subNode.empty()) 	//if node is not leaf add children else check support
			RITERATE(TSparseISubNodes,sni,currNode->subNode)
				nodeQue.push_back(sni->second);
		else
			if (currNode->weiSupp < minSupp) {
				currNode->parent->subNode.erase(currNode->value);
				delete currNode;
			}
	}
};


// generates all possible association rules from tree using given confidence
PAssociationRules TSparseItemsetTree::genRules(int maxDepth, float minConf, float nOfExamples, bool storeExamples) {
	typedef pair<TSparseItemsetNode *,int> NodeDepth; //<node,depth>

	int count=0;
	vector<NodeDepth> nodeQue;

	PAssociationRules rules = mlnew TAssociationRules();

	long *itemset = new long[maxDepth];
	int currDepth;
	TSparseItemsetNode *currNode;

	nodeQue.push_back(NodeDepth(root,0)); // put root in que

	while (!nodeQue.empty()) {						//repeats until que is empty
		currNode = nodeQue.back().first;			// node
		currDepth = nodeQue.back().second;			// depth

		nodeQue.pop_back();

		if (currDepth) itemset[currDepth - 1] = currNode->value;  // create itemset to check for confidence

		if (currDepth > 1)
			count += getItemsetRules(itemset, currDepth, minConf, currNode->weiSupp, nOfExamples, rules, storeExamples, currNode);	//creates rules from itemsets and adds them to rules

		RITERATE(TSparseISubNodes,sni,currNode->subNode)		//adds subnodes to list
			nodeQue.push_back(NodeDepth(sni->second, currDepth + 1));
	}

	return rules;
};

// checks if itemset generates some rules with enough confidence and adds these rules to resultset
long TSparseItemsetTree::getItemsetRules(long itemset[], int iLength, float minConf,
								   float nAppliesBoth, float nOfExamples,
								   PAssociationRules rules,
								   bool storeExamples, TSparseItemsetNode *bothNode) {

	float nAppliesLeft, nAppliesRight;
	long count = 0;
	PAssociationRule rule;
	TExample exLeft(domain), exRight(domain);
  const bool sparseRules = domain->variables->empty();

	nAppliesLeft=nAppliesBoth;
	nAppliesRight=nAppliesBoth;

	typedef pair<int,int> IntPair; // <parent node index, depth>
	typedef pair<TSparseItemsetNode *,IntPair> NodeDepth;

	vector<NodeDepth> nodeQue;

	int currDepth, i, j;
	int currPrIndex; //parent index
	TSparseItemsetNode *currNode, *tempNode;

	long *leftItemset = new long[iLength - 1];
	float thisConf;

	nodeQue.push_back(NodeDepth(root,IntPair(-1,0))); // put root in que

	while (!nodeQue.empty()) {			//repeats until que is empty
		currNode = nodeQue.back().first;			// node
		currPrIndex = nodeQue.back().second.first;	// parentIndex
		currDepth = nodeQue.back().second.second;	// depth

		nodeQue.pop_back();

		nAppliesLeft = currNode->weiSupp;			// support of left side
		thisConf = nAppliesBoth/nAppliesLeft;

		if (thisConf >= minConf) {	// if confidence > min confidence do ... else don't folow this branch
			if (currDepth) {
				leftItemset[currDepth-1] = currNode->value;

        if (sparseRules) {
          PExample exLeftS = mlnew TExample(domain, false);
          PExample exRightS = mlnew TExample(domain, false);

   			  tempNode = root;
          for(i=0, j=0; (i<currDepth) && (j<iLength); ) {
            if (itemset[j] < leftItemset[i]) {
              exRightS->setMeta(itemset[j], TValue(1.0));
              tempNode = (*tempNode)[itemset[j]];
              j++;
            }
            else {
              _ASSERT(itemset[j] == leftItemset[i]);
              exLeftS->setMeta(leftItemset[i], TValue(1.0));
              i++;
              j++;
            }
          }

          _ASSERT(i==currDepth);
          for(; j<iLength; j++) {
            exRightS->setMeta(itemset[j], TValue(1.0));
            tempNode = (*tempNode)[itemset[j]];
          }

/*
  				for (i=0; i<currDepth; i++)		//generating left and right example and get support of left side
					  exLeft[leftItemset[i]] = 1.0;

          tempNode = root;
          for (i=0; i< iLength; i++)
            if (   ) {
              exRight[itemset[i]] = 1.0;
              tempNode = (*tempNode)[itemset[i]];
            }
*/
				  nAppliesRight = tempNode->weiSupp;	//get support of left side

				  //add rules
				  rule = mlnew TAssociationRule(exLeftS, exRightS, nAppliesLeft, nAppliesRight, nAppliesBoth, nOfExamples, currDepth, iLength-currDepth);
				  if (storeExamples) {
				    rule->matchLeft = new TIntList(currNode->exampleIds);
				    rule->matchBoth = new TIntList(bothNode->exampleIds);
				  }

				  rules->push_back(rule);
				  count ++;
        }
        else {
  				for (i=0; i<currDepth;i++) {		//generating left and right example and get support of left side
					  exLeft[leftItemset[i]].setSpecial(false);
					  exLeft[leftItemset[i]].varType=0;
					}


				  tempNode = root;
				  for (i=0; i<iLength;i++)
					  if (exLeft[itemset[i]].isSpecial()) {
						  exRight[itemset[i]].setSpecial(false);
						  exRight[itemset[i]].varType=0;
						  tempNode = (*tempNode)[itemset[i]];
					  }

				  nAppliesRight = tempNode->weiSupp;	//get support of left side

				  //add rules
				  rule = mlnew TAssociationRule(mlnew TExample(exLeft), mlnew TExample(exRight), nAppliesLeft, nAppliesRight, nAppliesBoth, nOfExamples, currDepth, iLength-currDepth);
				  rules->push_back(rule);
				  count ++;

				  for (i=0; i<iLength;i++) {					//deleting left and right example
					  exLeft[itemset[i]].setSpecial(true);
					  exLeft[itemset[i]].varType=1;
					  exRight[itemset[i]].setSpecial(true);
					  exRight[itemset[i]].varType=1;
				  }
			  }
      }

			if (currDepth < iLength - 1)							//if we are not too deep
				for (i = iLength - 1; i!=currPrIndex; i--)		//go through all posible successors of this node
					if (currNode->hasNode(itemset[i]))				//if this node exists among childrens
						nodeQue.push_back(NodeDepth((*currNode)[itemset[i]],IntPair(i,currDepth + 1)));
		}
	}

	return count;
};

/****************************************************************************************
TAssociationRulesSparseInducer
*****************************************************************************************/

TAssociationRulesSparseInducer::TAssociationRulesSparseInducer(float asupp, float aconf, int awei)
: maxItemSets(15000),
  confidence(aconf),
  support(asupp),
  nOfExamples(0.0),
  storeExamples(false)
{}


TSparseItemsetTree *TAssociationRulesSparseInducer::buildTree(PExampleGenerator examples, const int &weightID, long &i, float &fullWeight)
{
	float nMinSupp;
	long currItemSets, newItemSets;

	// reformat examples in sparseExm for better efficacy
	TSparseExamples sparseExm(examples, weightID);

	fullWeight = sparseExm.fullWeight;

	// build first level of tree
	TSparseItemsetTree *tree = new TSparseItemsetTree(sparseExm);
	newItemSets = tree->buildLevelOne(sparseExm.intDomain);

	nMinSupp = support * sparseExm.fullWeight;

	//while it is possible to extend tree repeat...
	for(i=1;newItemSets;i++) {
		tree->considerExamples(&sparseExm,i);
		tree->delLeafSmall(nMinSupp);

		currItemSets = tree->countLeafNodes();

		newItemSets = tree->extendNextLevel(i, maxItemSets - currItemSets);

		//test if tree is too large
		if (newItemSets + currItemSets >= maxItemSets) {
			raiseError("too many itemsets (%i); increase 'support' or 'maxItemSets'", maxItemSets);
			newItemSets = 0;
		}
	}

	if (storeExamples)
	  tree->assignExamples(sparseExm);

	return tree;
}


PAssociationRules TAssociationRulesSparseInducer::operator()(PExampleGenerator examples, const int &weightID)
{
  long i;
  float fullWeight;
  TSparseItemsetTree *tree = buildTree(examples, weightID, i, fullWeight);
  PAssociationRules rules = tree->genRules(i, confidence, fullWeight, storeExamples);
  delete tree;

  if (storeExamples) {
    PExampleTable xmpls = mlnew TExampleTable(examples);
    PITERATE(TAssociationRules, ri, rules)
      (*ri)->examples = xmpls;
  }
  return rules;
}




/****************************************************************************************
TItemsetsSparseInducer
*****************************************************************************************/

TItemsetsSparseInducer::TItemsetsSparseInducer(float asupp, int awei)
: maxItemSets(15000),
  support(asupp),
  nOfExamples(0.0),
  storeExamples(false)
{}

PSparseItemsetTree TItemsetsSparseInducer::operator()(PExampleGenerator examples, const int &weightID)
{	float nMinSupp;
	long currItemSets, i,newItemSets;

	// reformat examples in sparseExm for better efficacy
	TSparseExamples sparseExm(examples, weightID);

	// build first level of tree
	TSparseItemsetTree *tree = new TSparseItemsetTree(sparseExm);
	newItemSets = tree->buildLevelOne(sparseExm.intDomain);

	nMinSupp = support * sparseExm.fullWeight;

	//while it is posible to extend tree repeat...
	for(i=1;newItemSets;i++) {
		tree->considerExamples(&sparseExm,i);
		tree->delLeafSmall(nMinSupp);

		currItemSets = tree->countLeafNodes();

		newItemSets = tree->extendNextLevel(i, maxItemSets - currItemSets);

		//test if tree is too large
		if (newItemSets + currItemSets >= maxItemSets) {
			raiseError("too many itemsets (%i); increase 'support' or 'maxItemSets'", maxItemSets);
			newItemSets = 0;
		}
	}

  if (storeExamples)
	  tree->assignExamples(sparseExm);

	return tree;
}
