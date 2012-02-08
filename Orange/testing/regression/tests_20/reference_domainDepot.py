# Description: Shows how to use domain depots
# Category:    basic classes
# Classes:     Domain, DomainDepot
# Uses:        
# Referenced:  DomainDepot.htm

# xtest: RANDOM

import orange

de = orange.DomainDepot()

names = ['mS#name', 'C#age', 'D#gender', 'D#race', 'cC#total', 'mS#SSN']
domain, status, metaStatus = de.prepareDomain(names)
print status

names = ['mS#SSN', 'mS#name', 'C#age', 'D#gender', 'D#race', 'cC#total']
domain, status, metaStatus = de.prepareDomain(names)
print status

names = ['mS#SSN', 'D#gender', 'C#race', 'cC#total']
domain, status, metaStatus = de.prepareDomain(names)
print status

names = ['mS#SSN', 'C#race', 'D#gender', 'cC#total']
domain, status, metaStatus = de.prepareDomain(names)
print status

names = ['mS#name', 'C#age', 'D#gender', 'D#race', 'cC#total']
domain, metaIDs, isNew = de.prepareDomain(names)
domain, status, metaStatus = de.prepareDomain(names)
print status

names = ['mS#SSN', 'mS#name', 'D#race', 'C#age', 'D#gender', 'cC#total']
domain, status, metaStatus = de.prepareDomain(names)
print status

names = ['mS#SSN', 'mS#name', 'C#age', 'C#gender', 'D#race', 'cC#total']
domain, status, metaStatus = de.prepareDomain(names)
print status

