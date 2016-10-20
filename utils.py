from random import *

"""
List functions
"""
def uniques(li):
    return list(set(li))

#:[[a]] -> [a]
#http://stackoverflow.com/questions/716477/join-list-of-lists-in-python
def concat(lli):
    return list(itertools.chain.from_iterable(lli))

def unzip(li):
    return ([x for (x,y) in li], [y for (x,y) in li])

#:[a] -> ([a],[a])
def split_eo(li):
    li1 = [li[i] for i in np.arange(0,len(li),2)]
    li2 = [li[i] for i in np.arange(1,len(li),2)]
    return li1, li2

#:[a] -> [[a]], segments of length length
def breakUpIntoSegments(li, length):
    li2 = []
    for i in range(len(li)-length+1):
        li2.append(li[i:i+length])
    return li2

def sample_without_replacement(n, k):
    li = []
    i = 0
    for i in range(k):
        r = randrange(0,n)
        while r in li:
            r = randrange(0,n)
        li.append(r)
    return li

"""
Loops
"""
def nested_for1(curli, li, f):
    if len(li)==0:
        return f(curli)
    result = []
    for i in li[0]:
        curli2 = list(curli)
        curli2.append(i)
        result.append(nested_for1(curli2, li[1:], f))
    return result
        
def nested_for(li, f):
    return nested_for1([], li, f)

"""
def nested_for1_(curli, li, f):
    if len(li)==0:
        return f(curli)
    result = []
    for i in li[0]:
        curli2 = list(curli)
        curli2.append(i)
        
def nested_for_(li, f):
    return nested_for1_([], li, f)
"""

#Example
#nested_for([range(3), range(1,5)], lambda li: [li[0],li[0]**li[1]])
"""
[[[0, 0], [0, 0], [0, 0], [0, 0]],
 [[1, 1], [1, 1], [1, 1], [1, 1]],
 [[2, 2], [2, 4], [2, 8], [2, 16]]]
 """

def list_prod(llis):
    if llis==[]:
        return []
    return [[llis[0]].append(li) for li in list_prod(llis[1:])]

# like nested_for, but flattened.
def fors(llis, f):
    return [f(x) for x in list_prod(llis)]

def fors_(llis, f):
    for x in list_prod(llis):
        f(x)

def fors_zip(llis, f):
    return fors(llis, lambda x: (x, f(x)))

"""
Print functions
"""
def printv(s, v, lim):
    if v >= lim:
        print(s)

