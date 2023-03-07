'''
  MIT License

  Copyright (c) 2022 Alexandre Felipe

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
'''

import numpy as np

def seq_levenstein(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
  '''
    Array of Levenshtein distances ending at given locations

    Internally it uses a skewed cost metric, such that
    final_cost[j] = cost[j] + j, this makes it possible to use
    a prefix minimum to compare 

    Parameters
    ----------
    query : array_like
        The array of symbols to be searched
    ref : array_like
        A long array of symbols where the query will be searched

    Returns
    -------
    final_cost : nd_array
        An array indicating the minimum cost of an alignment that ends
        at a given each position of the ref array

    Examples
    --------

    >>> seq_levenshtein([0, 1, 0]), [1, 1, 1, 0, 1, 1])
    array([3, 2, 2, 2, 1, 1, 1])

    >>> seq_levenshtein([4,5,5], [1,2,3,4,5,6,7,8])
    array([3, 3, 3, 3, 2, 1, 1, 2, 3])
  '''
  
  query = np.asarray(query)
  ref = np.asarray(ref)
  j = np.arange(len(ref) + 1, dtype=np.int32)
  cost = -j
  subst = np.zeros_like(cost);
  aux = np.zeros_like(cost);
  ins = np.zeros_like(cost);
  subst[0] = len(ref)
  for i in range(len(query)):
      # take from query
      delete = cost + 1;
      # easy: substitutions or reuses
      subst[1:] = cost[:-1] - (ref == query[i]);
      # tricky take from ref
      cost = np.minimum.accumulate(
          np.minimum(cost + 1, subst)
      );
  return cost + j;


def seq_levenshtein_range(query: np.ndarray, ref: np.ndarray, cost_dtype: np.dtype=np.int32) -> np.ndarray:
  '''
    Computes the Levenshtein distance using numpy vectorized operations
    
    In order to keep trak of the initial positions the mapping
    `f(c, i) -> M*c + i`, with inverse `(c, i) = divmod(f, M)`
    is applied, where `0 <= c <= M = len(query)` is the cost and 
    `0 <= i <= len(ref)` is the initial position of the aligned segment.
    So that `f(c, i)` and `c` have the same order, i.e.
      `f(c1, i1) < f(c2, i2)` iff `c1 < c2`
    so the augmented cost computation computation remains the same
    as the cost computation.
    
    In the end the cost and initial indices can be separated using divmod
    
    Internally it uses a skewed cost metric, such that
    final_cost[j] = cost[j] + j, this makes it possible to use
    a prefix minimum to compare.
    
    The data type to be used for the augmented cost computation
    is `int32` or `int64` depending on the size of the reference
    and the query
    
    Parameters
    ----------
    query : array_like
        The array of symbols to be searched
    ref : array_like
        A long array of symbols where the query will be searched
    Returns
    -------
    final_cost : ndarray
        An array indicating the minimum cost of an alignment that ends
        at each position of the ref array
    initial_index: ndarray
        An array indicating the starting position of the alignment that
        ends at each position of the ref array.
  '''
  L = (len(ref) + 2) * len(query)
  cost_dtype = np.int32 if L < 2**30 else np.int64

  M = np.array(len(ref) + 2, dtype=cost_dtype)
  query = np.asarray(query)
  ref = np.asarray(ref)
  j = np.arange(len(ref) + 1, dtype=cost_dtype)
  cost = -j * (M - 1)
  subst = np.zeros_like(cost);
  aux = np.zeros_like(cost);
  ins = np.zeros_like(cost);
  subst[0] = M*M
  for i in range(len(query)):
      # take from query
      delete = cost + M;
      # easy: substitutions or reuses
      subst[1:] = cost[:-1] - (ref == query[i])*M;
      # tricky take from ref
      cost = np.minimum.accumulate(
          np.minimum(cost + M, subst)
      );
  return divmod(cost + j*M, M);


def tokenize_ascii_char(s):
  return np.frombuffer(s.encode('ascii'), dtype=np.uint8)


def score_error_per_ref_char(c, i):
  j = np.arange(len(c))
  return c.astype('float') / np.maximum(1, j - i)

def score_error(c, i):
  return c

def levenshtein_alignments(query, ref, 
    num_candidates=1, 
    threshold=0.5, 
    overlapped=False, 
    score=score_error_per_ref_char, 
    tokenize=tokenize_ascii_char
  ):
  '''
    Get the N best alignments and costs accordingly to the Levenshtein distance metric

    Parameters
    ----------
        query:
            String to be searched
        ref:
            String where the query will be searched
        num_candidates:
            Number of candidates to be returned
        threshold:
            Maximum symbol error rate to be accepted in the candidate
            list.
        overlapped:
            Returns segments that overlap
        score:
            How to compare two alignments based on the number of errors
            and the starting point
        tokenize:
            A function that converts the string to a numpy array of symbols
    
    Returns:
        c : An array such that `c[k]` is the levenshtein distance between the
             query and the segment covering tokens ref_tokens[i[k]:r[k]]
        i : initial position of the segment tokens
        r : final positions of the segment tokens
  '''
  t_query = tokenize(query)
  t_ref = tokenize(ref)

  c, i = seq_levenshtein_range(t_query, t_ref)
  r = np.argsort(score(c, i))[:num_candidates]

  slices = []
  for rk in r:
    # there is room for improvement for large queries
    if not overlapped or all(c[i[rk]:rk] >= 0):
      if c[rk] > threshold * len(query):
        break;
      slices.append([c[rk], i[rk], rk])
      if len(slices) == num_candidates:
        break
      if not overlapped:
        c[i[rk]:rk] = -1
  return slices
