

# Goals

  Plan for text searching.  I propose to create a standalone Python package for this that could have
  multiple usage.  e.g. we can call the package fasttextsearch; it would have a C++ extension, we could
  perhaps use lilcom as the starting point in terms of package structure (unless someone has another idea);
  I am thinking we could make numpy a dependency, and lilcom already handles viewing Numpy arrays
  as C arrays.

  The immediate problem this package would solve is how to locate the reference text for a piece of
  a file that we decoded, when we have the reference text as a long piece of text with no correspondence.

  The core part of the package - the part in C - would be some code for creating and manipulating
  suffix arrays.  (For flexibility, at the "C" level we should use templates to support int8/char, int16 and int32 for
  the symbol type, and int32 and int64 for the index type.)

  The most technical part, for creating the suffix array (written in C) would look like this from
  its Python interface (be careful, it is a little quirky to save a copy).
  There is some code in k2 (see nbest.h) that we can just copy for the internals of this, which has already been tested.
  In fact we don't really need it in k2 so we could remove it from there at some point.
  (this project has other potential users from k2, and I want to separate it as k2 is always going to
  be harder to install).


```python

   def create_suffix_array(input: np.ndarray) -> np.ndarray:
      """
      Creates a suffix array from the input text and returns it as a NumPy array.  Read
      the usage carefully as it has some special requirements that will require careful data
      preparation.

      Args:
         input: an np.ndarray that must be one types uint8, uint16 or uint32.  Its shape
            should be (seq_len + 3,) where `seq_len` is the text sequence length INCLUDING
            EOS SYMBOL.
            The EOS (end of sequence) symbol must be the largest element of the
            type (i.e. of the form 2^n - 1), must be located at input[seq_len - 1] and
            must appear nowhere else in `input` (you may have to map the input
            symbols somehow to achieve this).  It must be followed by 3 zeros, for reasons
            related to how the algorithm works.
      Returns:
            Returns a suffix array of type np.uint64,
            of shape (seq_len,).  This will consist of some permutation of the elements
            0 .. seq_len - 1.
      """
```

```python
@dataclass
class TextSource:
      name: str  # might be a filename
      text: np.ndarray  # the text, probably as a sequence of bytes but could be utf-32 i.e. np.uint32.
                        # Length of the array may not be >= 2**32.


# we'll have a global list of text sources during the program lifetime, and indexes into this list
# (probably int32) will represent the text source.
TextSources = List[TextSource]
```

```python
@dataclass
class SourcedText:
      # represents a text with some meta-info that records from where in a collection of
      # texts it came.
      text: np.ndarray  # the text, a 1-d array, probably a sequence of bytes but could be uint32 representing utf-32.
                        # Note: this text array is 'owned' here so it can be modified by the user.
      pos: np.ndarray  # the text position in the original source for each element of the text.  Of type uint32;
      doc: Union[int, np.ndarray]   # the document index for each element of the text.
                                    # the np.ndarray would be of type uint32.
      sources: TextSource  # for reference, the list of available text sources that this text might come from.
```

```python
def texts_to_sourced_texts(sources: List[TextSource]) -> List[SourcedText]:
    pass
```

```python
def append_texts(texts: List[SourcedText]) -> SourcedText:
    pass
```

```python
def remove(t: SourcedText, keep: np.ndarray) -> SourcedText:
    """
    Removes some positions from a SourcedText (out-of-place).
    Args:
        t: the text to remove some positions of
        keep: an np.ndarray with dtype == np.bool, that is True
          for positions that should be kept; must have the same shape
          as t.text.
    Returns:
        A SourcedText with some positions removed.
```

for things like mapping bytes to e.g. lower-case or finding the positions to remove()
because of, for example, repeated spaces or punctuation, the user can do this manually
using normal Python and numpy operations.  This might get a little more complicated for
languages that use lots of non-ASCII code points; should be doable though.



  The following part should also probably be written in C:
```python
    def find_close_matches(suffix_array: np.ndarray, query_len: int) -> np.ndarray:
       """
       Assuming the suffix array was created from a text where the first `query_len`
       positions represented the query text and the remaining positions represent
       the reference text, return a list indicating, for each suffix position in the query
       text, the two suffix positions in the reference text that immediately precede and
       follow it lexicographically.  (I think suffix position refers to the last character
       of a suffix).     This is easy to do from the suffix array without computing,
       for example, the LCP array; and it produces exactly 2 matches per position in the
       query text, which is also convenient.

       (Note: the query and reference texts could each represent multiple separate
       sequences, but that is handled by other code; class SourcedText keeps track of that
       information.)

       Args:
        suffix_array: A suffix array as created by create_suffix_array(), of dtype
           np.uint32 or np.uint64 and shape (seq_len,).

         query_len: A number 0 <= query_len < seq_len, indicating the length in symbols
          (likely bytes) of the query part of the text that was used to create `suffix_array`.

       Returns an np.ndarray of shape (query_len * 2,), of the same dtype as suffix_array,
         in which positions 2*i and 2*i + 1 represent the two positions in the original
         text that are within the reference portion, and which immediately follow and
         precede, in the suffix array, query position i.  This means that the
         suffixes ending at those positions are reverse-lexicographically close
         to the suffix ending at position i.  As a special case, if one of these
         returned numbers would equal the EOS position (position seq_len - 1), or
         if a query position is before any reference position in the suffix aray, we
         output seq_len - 2 instead to avoid having to handle special cases later on
         (anyway, these would not represent a close match).
       """
```

This can be accomplished in a linear pass over the suffix array, e.g. (ignoring edge cases):

```c++
          last_ref_pos = -1;
          for (int i = 0; i < seq_len; i++) {
             text_pos = suffix_array[i];
             if (text_pos > query_len) {  // reference position.
                for (int j = last_ref_pos + 1; j < i; j++) {
                   int cur_ref_pos = text_pos;
                   query_pos = suffix_array[j + 1];
                   output[2 * query_pos] = last_ref_pos;
                   output[2 * query_pos + 1] = cur_ref_pos;
                   last_ref_pos = cur_ref_pos;
                }
             }
          }

```

```python
     def find_candidate_matches(close_matches: np.ndarray,
                                text: SourcedText,
                                length_ratio: float = 2.0,
                                num_candidates: int = 5) -> List[List[Tuple[int, int]]]:
       """
       Find candidate regions in reference document that could be good matches for
       each query document.

       Args:
          close_matches: an np.ndarray of shape (2*tot_query_symbols,) as returned by
             find_close_matches(), indicating two close matches within the reference
             text for each symbol in the query documents.
          text:  The SourcedText corresponding to the query and reference documents
             combined; needed for its `doc` member which can be assumed to be an np.ndarray.
          length_ratio: indicates the maximum candidate-region length, which will be reduced
             if the matching reference document was shorter than this.
           num_candidates:  the number of candidate regions to find for each query
             document.
       Returns:
          A list, one per query document, of close matches, where each close match
          is a (begin, end) position within `text`.
       """
```

  [TODO: finish this. uses the output of find_close_matches to locate candidate matches for
  Levenshtein or Smith-Waterman alignment.  This is done separately for each query text.

  We process pieces of `close_matches` separately for each query document (the number of query
  documents can be found from the size of close_matches and from text.doc).

  It will be helpful to go through the elements of
  'text.doc' and accumulate something equivalent to k2's "row_splits" (maybe just
  for the query documents); this can be used
  for a few things in this function, such as iterating over the query documents

  We are looking for the up-to-`num_candidates` pieces of reference text of length at most
  (this_query_len * length_ratio) that have the largest number of "hits" in the output of
  find_close_matches, subject to a "non-overlapping" constraint.  This may not be formally
  very well-defined when we consider the "non-overlapping" constraint but approximate
  solutions are OK.

  The basic algorithm would be a linear-time thing where, for each query
  document, we go through the sorted close-matches, always keeping track of the next element
  of the sorted close-matches that is no more than

```c++
   for (int q = 0; q < num_query_docs; ++q) {
      matches_start_pos = 2 * row_splits[q];
      matches_end_pos = 2 * row_splits[q+1];


      int j = matches_start_pos;
      for (i = matches_start_pos;  i < matches_end_pos; ++i) {
       // find the current candidate start-position in the reference part of the full text (this is
       // just a couple of array lookups where we find the reference document index
       // and the position 'pos' within that reference document.

       // Then we keep advancing j as far as we can while ensuring that the
       // position within that reference document is not more than `reference_chunk_length`
       // greater than the position for i and the document-index is not different from that of i.

       // j - i (i.e. the number of matches ) is the metric we are trying to maximize.
       // basically we will keep up to a specified number of 'best' matches, e.g. 5, with the limitation
       // that they cannot overlap (if they overlap, we take the better one).  no need for fancy
       // algorithms here, we can just assume num_candidates is small and use iteration over
       // that number.
    }
```
 If the core part of this is in C++, it might be easiest to format the output as an ndarray conceptually
 of shape (num_query_docs, num_candidates, 2) which can probably be just treated as a uint64_t*
 within the C++ code.




## Next: stuff for Levenshtein or Smith-Waterman alignment.

  Once we have the candidate matches:
     We extend them slightly to the left and right, e.g. by 10 + query_length // 8, this can be done
     at a fairly outer level of the code.

     We match each of them with the query text using a Levenshtein-like alignment process that
     allows un-penalized insertions of reference text at the very beginning, or very end, of
     the query text.   (easy to do: just initialize with zeros along one side of the array, and
     once all scores are computed, take minimum from other side of the array).
     This will return the alignment and the corresponding score.
     The score can be used to find the best-matching region, and the alignment (probably
     pairs of: ref-position, query-position) can be used to find the beginning and end
     position in the reference text.  (I think we can just convert the alignments from
     np.ndarray to python objects and use Python code to find the beginning/end positions,
     since this part is linear time).


 The output that we'll be writing to disk would then be, for each query text:
    (query_length, num_errs, reference_name, reference_first_byte, reference_last_byte).

 To find the reference_first_byte and reference_last_byte, we will use the "pos" elements
 in the SourcedText objects.  This is why we added the "pos" elements, because we need
 to backtrack through the normalization process, which would otherwise be nontrivial.



# A note on the larger plan:

  You might be worried about text-normalization issues, and how we will align things like digits
  that may look very diffrent between normalized and un-normalized text, i.e. between our automatic
  transcript and the reference text.
  What I am planning is that most of the time,  the query texts will be fairly long, like 30 seconds
  or a minute, and even if there are lots of digits internally, if there are words at the start and end
  we should still find the right match in the reference text.  Then we can later use algorithms
  involving the acoustics to do further cleanup of the reference text.

