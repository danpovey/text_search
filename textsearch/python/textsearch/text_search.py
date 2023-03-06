# Copyright      2023   Xiaomi Corp.       (author: Wei Kang)
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Union
from .suffix_array import create_suffix_array


@dataclass
class TextSource:
    name: str  # might be a filename
    text: np.ndarray  # the text, probably as a sequence of bytes but could be utf-32 i.e. np.uint32.
    # Length of the array may not be >= 2**32.


@dataclass
class SourcedText:
    text: np.ndarray  # the text, a 1-d array, probably a sequence of bytes but could be uint32 representing utf-32.
    # Note: this text array is 'owned' here so it can be modified by the user.
    pos: np.ndarray  # the text position in the original source for each element of the text.  Of type int64;
    doc: Union[int, np.ndarray]  # the document index for each element of the text.
    # the np.ndarray would be of type int64.
    sources: List[
        TextSource
    ]  # for reference, the list of available text sources that this text might come from.

    @staticmethod
    def from_texts(sources: List[TextSource]) -> "SourcedText":
        text_list = []
        pos_list = []
        doc_list = []
        for i in range(len(sources)):
            text_list.append(sources[i].text)
            pos_list.append(np.arange(0, len(sources[i].text), dtype=np.int64))
            doc_list.append(np.full(len(sources[i].text), i, dtype=np.int64))
        return SourcedText(
            text=np.concatenate(text_list, axis=0),
            pos=np.concatenate(pos_list, axis=0),
            doc=np.concatenate(doc_list, axis=0),
            sources=sources,
        )

    def create_suffix_array(self) -> np.ndarray:
        extend_text = np.concatenate(
            (
                self.text,
                np.array(
                    [np.iinfo(self.text.dtype).max - 1, 0, 0, 0], dtype=self.text.dtype
                ),
            ),
            axis=0,
        )
        return create_suffix_array(extend_text)


def find_candidate_matches(
    close_matches: np.ndarray,
    text: SourcedText,
    length_ratio: float = 2.0,
    num_candidates: int = 5,
) -> List[List[Tuple[int, int]]]:
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
    assert close_matches.ndim == 1, close_matches.ndim
    assert close_matches.size % 2 == 0, close_matches.size
    tot_query_symbols = close_matches.size // 2
    assert text.text.size > tot_query_symbols, (text.text.size, tot_query_symbols)

    # can we assert the query docs are sorted by doc ids ?
    num_query_docs = np.unique(text.doc[0:tot_query_symbols]).size
    assert num_query_docs == text.doc[tot_query_symbols]

    row_splits = np.zeros(num_query_docs + 1, dtype=np.int64)
    for i in range(1, tot_query_symbols + 1):
        if text.doc[i] != text.doc[i - 1]:
            row_splits[text.doc[i]] = i

    candidate_matches = []

    for q in range(num_query_docs):
        matches_start_pos = 2 * row_splits[q]
        matches_end_pos = 2 * row_splits[q + 1]
        current_query_len = row_splits[q + 1] - row_splits[q]
        reference_chunk_length = current_query_len * length_ratio

        current_close_matches = np.sort(
            close_matches[matches_start_pos:matches_end_pos]
        )

        # (start pos in reference, end pos in reference, hits)
        current_candidates = [(0, 0, 0)] * num_candidates
        for i in range(current_close_matches.size):
            j = i + 1
            doc_id = text.doc[current_close_matches[i]]
            pos_id = text.pos[current_close_matches[i]]
            while (
                j < current_close_matches.size
                and text.doc[current_close_matches[j]] == doc_id
                and text.pos[current_close_matches[j]]
                < (pos_id + reference_chunk_length)
            ):
                j += 1

            candidate = (current_close_matches[i], current_close_matches[j - 1] + 1, j - i)
            current_candidates.append(candidate)

            current_candidates = sorted(
                current_candidates, key=lambda s: s[2], reverse=True
            )
            delete_id = num_candidates
            for k in range(1, num_candidates):
                # handle the overlapping
                if (
                    current_candidates[k][0] >= current_candidates[k - 1][0]
                    and current_candidates[k][1] <= current_candidates[k - 1][1]
                ):
                    delete_id = k
                    break
            del current_candidates[delete_id]
        candidate_matches.append([(c[0], c[1]) for c in current_candidates])
    return candidate_matches

