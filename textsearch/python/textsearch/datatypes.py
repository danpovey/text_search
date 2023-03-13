import json
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


@dataclass
class TextSource:
    """
    Represents the full text of a utf-8 document.

    The user can choose whether to save it in a utf-8 encoded np.uint8 array
    or to save its Unicode codepoint into a np.int32 array.
    """

    # filename or ID
    name: str

    # It can be either a np.uint8 array or a np.int32 array.
    # If it is a np.uint8 array, it contains the utf-8 encoded bytes;
    # If it is a np.int32 array, it contains the Unicode codepoint of the text.
    # Length of the array may not be >= 2**32.
    binary_text: np.ndarray

    # Only used when binary_text.dtype is np.int32.
    # It contains the mapping from utf-8 character position to byte position.
    # That is pos[i] contains the byte position in binary_text for the i-th
    # utf-8 character
    pos: Optional[np.ndarray] = None

    @property
    def text(self) -> str:
        """Return Python string representation of self.binary_text decoded as UTF-8."""
        if self.binary_text.dtype == np.uint8:
            return self.binary_text.tobytes().decode("utf-8")
        else:
            assert self.binary_text.dtype == np.int32, self.binary_text.dtype
            return "".join([chr(i) for i in self.binary_text])

    @staticmethod
    def from_str(name: str, s: str, use_utf8: bool) -> "TextSource":
        """Construct an instance of TextSource from a string.

        Args:
          name:
            Name of the returned instance. It can be either a filename or an ID.
          s:
            It contains the text string.
          use_utf8:
            True to encode the text with utf-8.
            False to save the Unicode codepoint of the text.
        """
        if use_utf8:
            binary_text = s.encode("utf-8")
            return TextSource(
                name=name,
                binary_text=np.frombuffer(binary_text, dtype=np.uint8),
                pos=None,
            )
        else:
            binary_text = np.fromiter((ord(i) for i in s), dtype=np.int32, count=len(s))
            pos = _find_byte_offsets_for_utf8_symbols(binary_text)

            return TextSource(
                name=name,
                binary_text=binary_text,
                pos=pos,
            )


def _find_byte_offsets_for_utf8_symbols(binary_text: np.ndarray) -> np.ndarray:
    """
    Args:
      binary_text:
        A 1-D int32 array containing Unicode codepoint of characters.
    Returns:
      Return a np.uint32 1-D array. Its length equals to the number of
      characters contained in binary_text before encoding. ans[i]
      contains the byte position in binary_text for the i-th character.
    """
    byte_offsets = []

    # Iterate over each byte in the binary text
    byte_index = 0
    for i in binary_text:
        if i < 0x80:
            # Single-byte UTF-8 symbol
            symbol_length = 1
        elif i < 0x800:
            # Two-byte UTF-8 symbol
            symbol_length = 2
        elif i < 0x10000:
            # Three-byte UTF-8 symbol
            symbol_length = 3
        else:
            # Four-byte UTF-8 symbol
            symbol_length = 4

        # Add the byte index to the list of offsets
        byte_offsets.append(byte_index)

        # Move the byte index to the start of the next symbol
        byte_index += symbol_length

    # Note: uint32 should be sufficient for 4GB of text,
    # I don't expect we'd deal with such large documents here.
    return np.asarray(byte_offsets, dtype=np.uint32)


@dataclass
class Transcript:
    """
    Type that represents automatically transcribed text.
    The time marks are at the byte level.  We can easily turn automatic
    transcripts of text into this type as long as we keep track of the time for each symbol.
    """

    # A filename or an ID.
    name: str

    # the text as a sequence of bytes or (more likely) int32 corresponding to UTF codepoints.
    # We will have to expand BPE tokens into bytes, presumably converting _ into space.
    #
    # When its dtype is np.uint8, it is encoded with utf-8
    # When its dtype is np.int32, it contains the Unicode code point of characters
    binary_text: np.ndarray

    # Gives the time in seconds for each byte of the text.  (Should be in non-decreasing order).
    # Its dtype is np.float32
    times: np.ndarray

    @property
    def text(self) -> str:
        """Return Python string representation of self.binary_text decoded as UTF-8."""
        return self.binary_text.tobytes().decode("utf-8")

    @staticmethod
    def from_dict(name: str, d: dict, use_utf8: bool) -> "Transcript":
        """
        Args:
          name:
            A filename or an ID.
          d:
            A dict containing:

              - d["text"]: List[str]. Each element in the list is a token or a word.

              - d["begin_times]: List[float]. len(d["text"]) == d["begin_times"].
                d["begin_times"][i] is the begin time for d["text"][i]
          use_utf8:
            True to use utf-8 to encode the text.
            False to save the Unicode code point of the text.
        """
        assert "text" in d, list(d.keys())
        assert "begin_times" in d, list(d.keys())
        assert len(d["text"]) == len(d["begin_times"]), (
            len(d["text"]),
            len(d["begin_times"]),
            d["text"],
            d["begin_times"],
        )

        if use_utf8:
            byte_list = []
            time_list = []
            for text, begin_time in zip(d["text"], d["begin_times"]):
                b = text.encode("utf-8")

                if time_list:
                    # Check that begin_time is non-decreasing.
                    #
                    # < here requires that it is strictly increasing.
                    assert time_list[-1] < begin_time, (time_list[-1], begin_time)

                # bytes belonging to the same text have the same begin time
                time_list += [begin_time] * len(b)
                byte_list.append(b)

            return Transcript(
                name=name,
                binary_text=np.frombuffer(b"".join(byte_list), dtype=np.uint8),
                times=np.asarray(time_list, dtype=np.float32),
            )
        else:
            codepoint_list = []
            time_list = []
            for text, begin_time in zip(d["text"], d["begin_times"]):
                codepoint_list.append(ord(i) for i in text)

                if time_list:
                    # Check that begin_time is non-decreasing.
                    #
                    # < here requires that it is strictly increasing.
                    assert time_list[-1] < begin_time, (time_list[-1], begin_time)

                # bytes belonging to the same text have the same begin time
                # Each character occupies 4 bytes, so it is multiplied by 4
                time_list += [begin_time] * (len(text) * 4)

            return Transcript(
                name=name,
                binary_text=np.fromiter(chain(*codepoint_list), dtype=np.int32),
                times=np.asarray(time_list, dtype=np.float32),
            )


# we'll have a global list of text sources during the program lifetime, and indexes into this list
# (probably int32) will represent the text source.
TextSources = List[Union[TextSource, Transcript]]


@dataclass
class SourcedText:
    """
    Represents a text with some meta-info that records from where in a collection of
    texts it came.
    """

    # A 1-d array. Its data type can be either np.uint8 or np.int32.
    # If it is np.uint8, it represents utf-8 encoded strings.
    # If it is np.int32, it represents strings with Unicode code point.
    #
    # Note: this text array is 'owned' here so it can be modified by the user.
    binary_text: np.ndarray

    # The text position in the original source for each element of the text. Of type uint32;
    pos: np.ndarray

    # The document index for each element of the text.
    # The np.ndarray would be of type uint32.
    doc: Union[int, np.ndarray]

    # For reference, the list of available text sources that this text might come from.
    sources: TextSources

    # Optional: an array derived from `doc` which is like the row
    # splits in k2 (or in tensorflow ragged arrays), with `doc` being
    # the row-ids.
    doc_splits: Optional[np.ndarray] = None


def texts_to_sourced_texts(sources: TextSources) -> List[SourcedText]:
    """Construct a list of SourcedText from TextSources.
    We have len(ans) == len(sources).
    """
    ans = []

    for s in sources:
        # pos[i] contains the index of the i-th entry in s.binary_text
        #
        # If s.binary_text.dtype is np.int32, pos[i] is the position of the i-th
        # Unicode code point in s.binary_text
        #
        # If s.binary_text.dtype is np.uint8, pos[i] is the position of the
        # i-th byte of s.binary_text
        pos = np.arange(s.binary_text.size, dtype=np.uint32)

        ans.append(
            SourcedText(
                binary_text=s.binary_text,
                pos=pos,
                doc=0,  # indexes into the following `sources` attribte
                sources=[s],
                doc_splits=None,
            )
        )

    return ans


def append_texts(texts: List[SourcedText]) -> SourcedText:
    if len(texts) == 1:
        # return a shallow copy
        return texts[0]

    # Check that all texts have the same dtype
    for t in texts[1:]:
        assert t.binary_text.dtype in (np.uint8, np.int32), t.dtype
        assert texts[0].binary_text.dtype == t.binary_text.dtype, (
            texts[0].binary_text.dtype,
            t.binary_text.dtype,
        )

    binary_text = np.concatenate([t.binary_text for t in texts])
    pos = np.concatenate([t.pos for t in texts])

    doc_list = []
    for t in texts:
        if isinstance(t.doc, np.ndarray):
            assert t.doc.dtype == np.uint32, t.doc.dtype
            doc_list.append(t.doc)
        else:
            assert isinstance(t.doc, int), type(t.doc)
            doc_list.append(np.zeros(t.binary_text.size, dtype=np.uint32))

    num_docs = 0
    for i, d in enumerate(texts):
        if i == 0:
            num_docs = len(d.sources)
            continue
        doc_list[i] += num_docs
        num_docs += len(d.sources)

    doc = np.concatenate(doc_list)
    assert doc.dtype == np.uint32, doc.dtype

    sources = [s for t in texts for s in t.sources]

    # TODO(fangjun: Add row_ids_to_row_splits() to compute doc_splits from doc

    return SourcedText(
        binary_text=binary_text,
        pos=pos,
        doc=doc,
        sources=sources,
        doc_splits=None,
    )


'''
TODO
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
    """
    pass

'''
