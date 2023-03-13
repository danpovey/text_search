#!/usr/bin/env python3

import unittest

import numpy as np

from textsearch import SourcedText
from textsearch import TextSource
from textsearch import Transcript
from textsearch import append_texts
from textsearch import texts_to_sourced_texts


class TestSourcedText(unittest.TestCase):
    def test_from_text_source_utf8(self):
        name = "test_text_source_utf8"
        s = "zażółć gęślą jaźń\n你好Hallo"
        text_source = TextSource.from_str(name=name, s=s, use_utf8=True)

        sourced_text = texts_to_sourced_texts([text_source])
        assert len(sourced_text) == 1, len(sourced_text)
        assert sourced_text[0].pos.dtype == np.uint32, sourced_text[0].pos.dtype
        np.testing.assert_equal(
            sourced_text[0].pos, np.arange(text_source.binary_text.size)
        )
        assert sourced_text[0].doc == 0, sourced_text[0].doc
        assert len(sourced_text[0].sources) == 1, len(sourced_text[0].sources)
        assert sourced_text[0].sources[0] is text_source

        # TODO(fangjun): Test doc_splits. Compute doc_splits on the fly

    def test_from_text_source_unicode_code_point(self):
        name = "test_text_source_unicode_code_point"
        s = "zażółć gęślą jaźń\n你好Hallo"
        text_source = TextSource.from_str(name=name, s=s, use_utf8=False)

        sourced_text = texts_to_sourced_texts([text_source])
        assert len(sourced_text) == 1, len(sourced_text)
        assert sourced_text[0].pos.dtype == np.uint32, sourced_text[0].pos.dtype
        np.testing.assert_equal(
            sourced_text[0].pos, np.arange(text_source.binary_text.size)
        )
        assert sourced_text[0].doc == 0, sourced_text[0].doc
        assert len(sourced_text[0].sources) == 1, len(sourced_text[0].sources)
        assert sourced_text[0].sources[0] is text_source
        # TODO(fangjun): Test doc_splits. Compute doc_splits on the fly

    def test_from_transcript_utf8(self):
        name = "test_from_transcript_utf8"
        text = "▁I ▁saw ▁a ▁girl ▁with ▁a ▁ te le s c o pe".split()
        begin_times = []
        for i in range(len(text)):
            begin_times.append(i * 0.25)
        d = {
            "text": text,
            "begin_times": begin_times,
        }

        transcript = Transcript.from_dict(name=name, d=d, use_utf8=True)

        sourced_text = texts_to_sourced_texts([transcript])
        assert len(sourced_text) == 1, len(sourced_text)
        assert sourced_text[0].pos.dtype == np.uint32, sourced_text[0].pos.dtype
        np.testing.assert_equal(
            sourced_text[0].pos, np.arange(transcript.binary_text.size)
        )
        assert sourced_text[0].doc == 0, sourced_text[0].doc
        assert len(sourced_text[0].sources) == 1, len(sourced_text[0].sources)
        assert sourced_text[0].sources[0] is transcript

    def test_from_transcript_unicode_code_point(self):
        name = "test_from_transcript_unicode_code_point"
        text = "▁I ▁saw ▁a ▁girl ▁with ▁a ▁ te le s c o pe".split()
        begin_times = []
        for i in range(len(text)):
            begin_times.append(i * 0.25)
        d = {
            "text": text,
            "begin_times": begin_times,
        }

        transcript = Transcript.from_dict(name=name, d=d, use_utf8=False)

        sourced_text = texts_to_sourced_texts([transcript])
        assert len(sourced_text) == 1, len(sourced_text)
        assert sourced_text[0].pos.dtype == np.uint32, sourced_text[0].pos.dtype
        np.testing.assert_equal(
            sourced_text[0].pos, np.arange(transcript.binary_text.size)
        )
        assert sourced_text[0].doc == 0, sourced_text[0].doc
        assert len(sourced_text[0].sources) == 1, len(sourced_text[0].sources)
        assert sourced_text[0].sources[0] is transcript

    def test_append_texts_text_source_utf8(self):
        name0 = "test_append_texts_text_source_utf8_0"
        s0 = "zażółć gęślą jaźń\n"
        text_source0 = TextSource.from_str(name=name0, s=s0, use_utf8=True)

        name1 = "test_append_texts_text_source_utf8_1"
        s1 = "你好Hallo"
        text_source1 = TextSource.from_str(name=name1, s=s1, use_utf8=True)

        sourced_text0 = texts_to_sourced_texts([text_source0])
        sourced_text1 = texts_to_sourced_texts([text_source1])
        sourced_text = append_texts([sourced_text0[0], sourced_text1[0]])
        print(sourced_text)


if __name__ == "__main__":
    unittest.main()
