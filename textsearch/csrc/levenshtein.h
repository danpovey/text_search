/**
 * Copyright      2023     Xiaomi Corporation (authors: Wei Kang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TEXTSEARCH_CSRC_LEVENSHTEIN_H_
#define TEXTSEARCH_CSRC_LEVENSHTEIN_H_

#include <algorithm>
#include <bitset>
#include <cassert>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fasttextsearch {

// See docs in Backtrace below, it is a segment of backtrace containing 64 bits
// of path trace.
struct DynamicBacktrace {
  int64_t bitmap;
  std::shared_ptr<DynamicBacktrace> prev;

  DynamicBacktrace(int64_t bitmap, std::shared_ptr<DynamicBacktrace> prev)
      : bitmap(bitmap), prev(prev) {}

  explicit DynamicBacktrace(int64_t bitmap) : bitmap(bitmap) {}

  DynamicBacktrace &operator=(const DynamicBacktrace &src) = default;
  DynamicBacktrace(const DynamicBacktrace &src) = default;
  // Move constructor
  DynamicBacktrace(DynamicBacktrace &&src) = default;
  DynamicBacktrace &operator=(DynamicBacktrace &&src) = default;
};

struct Backtrace {
  int64_t bitmap; // bitmap records shape of most recent backtrace, where every
                  // time we consume a query symbol we do: bitmap |= (1 <<
                  // (num_bits++));   and every time we consume a reference
                  // symbol we do: bitmap  |= (0 << (num_bits++));  which
                  // can be optimized to: num_bits++;  When num_bits reaches 64
                  // we allocate a DynamicBacktrace object and reset it to 0.
  int32_t num_bits;
  std::shared_ptr<DynamicBacktrace> prev;

  Backtrace() : bitmap(0), num_bits(0), prev(nullptr){};

  Backtrace &operator=(const Backtrace &src) = default;
  Backtrace(const Backtrace &src) = default;
  // Move constructor
  Backtrace(Backtrace &&src) = default;
  Backtrace &operator=(Backtrace &&src) = default;

  /*
   * Update the Backtrace by increasing the num_bits with 1.
   *
   * param [in] update_bitmap If true, bitmap will be updated, it means we are
   *                          consuming a query symbol. See the docs of bitmap
   *                          for more details.
   */
  void Update(bool update_bitmap) {
    if (update_bitmap)
      bitmap |= 1 << num_bits;

    num_bits++;

    if (num_bits == 64) {
      if (prev == nullptr) {
        prev = std::make_shared<DynamicBacktrace>(bitmap);
      } else {
        prev = std::make_shared<DynamicBacktrace>(bitmap, prev->prev);
      }
      bitmap = 0;
      num_bits = 0;
    }
  }

  /*
   * Convert the bitmap backtrace to string (i.e. the bit string like '011001').
   *
   * Note: In the original bitmap the latest bits are on the left side (because
   * we update bitmap with `bitmap |= 1 << num_bits`), but in the return string
   * the latest bits are on the right side. For example, the original bitmap is
   * `0b110110`, the returned string will be `011011`.
   */
  std::string ToString() const {
    std::ostringstream oss;
    if (num_bits != 0) {
      auto str = std::bitset<64>(bitmap).to_string().substr(64 - num_bits);
      oss << str.substr(0, num_bits);
    }
    auto trace = prev;
    while (trace != nullptr) {
      oss << std::bitset<64>(trace->bitmap);
      trace = trace->prev;
    }
    auto str = oss.str();
    std::reverse(str.begin(), str.end());
    return str;
  }
};

/*
 * The class represents the internal state of levenshtein dp recursion
 * containing the cost up to now and the backtrace.
 */
struct LevenshteinElement {
  int32_t cost; // The cost (levenshtein distance) up to current element
  int64_t
      position; // The index into the target array indicating the position of
                // the target where this element comes from. This will be useful
                // when recovering the alignment between query and target.
  Backtrace backtrace; // The backtrace from which we can recover the alignment.

  explicit LevenshteinElement(int32_t cost) : cost(cost) {}

  LevenshteinElement(int32_t cost, Backtrace backtrace)
      : cost(cost), backtrace(backtrace) {}

  LevenshteinElement() = default;

  LevenshteinElement &operator=(const LevenshteinElement &src) = default;
  LevenshteinElement(const LevenshteinElement &src) = default;
  // Move constructor
  LevenshteinElement(LevenshteinElement &&src) = default;
  LevenshteinElement &operator=(LevenshteinElement &&src) = default;

  /*
   * Handle deletion error given current element.
   *
   * @param [in] c The deletion cost.
   *
   * @return Return a new LevenshteinElement object with cost and
   *         backtrace updated based on current element.
   */
  LevenshteinElement Delete(int32_t c) const {
    auto res = LevenshteinElement(cost + c, backtrace);
    res.backtrace.Update(false); // consuming target symbol
    return res;
  }

  /*
   * Handle insertion error given current element.
   *
   * @param [in] c The insertion cost.
   *
   * @return Return a new LevenshteinElement object with cost and
   *         backtrace updated based on current element.
   */
  LevenshteinElement Insert(int32_t c) const {
    auto res = LevenshteinElement(cost + c, backtrace);
    res.backtrace.Update(true); // consuming query symbol
    return res;
  }

  /*
   * Handle replacement error given current element.
   *
   * @param [in] c The replacement cost.
   *
   * @return Return a new LevenshteinElement object with cost and
   *         backtrace updated based on current element.
   */
  LevenshteinElement Replace(int32_t c) const {
    auto res = LevenshteinElement(cost + c, backtrace);
    // Consuming both query and target symbols
    // Caution: DO NOT change the order of following two lines, it determines
    // how we recover the alignment.
    res.backtrace.Update(false);
    res.backtrace.Update(true);
    return res;
  }

  /*
   * Handle equal given current element.
   *
   * Note: Only the backtrace will be updated.
   *
   * @return Return a new LevenshteinElement object with
   *         backtrace updated based on current element.
   */
  LevenshteinElement Equal() const {
    auto res = LevenshteinElement(cost, backtrace);
    // Consuming both query and target symbols
    // Caution: DO NOT change the order of following two lines, it determines
    // how we recover the alignment.
    res.backtrace.Update(false);
    res.backtrace.Update(true);
    return res;
  }
};

/*
 * Calculate the levenshtein distance between query and target and also return
 * the alignments (can be constructed from backtrace in LevenshteinElement).
 *
 * Note: We are doing the infix search, gaps at query end and start are not
 * penalized. What that means is that deleting elements from the start and end
 * of target is "free"! For example, if we had ACT and CGACTGAC, the levenshtein
 * distance would be 0, because removing CG from the start and GAC from the end
 * of target is "free" and does not count into total levenshtein distance.
 *
 * @param [in] query The pointer to the query sequence.
 * @param [in] query_length The length of the query sequence.
 * @param [in] target The pointer to the target sequence.
 * @param [in] target_length The length of the target sequence.
 * @param [in,out] alignments  The container that alignments info will write to.
 *                             Because we are doing infix search, so there might
 *                             be multiple matches in target sequence with the
 *                             same levenshtein distance. `alignments` will be
 *                             reallocated in this function, it's size will be
 *                             the number of matching segments.
 * @param [in] insert_cost  The cost of insertion.
 * @param [in] delete_cost  The cost of deletion.
 * @param [in] replace_cost  The cost of replacement.
 *
 * @return  Returns the levenshtein distance between query and target.
 */
template <typename T>
int32_t LevenshteinDistance(const T *query, size_t query_length,
                            const T *target, size_t target_length,
                            std::vector<LevenshteinElement> *alignments,
                            int32_t insert_cost = 1, int32_t delete_cost = 1,
                            int32_t replace_cost = 1) {

  LevenshteinElement best_score = LevenshteinElement(-1);

  assert(target_length != 0);
  if (query_length == 0) {
    return 0;
  }

  auto scores = std::vector<LevenshteinElement>(query_length + 1);

  scores[0] = LevenshteinElement(0);
  for (size_t i = 1; i <= query_length; i++) {
    scores[i] = scores[i - 1].Insert(insert_cost);
  }

  for (size_t j = 1; j <= target_length; j++) {
    LevenshteinElement prev_diag = scores[0], prev_diag_cache;

    // we are doing infix search, the cost of the beginning symbol will always
    // be 0.
    scores[0] = LevenshteinElement(0);

    for (size_t k = 1; k <= query_length; k++) {
      prev_diag_cache = scores[k];
      if (query[k - 1] == target[j - 1]) {
        scores[k] = prev_diag.Equal(); // equal
      } else {
        if (scores[k].cost <= scores[k - 1].cost &&
            scores[k].cost <= prev_diag.cost) { // deletion
          scores[k] = scores[k].Delete(delete_cost);
        } else if (scores[k - 1].cost <= scores[k].cost &&
                   scores[k - 1].cost <= prev_diag.cost) { // insertion
          scores[k] = scores[k - 1].Insert(insert_cost);
        } else { // replacement
          scores[k] = prev_diag.Replace(replace_cost);
        }
      }
      prev_diag = prev_diag_cache;
    }

    auto score = scores[query_length];
    if (best_score.cost == -1 || score.cost <= best_score.cost) {
      if (score.cost < best_score.cost) {
        alignments->clear();
      }
      best_score = score;
      score.position = j - 1;
      alignments->push_back(score);
    }
  }
  return best_score.cost;
}
} // namespace fasttextsearch

#endif // TEXTSEARCH_CSRC_LEVENSHTEIN_H_
