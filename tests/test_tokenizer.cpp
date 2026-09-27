// Phase 5: tokenizer tests vs the mini fixture.
// Fixture: tests/test_data/mini_tokenizer.json (scripts/make_test_tokenizer.py).
// Expected ids verified against the Python `tokenizers` library.

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tinyinfer/tokenizer.h"

namespace {

const std::string kTok = std::string(TEST_DATA_DIR) + "/mini_tokenizer.json";

tinyinfer::Tokenizer LoadTok() {
    tinyinfer::Tokenizer t;
    t.load(kTok);
    return t;
}

TEST(Tokenizer, Loads) {
    auto t = LoadTok();
    EXPECT_TRUE(t.is_loaded());
    EXPECT_EQ(t.vocab_size(), 14);
    EXPECT_EQ(t.bos_id(), 1);
    EXPECT_EQ(t.eos_id(), 2);
    EXPECT_EQ(t.unk_id(), 0);
}

TEST(Tokenizer, BpeMergesByRank) {
    auto t = LoadTok();
    // "abc": [▁,a,b,c] -> merge (a,b) r0 -> [▁,ab,c] -> merge (ab,c) r1 -> [▁,abc]
    EXPECT_EQ(t.encode("abc", false), (std::vector<int>{9, 8}));
    // "ab": [▁,a,b] -> (a,b) r0 -> [▁,ab]; (▁,ab) has no merge
    EXPECT_EQ(t.encode("ab", false), (std::vector<int>{9, 6}));
    EXPECT_EQ(t.encode("ab", true), (std::vector<int>{1, 9, 6}));
}

TEST(Tokenizer, SpecialTokensNormalized) {
    // added_tokens have normalized=true (like the real Llama tokenizer):
    // whole input is normalized first, then split on the NORMALIZED
    // spellings ('▁<s>' etc.). A '<s>' glued to a word is NOT special.
    auto t = LoadTok();
    EXPECT_EQ(t.encode("a<s>bc", false), (std::vector<int>{10, 0, 7}));
    EXPECT_EQ(t.encode("<s>bc", false), (std::vector<int>{1, 7}));
    EXPECT_EQ(t.encode("<s> bc", false), (std::vector<int>{1, 9, 7}));
}

TEST(Tokenizer, SpecialTokensRaw) {
    // mini_tokenizer_raw.json has normalized=false added tokens: the raw
    // text is split first, each piece normalized independently.
    tinyinfer::Tokenizer t;
    t.load(std::string(TEST_DATA_DIR) + "/mini_tokenizer_raw.json");
    EXPECT_EQ(t.encode("a<s>bc", false), (std::vector<int>{10, 1, 9, 7}));
    EXPECT_EQ(t.encode("<s>bc", false), (std::vector<int>{1, 9, 7}));
}

TEST(Tokenizer, ByteFallback) {
    auto t = LoadTok();
    // 'é' (U+00E9) not in vocab -> UTF-8 bytes C3 A9 -> <0xC3><0xA9>
    EXPECT_EQ(t.encode("é", false), (std::vector<int>{9, 12, 13}));
    EXPECT_EQ(t.decode({9, 12, 13}), "é");
}

TEST(Tokenizer, Decode) {
    auto t = LoadTok();
    EXPECT_EQ(t.decode({9, 8}), "abc");
    EXPECT_EQ(t.decode({1, 9, 8, 2}), "abc"); // specials skipped
    EXPECT_EQ(t.decode({10, 9, 4}), "a b");   // ▁ -> ' ', 1 leading space stripped
}

TEST(Tokenizer, RoundTrip) {
    auto t = LoadTok();
    for (const std::string& s : {"abc", "a b", "é", "ab bc", ""}) {
        EXPECT_EQ(t.decode(t.encode(s, false)), s) << "round trip failed for " << s;
    }
}

TEST(Tokenizer, MissingFileThrows) {
    tinyinfer::Tokenizer t;
    EXPECT_THROW(t.load("/nonexistent/tok.json"), std::runtime_error);
    EXPECT_THROW(t.encode("hi", false), std::runtime_error);
}

} // namespace
