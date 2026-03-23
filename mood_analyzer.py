# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps applied in order:
          1. Replace text emoticons (":)", ":-(") with sentiment words
          2. Replace common emojis with sentiment words
          3. Lowercase everything
          4. Normalize repeated characters ("soooo" -> "soo")
          5. Remove punctuation
          6. Split on whitespace and drop empty tokens
        """
        # 1. Text emoticons → sentiment words
        emoticon_map = {
            ":)": "happy", ":-)": "happy", ":D": "happy", "=)": "happy",
            ":(": "sad",   ":-(": "sad",   ":/": "sad",
        }
        for emoticon, replacement in emoticon_map.items():
            text = text.replace(emoticon, f" {replacement} ")

        # 2. Unicode emojis → sentiment words (unrecognized emojis are dropped in step 5)
        emoji_map = {
            "😊": "happy",  "😀": "happy",  "😁": "happy",  "🙌": "happy",
            "✨": "happy",  "❤️": "love",   "💕": "love",   "😂": "happy",
            "🤣": "happy",  "🥹": "happy",
            "😭": "sad",    "😢": "sad",    "😞": "sad",    "😔": "sad",
            "😡": "angry",  "😤": "angry",  "😠": "angry",
            "💀": "devastated", "😱": "shocked", "😐": "neutral",
        }
        for emoji, replacement in emoji_map.items():
            text = text.replace(emoji, f" {replacement} ")

        # 3. Lowercase
        text = text.lower()

        # 4. Normalize repeated characters: "soooo" -> "soo", "!!!" -> "!!"
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # 5. Remove punctuation (keep letters, digits, spaces)
        text = re.sub(r"[^\w\s]", " ", text)

        # 6. Split and drop empty tokens
        tokens = [t for t in text.split() if t]

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Base logic:
          - Each positive word token: +1
          - Each negative word token: -1

        Enhancement — negation handling:
          If a negation word ("not", "never", "no", "dont", "doesn", "isn",
          "wasn", "can") appears directly before a sentiment word, the score
          contribution of that sentiment word is flipped.

          Examples:
            "not happy"  -> "happy" would be +1, flipped to -1
            "not bad"    -> "bad" would be -1, flipped to +1
        """
        NEGATIONS = {"not", "never", "no", "dont", "doesn", "isn", "wasn", "can"}

        tokens = self.preprocess(text)
        score = 0

        for i, token in enumerate(tokens):
            # Check whether the previous token was a negation word
            negated = i > 0 and tokens[i - 1] in NEGATIONS

            if token in self.positive_words:
                score += -1 if negated else 1
            elif token in self.negative_words:
                score += 1 if negated else -1

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        Mapping:
          - score > 0                          -> "positive"
          - score < 0                          -> "negative"
          - score == 0, both pos & neg hits    -> "mixed"
          - score == 0, no sentiment hits      -> "neutral"

        Why distinguish mixed from neutral:
          A score of 0 can mean two things — the text had no sentiment words
          at all (neutral), or it had equal positive and negative signals that
          cancelled out (mixed). Checking for both hit types separates them.
        """
        score  = self.score_text(text)
        tokens = self.preprocess(text)

        has_positive = any(t in self.positive_words for t in tokens)
        has_negative = any(t in self.negative_words for t in tokens)

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        # score == 0
        if has_positive and has_negative:
            return "mixed"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )


if __name__ == "__main__":
    analyzer = MoodAnalyzer()

    test_cases = [
        # (post, expected_label, note)
        ("I love this class so much",          "positive", "plain positive"),
        ("Today was a terrible day",           "negative", "plain negative"),
        ("I am not happy about this",          "negative", "negation flips positive"),
        ("not bad at all",                     "positive", "negation flips negative"),
        ("Feeling tired but kind of hopeful",  "mixed",    "cancelled signals → mixed"),
        ("no cap this is the best day 😭🙌",  "mixed",    "emoji sad+happy cancel → mixed"),
        ("whatever. doesn't matter anyway",    "neutral",  "no sentiment words → neutral"),
        ("This is fine",                       "neutral",  "no sentiment words → neutral"),
    ]

    for post, expected, note in test_cases:
        score     = analyzer.score_text(post)
        predicted = analyzer.predict_label(post)
        match     = "✓" if predicted == expected else f"✗ (expected {expected})"
        print(f"[{note}]")
        print(f"  Input    : {post}")
        print(f"  Score    : {score}  →  {predicted}  {match}")
        print()
