# Model Card: Mood Machine

This model card covers the Mood Machine project, which includes two versions of a mood classifier:

1. A **rule-based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit-learn

Both models were developed and evaluated during this lab. The rule-based model is the primary system; the ML model was explored as a comparison.

---

## 1. Model Overview

**Model type:** Both models were built and compared.

**Intended purpose:**
Classify short text messages (social media style posts, messages) into one of four mood labels: `positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**

- **Rule-based:** Text is preprocessed into tokens, then each token is matched against hand-curated positive and negative word lists. Matches add or subtract from a running score. A negation check flips the contribution of any sentiment word that immediately follows a negation word. The final integer score is mapped to a label.
- **ML model:** Each post is converted to a bag-of-words vector using `CountVectorizer`. A `LogisticRegression` classifier is trained on those vectors paired with the `TRUE_LABELS` from `dataset.py`. At inference time, the same vectorizer transforms new text and the classifier predicts a label.

---

## 2. Data

**Dataset description:**
The dataset started with 6 posts from the lab starter. Nine posts were added during this lab, bringing the total to 15. Posts are stored in `SAMPLE_POSTS` in `dataset.py`, with matching labels in `TRUE_LABELS`.

**Labeling process:**
Labels were assigned by hand. Clear-cut cases (explicit positive/negative words, exclamation points, unambiguous tone) were straightforward. Harder cases required judgment:

- `"no cap this is the best day of my life 😭🙌"` — labeled `positive` because `😭` here signals overwhelmed joy, not sadness. This is a culturally specific emoji usage that could reasonably be labeled differently.
- `"it's giving chaos but also kind of fun idk"` — labeled `mixed`; the `idk` signals the speaker's own uncertainty about their feeling.
- `"whatever. doesn't matter anyway"` — labeled `negative` due to resigned/dismissive tone, despite containing no negative words from the word list.
- `"bro I failed the exam I studied 8 hours for 💀"` — labeled `negative`; `💀` is used to signal devastation, not literal death.

**Important characteristics:**

- Contains Gen-Z slang: `"no cap"`, `"lowkey"`, `"highkey"`, `"ngl"`, `"it's giving"`
- Contains Unicode emojis used in culturally specific ways (`😭` = overwhelmed joy, `💀` = devastated)
- Contains text emoticons: `:(`, `:)`
- Includes sarcasm: `"I absolutely love sitting in traffic for two hours"`
- Includes posts with mixed or ambiguous feelings
- Includes posts with no explicit sentiment words at all

**Possible issues with the dataset:**

- **Small size:** 15 examples across 4 classes is too few for reliable ML generalization (~4 positive, ~4 negative, ~4 mixed, ~3 neutral).
- **Label subjectivity:** Several posts could be labeled differently by different people. `"no cap this is the best day of my life 😭🙌"` and `"whatever. doesn't matter anyway"` are reasonable points of disagreement.
- **English only:** All posts are in English, mostly reflecting internet/social media register. The model has no exposure to other languages or dialects.
- **No truly long text:** All posts are short (under 15 words). The model has not been tested on multi-sentence text.

---

## 3. How the Rule-Based Model Works

**Preprocessing pipeline (`preprocess`):**

1. Text emoticons like `:)` and `:(` are replaced with sentiment words (`happy`, `sad`) before any other processing.
2. Common Unicode emojis are mapped to sentiment tokens (e.g., `😭` → `sad`, `🙌` → `happy`, `💀` → `devastated`). Unrecognized emojis are dropped in the punctuation removal step.
3. Text is lowercased.
4. Repeated characters are normalized: `"soooo"` → `"soo"`, `"!!!"` → `"!!"` (capped at 2 consecutive identical characters).
5. Punctuation is removed, leaving only letters, digits, and spaces.
6. Text is split on whitespace; empty tokens are dropped.

**Scoring rules (`score_text`):**

- Each token in `POSITIVE_WORDS` contributes **+1** to the score.
- Each token in `NEGATIVE_WORDS` contributes **-1** to the score.
- **Negation handling:** If the immediately preceding token is a negation word (`not`, `never`, `no`, `dont`, `doesn`, `isn`, `wasn`, `can`), the contribution of the sentiment word is **flipped**.
  - `"not happy"` → `happy` would be +1, flipped to **-1**
  - `"not bad"` → `bad` would be -1, flipped to **+1**

**Label thresholds (`predict_label`):**

| Score | Condition | Label |
|-------|-----------|-------|
| > 0 | — | `positive` |
| < 0 | — | `negative` |
| = 0 | both positive and negative tokens found | `mixed` |
| = 0 | no sentiment tokens found | `neutral` |

The `mixed` vs `neutral` distinction is important: a score of 0 does not mean the model saw nothing — it may mean equal signals cancelled out.

**Strengths:**

- Transparent and fully inspectable — `explain()` shows exactly which tokens contributed.
- Handles negation for common patterns (`"not happy"`, `"not bad"`).
- Emoji and emoticon mapping extends coverage beyond plain text.
- Predictable: the same input always produces the same output.

**Weaknesses:**

- Depends entirely on word list coverage. Words not in `POSITIVE_WORDS` or `NEGATIVE_WORDS` contribute 0, regardless of their actual sentiment. `"hopeful"` is not in the word list, so `"Feeling tired but kind of hopeful"` scores -1 (negative) instead of mixed.
- Cannot detect sarcasm. `"I absolutely love sitting in traffic for two hours"` scores +1 (positive) because `love` is in `POSITIVE_WORDS` — the sarcastic intent is invisible to the model.
- Negation window is exactly 1 token. `"not even remotely happy"` would not be caught because `"happy"` is not immediately after `"not"`.
- Contractions split unexpectedly after punctuation removal: `"don't"` → `["don", "t"]`. The negation list includes `"don"` as a workaround, but `"t"` becomes a leftover token.

---

## 4. How the ML Model Works

**Features used:**
Bag-of-words vectors using `CountVectorizer` from scikit-learn. Each post becomes a sparse vector of word counts across the full vocabulary of the training set.

**Training data:**
Trained on the 15 posts in `SAMPLE_POSTS` paired with `TRUE_LABELS` from `dataset.py`. An 80/20 train/test split was not used — the model trains and evaluates on the same data, so reported accuracy is training accuracy, not generalization accuracy.

**Training behavior:**
The ML model is highly sensitive to the labels in `TRUE_LABELS`. With only 15 examples, small label changes shift the learned weights dramatically. During interactive testing, the model predicted `negative` for `"I love traffic"` and `"I love this"` — both clearly positive. Investigation revealed that the word `"I"` co-occurs with more negative examples than positive ones in the training set, so the model learned a spurious association between `"I"` and `negative`.

**Strengths and weaknesses:**

- **Strength:** Learns patterns automatically without a hand-curated word list. With enough data, would outperform the rule-based model.
- **Weakness:** With 15 training examples, it overfits to word co-occurrence accidents rather than genuine sentiment signals. New words not seen during training (e.g., `"traffic"`) get zero weight, so predictions fall back to the most common label in the training set.
- **Weakness:** Training accuracy is not the same as generalization accuracy. The model scoring 100% on its own training data says nothing about how it would perform on new posts.

---

## 5. Evaluation

**How evaluated:**
The rule-based model was evaluated by running `predict_label` on each post in `SAMPLE_POSTS` and comparing predictions to `TRUE_LABELS`. The ML model was evaluated using `evaluate_on_dataset` in `ml_experiments.py` (training accuracy only).

**Examples of correct predictions (rule-based):**

| Post | Predicted | True | Why correct |
|------|-----------|------|-------------|
| `"I love this class so much"` | positive | positive | `love` matched +1 |
| `"Today was a terrible day"` | negative | negative | `terrible` matched -1 |
| `"I am not happy about this"` | negative | negative | negation flipped `happy` from +1 to -1 |
| `"no cap this is the best day 😭🙌"` | mixed | mixed | `😭`→`sad` (-1) and `🙌`→`happy` (+1) cancel to score=0 with both sides present |

**Examples of incorrect predictions:**

| Post | Predicted | True | Why wrong |
|------|-----------|------|-----------|
| `"Feeling tired but kind of hopeful"` | negative | mixed | `hopeful` is not in `POSITIVE_WORDS`, so only `tired` (-1) scores; the positive signal is invisible |
| `"I absolutely love sitting in traffic for two hours"` | positive | negative | Sarcasm. `love` scores +1; the model has no way to detect ironic intent |
| `"whatever. doesn't matter anyway"` | neutral | negative | No words from either word list appear; resigned tone is not detectable by vocabulary matching |

**Rule-based vs ML model comparison:**
The rule-based model and ML model failed in different ways on the interactive test:

- Rule-based correctly handled `"I love traffic"` → positive (found `love`), but failed on `"Feeling tired but kind of hopeful"` (missing word coverage).
- ML model failed on `"I love traffic"` → negative (spurious `"I"` association), but could in principle learn `"hopeful"` as positive if enough training examples contained it with a positive label.

The rule-based model is more reliable at this dataset size. The ML model needs substantially more labeled data before its learned patterns become trustworthy.

---

## 6. Limitations

**Small dataset.** 15 examples across 4 classes is not enough for either model to generalize reliably. The ML model in particular memorizes training examples rather than learning sentiment.

**Word list coverage (rule-based).** Any word not explicitly listed contributes nothing to the score. This means an entirely unlisted-word post (e.g., `"whatever. doesn't matter anyway"`) returns `neutral` even when the tone is clearly negative.

**No sarcasm detection.** Neither model can identify sarcasm. `"I absolutely love sitting in traffic for two hours"` is misclassified as `positive` by the rule-based model because it takes `love` at face value.

**Narrow negation window (rule-based).** Only the immediately preceding token triggers a negation flip. Multi-word negation phrases like `"I am really not that happy"` are not handled correctly.

**English internet register only.** The dataset consists entirely of English social media style posts. The model has no coverage of other languages, formal registers, or dialects.

**Training = test set (ML model).** The ML model's reported accuracy is measured on its own training data. This overstates real-world performance.

---

## 7. Ethical Considerations

**Misclassifying distress.** A post expressing emotional distress might be labeled `neutral` if it avoids explicit negative vocabulary. `"whatever. doesn't matter anyway"` is an example: the model returns `neutral`, but this type of phrasing can signal resignation or hopelessness. In any application involving real user wellbeing, a missed negative signal could have serious consequences.

**Sarcasm misreads amplify harm.** The model treats `"I absolutely love getting stuck in traffic"` as positive. In a context where mood is used to make decisions about a person (content moderation, mental health monitoring, HR screening), systematic sarcasm misreads would consistently misrepresent users who rely heavily on ironic speech — a common feature of many online communities.

**Dataset bias toward one language community.** All training examples use English, primarily in a Gen-Z internet register (`"no cap"`, `"lowkey"`, `"it's giving"`). Speakers whose natural register differs — formal English speakers, non-native speakers, or users of African American Vernacular English — are not represented in the training data. The model is likely to misclassify language patterns it was never trained on.

**Emoji interpretation is culturally specific.** `😭` was mapped to `sad` in preprocessing, but in many online communities it signals joy or laughter. This mapping bakes in one cultural interpretation and will misread users who use the emoji differently. The same applies to `💀` (devastated vs. humorous).

**Privacy.** Mood classification operates on personal messages. Deploying this system on real user data without consent raises significant privacy concerns, regardless of how simple the underlying model is.

---

## 8. Ideas for Improvement

- **Add more labeled data.** The most impactful change. Aim for at least 20–30 examples per class before trusting the ML model.
- **Expand the word list.** Add `"hopeful"`, `"proud"`, `"frustrated"`, `"overwhelmed"`, and other common sentiment words that currently score 0.
- **Widen the negation window.** Check the 2–3 tokens before a sentiment word instead of just 1, to catch `"not even remotely happy"`.
- **Use TF-IDF instead of CountVectorizer.** Down-weights frequent but uninformative words (like `"I"`, `"this"`) which cause the ML model's spurious associations.
- **Add a real held-out test set.** Split `SAMPLE_POSTS` into train and test to measure actual generalization, not just training accuracy.
- **Improve emoji handling.** Replace the static emoji→word map with a lookup against a sentiment emoji lexicon, or use a library like `emoji` to convert emojis to text descriptions before scoring.
- **Use a pre-trained model.** A small transformer model (e.g., DistilBERT fine-tuned on sentiment) would handle sarcasm, context, and unseen words far better than either current approach.
