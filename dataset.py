"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    # Added posts: slang, emojis, mixed emotions, subtle tone
    "no cap this is the best day of my life 😭🙌",
    "I absolutely love sitting in traffic for two hours",
    "lowkey stressed but highkey proud of myself ngl",
    "whatever. doesn't matter anyway",
    "just got the job offer!!! 🥹✨ can't believe it",
    "it's giving chaos but also kind of fun idk",
    "so tired I could cry but at least the coffee is good ☕",
    "today happened and now it's over",
    "bro I failed the exam I studied 8 hours for 💀",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    # Added labels
    "positive",  # "no cap this is the best day of my life 😭🙌" — crying emoji used positively (overwhelmed joy)
    "negative",  # "I absolutely love sitting in traffic for two hours" — sarcasm, true sentiment is negative
    "mixed",     # "lowkey stressed but highkey proud of myself ngl" — explicit tension between two feelings
    "negative",  # "whatever. doesn't matter anyway" — resigned/dismissive, subtle negativity
    "positive",  # "just got the job offer!!! 🥹✨ can't believe it" — clear excitement
    "mixed",     # "it's giving chaos but also kind of fun idk" — uncertain, acknowledges both sides
    "mixed",     # "so tired I could cry but at least the coffee is good ☕" — negative situation, small positive
    "neutral",   # "today happened and now it's over" — flat, no emotional signal
    "negative",  # "bro I failed the exam I studied 8 hours for 💀" — frustration/disappointment; 💀 = devastated
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
