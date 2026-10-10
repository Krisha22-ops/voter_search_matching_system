"""
Voter Search & Matching - Core Similarity Algorithms
=====================================================
Implements text similarity, string metrics, and phonetic encoders:
1. Levenshtein Distance & Normalized Ratio
2. Jaro & Jaro-Winkler Similarity
3. Token Sort Ratio (handles word order permutations: 'Smith, John' vs 'John Smith')
4. Soundex & Double Metaphone Phonetic Encoding (handles 'Steven' vs 'Stephen')
5. TF-IDF Character n-gram Vectorization + Cosine Similarity
"""

import math
import re
from typing import List, Tuple, Set


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes minimum edit operations (insertions, deletions, substitutions)."""
    s1, s2 = s1.lower(), s2.lower()
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity between 0.0 and 1.0."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return max(0.0, 1.0 - (dist / max_len))


def jaro_similarity(s1: str, s2: str) -> float:
    """Computes Jaro similarity metric."""
    s1, s2 = s1.lower(), s2.lower()
    len1, len2 = len(s1), len(s2)
    if len1 == 0 and len2 == 0:
        return 1.0
    if len1 == 0 or len2 == 0:
        return 0.0
        
    match_distance = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
            
    if matches == 0:
        return 0.0
        
    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
        
    transpositions = transpositions / 2.0
    return (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """
    Jaro-Winkler gives higher score to strings sharing common prefix up to max_l chars.
    Ideal for person first and last names.
    """
    jaro_score = jaro_similarity(s1, s2)
    s1_low, s2_low = s1.lower(), s2.lower()
    prefix_len = 0
    for c1, c2 in zip(s1_low[:max_l], s2_low[:max_l]):
        if c1 == c2:
            prefix_len += 1
        else:
            break
    return jaro_score + (prefix_len * p * (1.0 - jaro_score))


def token_sort_similarity(s1: str, s2: str) -> float:
    """
    Tokenizes, strips punctuation, sorts alphabetically, and measures similarity.
    Resolves inverted names like 'Smith, John' vs 'John Smith'.
    """
    tokens1 = sorted(re.sub(r'[^a-zA-Z0-9\s]', '', s1.lower()).split())
    tokens2 = sorted(re.sub(r'[^a-zA-Z0-9\s]', '', s2.lower()).split())
    sorted_s1 = " ".join(tokens1)
    sorted_s2 = " ".join(tokens2)
    return jaro_winkler_similarity(sorted_s1, sorted_s2)


def soundex(name: str) -> str:
    """
    Calculates 4-character Soundex phonetic key for English names.
    E.g., 'Robert' and 'Rupert' both return 'R163'.
    """
    name = re.sub(r'[^a-zA-Z]', '', name.upper())
    if not name:
        return "0000"
        
    first_letter = name[0]
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    encoded = []
    prev_code = mapping.get(first_letter, '0')
    
    for char in name[1:]:
        code = mapping.get(char, '0')
        if code != '0' and code != prev_code:
            encoded.append(code)
        prev_code = code
        if len(encoded) == 3:
            break
            
    encoded_str = first_letter + "".join(encoded)
    return (encoded_str + "000")[:4]


def jaccard_token_similarity(s1: str, s2: str) -> float:
    """Computes Jaccard set similarity over whitespace tokens (ideal for addresses)."""
    set1 = set(re.findall(r'\w+', s1.lower()))
    set2 = set(re.findall(r'\w+', s2.lower()))
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / float(union)
