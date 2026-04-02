import re
from typing import Callable, Dict, List, Tuple


def parse_keywords(raw_keywords: str | List[str]) -> List[str]:
    if isinstance(raw_keywords, str):
        keywords = [k.strip() for k in raw_keywords.split(",")]
    else:
        keywords = [str(k).strip() for k in raw_keywords]
    return [k for k in keywords if k]


def keyword_match(student_text: str, keywords: List[str]) -> Tuple[float, List[str], List[str]]:
    student_lower = (student_text or "").lower()
    student_tokens = set(re.findall(r"\b\w+\b", student_lower))

    found_keywords: List[str] = []
    missing_keywords: List[str] = []

    for key in keywords:
        normalized = key.lower().strip()
        if not normalized:
            continue

        is_phrase = " " in normalized
        phrase_match = re.search(rf"\b{re.escape(normalized)}\b", student_lower)

        if (is_phrase and phrase_match) or (not is_phrase and normalized in student_tokens):
            found_keywords.append(key)
        else:
            missing_keywords.append(key)

    keyword_score = len(found_keywords) / len(keywords) if keywords else 0.0
    return float(keyword_score), found_keywords, missing_keywords


def length_match(student_text: str, reference_text: str) -> float:
    ideal_len = len((reference_text or "").split())
    student_len = len((student_text or "").split())
    if ideal_len <= 0:
        return 0.0
    return float(min(student_len / ideal_len, 1.0))


def extract_feature_metrics(
    student_text: str,
    reference_text: str,
    keywords: List[str],
    semantic_fn: Callable[[str, str], float],
) -> Dict[str, object]:
    semantic_score = float(max(0.0, min(semantic_fn(student_text, reference_text), 1.0)))
    keyword_score, found_keywords, missing_keywords = keyword_match(student_text, keywords)
    length_score = length_match(student_text, reference_text)

    return {
        "semantic_score": round(semantic_score, 4),
        "keyword_score": round(keyword_score, 4),
        "length_score": round(length_score, 4),
        "found_keywords": found_keywords,
        "missing_keywords": missing_keywords,
    }


def to_feature_vector(metrics: Dict[str, object]) -> List[float]:
    return [
        float(metrics["semantic_score"]),
        float(metrics["keyword_score"]),
        float(metrics["length_score"]),
    ]
