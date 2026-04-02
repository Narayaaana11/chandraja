import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_eval.ml.features import extract_feature_metrics, parse_keywords
from smart_eval.ml.predict import TrainedScorePredictor
from smart_eval.services.evaluation import SemanticScorer
from smart_eval.config import BEST_MODEL_FILE, LEGACY_MODEL_FILE


def main() -> None:
    """Predict score using trained model artifacts."""
    parser = argparse.ArgumentParser(description="Predict score using trained Smart Eval model")
    parser.add_argument("--student", required=True, help="Student answer text")
    parser.add_argument("--reference", required=True, help="Reference answer text")
    parser.add_argument("--keywords", default="", help="Comma-separated keywords")
    parser.add_argument("--max-marks", type=float, default=10.0, help="Maximum marks")
    args = parser.parse_args()

    predictor = TrainedScorePredictor.from_file(BEST_MODEL_FILE)
    if predictor is None:
        predictor = TrainedScorePredictor.from_file(LEGACY_MODEL_FILE)
    if predictor is None:
        raise RuntimeError("No trained model found. Run scripts/train_model.py first.")

    scorer = SemanticScorer()
    metrics = extract_feature_metrics(
        student_text=args.student,
        reference_text=args.reference,
        keywords=parse_keywords(args.keywords),
        semantic_fn=scorer.similarity,
    )

    ratio = predictor.predict_ratio(
        semantic_score=float(metrics["semantic_score"]),
        keyword_score=float(metrics["keyword_score"]),
        length_score=float(metrics["length_score"]),
    )
    marks = ratio * args.max_marks

    print("Prediction complete")
    print(f"Semantic: {metrics['semantic_score']:.4f}")
    print(f"Keyword: {metrics['keyword_score']:.4f}")
    print(f"Length: {metrics['length_score']:.4f}")
    print(f"Predicted marks: {marks:.2f}/{args.max_marks:.2f}")


if __name__ == "__main__":
    main()
