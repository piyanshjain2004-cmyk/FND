import math

from services.model_service import get_model_components
from services.text_preprocessing import preprocess


def fake_news_det(news):
    model, vector, _ = get_model_components()
    processed = preprocess(news)
    vect = vector.transform([processed])
    pred = model.predict(vect)
    return pred, vect, processed


def build_prediction_details(news):
    model, vector, metrics = get_model_components()
    pred, vect, processed = fake_news_det(news)
    label = "Fake News" if pred[0] == 1 else "Real News"

    # PassiveAggressiveClassifier exposes a decision score but not calibrated probabilities.
    score = float(model.decision_function(vect)[0])
    confidence = round(50 + 50 * math.tanh(abs(score)), 2)

    feature_names = vector.get_feature_names_out()
    vocab_weights = model.coef_[0]
    indices = vect.nonzero()[1]

    contributions = []
    for idx in indices:
        tfidf_value = vect[0, idx]
        weight = vocab_weights[idx]
        contribution = float(tfidf_value * weight)
        contributions.append((feature_names[idx], contribution))

    if label == "Fake News":
        ranked_terms = sorted(
            [item for item in contributions if item[1] > 0],
            key=lambda item: item[1],
            reverse=True,
        )
    else:
        ranked_terms = sorted(
            [item for item in contributions if item[1] < 0],
            key=lambda item: item[1],
        )

    top_terms = [term for term, _ in ranked_terms[:3]]
    if top_terms:
        reason = (
            "The model found stronger alignment with "
            f"{label.lower()} language patterns through terms such as {', '.join(top_terms)}."
        )
    else:
        reason = (
            "The model relied more on overall text structure and language distribution because there were "
            "fewer strongly weighted terms in the input."
        )

    processed_terms = processed.split()
    informative_terms = len(indices)
    metrics_accuracy = metrics.get("accuracy") if isinstance(metrics, dict) else None

    evidence_points = [
        f"Processed word count: {len(processed_terms)}",
        f"Informative terms used by the model: {informative_terms}",
        f"Decision margin score: {score:.3f}",
    ]

    if top_terms:
        evidence_points.append(f"Top signal terms: {', '.join(top_terms)}")
    if metrics_accuracy is not None:
        evidence_points.append(f"Reference test accuracy: {round(metrics_accuracy * 100, 2)}%")

    return {
        "prediction_text": f"Prediction: {label}",
        "prediction_label": label,
        "confidence": confidence,
        "reason": reason,
        "evidence_points": evidence_points,
    }
