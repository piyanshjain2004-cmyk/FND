import math

from api.news_verification_api import fetch_related_news
from services.model_service import get_model_components
from services.text_preprocessing import preprocess


def fake_news_det(news):
    model, vector, _ = get_model_components()
    processed = preprocess(news)
    vect = vector.transform([processed])
    pred = model.predict(vect)
    return pred, vect, processed


def build_verification_note(verification):
    status = verification.get("status")
    article_count = len(verification.get("articles", []))
    query = verification.get("query")
    trusted_match_count = verification.get("trusted_match_count", 0)
    trusted_sources = []
    for article in verification.get("articles", []):
        source = article.get("source")
        if article.get("trusted_source") and source and source not in trusted_sources:
            trusted_sources.append(source)
        if len(trusted_sources) >= 3:
            break

    status_copy = {
        "supported": "Trusted live coverage closely matches this claim",
        "mixed": "Some live coverage matches this claim, but the overlap is partial",
        "unsupported": "Trusted live coverage did not strongly match this claim",
        "unavailable": "Live verification is unavailable right now",
    }
    summary = status_copy.get(status, "Live verification is unavailable right now")

    details = []
    if trusted_match_count:
        details.append(f"{trusted_match_count} trusted-source match{'es' if trusted_match_count != 1 else ''}")
    if article_count:
        details.append(f"{article_count} related result{'s' if article_count != 1 else ''}")
    if query:
        details.append(f"query: {query}")

    if trusted_sources:
        summary += f". Trusted sources seen: {', '.join(trusted_sources)}"
    if details:
        summary += f" ({'; '.join(details)})."
    else:
        summary += "."
    return summary


def merge_api_verification(label, confidence, verification):
    if not verification or verification.get("status") == "unavailable":
        return label, confidence, None

    status = verification.get("status")
    policy_like = verification.get("policy_like", False)
    trusted_match_count = verification.get("trusted_match_count", 0)
    strong_match_count = verification.get("strong_match_count", 0)
    best_score = max((article.get("score", 0) for article in verification.get("articles", [])), default=0)
    trusted_support = trusted_match_count >= 2 or (trusted_match_count >= 1 and best_score >= 0.24)

    note = build_verification_note(verification)

    if status == "supported":
        if trusted_support:
            if label == "Fake News":
                return "Real News", max(72, round(confidence - 12, 2)), note
            return label, min(max(confidence, 82), 97), note
        if label == "Fake News":
            return "Needs Verification", min(max(confidence, 70), 82), note
        return label, min(max(confidence, 80), 95), note

    if status == "unsupported":
        if policy_like:
            if label == "Real News":
                return "Likely Fake", max(confidence, 92), note
            return "Fake News", max(confidence, 90), note
        if label == "Real News":
            if confidence >= 92:
                return "Needs Verification", 74, note
            return "Likely Fake", max(confidence, 82), note
        return label, min(max(confidence, 85), 98), note

    if status == "mixed":
        if trusted_support or strong_match_count >= 2:
            if label == "Fake News":
                return "Needs Verification", min(max(confidence, 68), 80), note
            return "Real News", min(max(confidence, 74), 90), note
        if policy_like or label == "Real News":
            return "Needs Verification", min(max(confidence, 72), 84), note
        return label, confidence, note

    return label, confidence, note


def build_prediction_details(news):
    model, vector, metrics = get_model_components()
    pred, vect, processed = fake_news_det(news)
    model_label = "Fake News" if pred[0] == 1 else "Real News"
    label = model_label

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

    if model_label == "Fake News":
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
            f"{model_label.lower()} language patterns through terms such as {', '.join(top_terms)}."
        )
    else:
        reason = (
            "The model relied more on overall text structure and language distribution because there were "
            "fewer strongly weighted terms in the input."
        )

    verification = fetch_related_news(news)
    label, confidence, verification_note = merge_api_verification(label, confidence, verification)
    api_status = verification.get("status", "unavailable") if verification else "unavailable"
    api_label_map = {
        "supported": "Trusted Source Support",
        "mixed": "Partial Trusted Coverage",
        "unsupported": "No Trusted Match Found",
        "unavailable": "Live Verification Unavailable",
    }
    api_result = api_label_map.get(api_status, "Live Verification Unavailable")
    api_articles = verification.get("articles", [])[:3] if verification else []

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
    if verification_note:
        evidence_points.append(verification_note)
    elif verification and verification.get("reason"):
        evidence_points.append(verification["reason"])
    if metrics_accuracy is not None:
        evidence_points.append(f"Reference test accuracy: {round(metrics_accuracy * 100, 2)}%")

    return {
        "prediction_text": f"Prediction: {label}",
        "prediction_label": label,
        "model_label": model_label,
        "confidence": confidence,
        "reason": reason,
        "evidence_points": evidence_points,
        "api_result": api_result,
        "api_query": verification.get("query") if verification else None,
        "api_status": api_status,
        "api_articles": api_articles,
        "api_reason": verification_note or (verification.get("reason") if verification else None),
    }
