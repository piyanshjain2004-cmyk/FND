from flask import Blueprint, render_template, request

from services.model_service import get_model_components
from services.prediction_service import build_prediction_details

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def home():
    return render_template("index.html")


@main_bp.route("/about")
def about():
    return render_template("about.html")


@main_bp.route("/how-it-works")
@main_bp.route("/why-truthlens")
def how_it_works():
    _, _, metrics = get_model_components()
    return render_template("how_it_works.html", metrics=metrics)


@main_bp.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        message = request.form.get("news")

        if not message:
            return render_template("prediction.html", prediction_text="No input provided")

        details = build_prediction_details(message)
        return render_template("prediction.html", **details)

    return render_template("prediction.html")
