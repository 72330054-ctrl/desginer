from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ==================================
# LOAD MODEL
# ==================================

saved = joblib.load("designer_model (1).pkl")

model = saved["model"]
question_encoders = saved["question_encoders"]
target_encoder = saved["target_encoder"]
feature_columns = saved["feature_columns"]

# ==================================
# SKILL NAME MAPPING
# ==================================

skill_names = {
    "CC": "Creative Confidence",
    "DMU": "Decision-Making Under Uncertainty",
    "CF": "Cognitive Flexibility",
    "PF": "Problem Framing",
    "AT": "Abstract Thinking",
    "VPA": "Visual Perception & Attention",
    "ER": "Emotional Resilience",
    "MD": "Motivation & Discipline",
    "PV": "Perception of Value",
    "IF": "Identity Formation"
}

# ==================================
# HOME PAGE
# ==================================

@app.route("/", methods=["GET", "POST"])
def index():

    top_skills = []

    if request.method == "POST":

        # ==========================
        # GET FORM ANSWERS
        # ==========================

        q1 = request.form["Q1"]
        q2 = request.form["Q2"]
        q3 = request.form["Q3"]
        q4 = request.form["Q4"]
        q5 = request.form["Q5"]
        q6 = request.form["Q6"]
        q7 = request.form["Q7"]
        q8 = request.form["Q8"]

        # ==========================
        # CREATE DATAFRAME
        # ==========================

        input_df = pd.DataFrame([{
            "Q1": q1,
            "Q2": q2,
            "Q3": q3,
            "Q4": q4,
            "Q5": q5,
            "Q6": q6,
            "Q7": q7,
            "Q8": q8
        }])

        # ==========================
        # ENCODE INPUTS
        # ==========================

        for col in feature_columns:
            input_df[col] = question_encoders[col].transform(
                input_df[col]
            )

        # ==========================
        # GET PROBABILITIES
        # ==========================

        probabilities = model.predict_proba(input_df)[0]

        # Top 4 predictions
        top_indices = probabilities.argsort()[-4:][::-1]

        # ==========================
        # BUILD RESULTS
        # ==========================

        for idx in top_indices:

            short_skill = target_encoder.inverse_transform([idx])[0]

            full_skill = skill_names.get(
                short_skill,
                short_skill
            )

            confidence = round(
                probabilities[idx] * 100,
                2
            )

            top_skills.append({
                "skill": full_skill,
                "confidence": confidence
            })

    return render_template(
        "index.html",
        top_skills=top_skills
    )

# ==================================
# RUN APP
# ==================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)