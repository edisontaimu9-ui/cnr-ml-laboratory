"""Planned model catalogue. These are *architecture slots*, not trained models.
No model is marked available until it has a dataset, an evaluation report and
clinical-reviewer sign-off (see docs/GOVERNANCE.md)."""

LABEL_NOTE = (
    "To be defined with the data owner before any training. Where the label is a "
    "nutrition status, derive it from CNR's deterministic calculations rather than "
    "re-implementing them here."
)

CATALOG = [
    ("child-malnutrition-risk", "Childhood malnutrition risk", "classification",
     "Estimate the risk of stunting, wasting or underweight in children under five from household, feeding and anthropometric context."),
    ("acute-malnutrition-classification", "Acute malnutrition classification", "classification",
     "Classify acute malnutrition category from anthropometric and clinical features, with labels taken from CNR's deterministic assessment."),
    ("diabetes-risk", "Diabetes risk", "classification",
     "Estimate risk of diabetes from clinical, anthropometric and dietary features."),
    ("hypertension-risk", "Hypertension risk", "classification",
     "Estimate risk of hypertension from age, BMI, waist circumference, diet (salt, alcohol) and physical activity, "
     "for community screening when a blood pressure cuff is not always available."),
    ("ckd-nutrition-risk", "CKD nutrition risk", "classification",
     "Estimate nutrition risk in patients with chronic kidney disease."),
    ("hospital-malnutrition-risk", "Hospital malnutrition risk", "classification",
     "Estimate malnutrition risk at hospital admission to support screening."),
    ("food-insecurity-prediction", "Food insecurity", "classification",
     "Estimate household food insecurity from socioeconomic and agricultural context."),
    ("nutritional-outcome-prediction", "Nutritional outcome", "regression",
     "Predict nutritional outcomes (for example weight change or recovery time) after intervention."),
]

# Models whose label needs a specific definition, shown instead of LABEL_NOTE.
LABEL_NOTES = {
    "hypertension-risk": (
        "Label from measured blood pressure (for example average of two or more seated readings at or above 140/90 mmHg, "
        "or the threshold the data owner agrees). Blood pressure readings, a recorded hypertension diagnosis and "
        "antihypertensive treatment must never be used as features: the model is meant to work where no cuff is available. "
        "Features: age, BMI, waist circumference, salt and alcohol intake, physical activity. A positive result means "
        "'measure blood pressure', never 'has hypertension'."
    ),
}

# Column-name words that may not be features for a model (they define or reveal its label). Whole words, split on
# anything that is not a letter or digit, plus the run-together forms in LEAKAGE_SUBSTRINGS.
LEAKAGE_TOKENS = {
    "hypertension-risk": {"systolic", "diastolic", "sbp", "dbp", "bp", "htn", "hypertension", "hypertensive", "antihypertensive"},
}
LEAKAGE_SUBSTRINGS = {
    "hypertension-risk": ("bloodpressure", "systolic", "diastolic", "hypertens"),
}

SAFETY_NOTICE = (
    "Model outputs are risk estimates to support, not replace, clinical judgement. "
    "They are never confirmed diagnoses."
)
