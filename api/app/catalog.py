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
    ("ckd-nutrition-risk", "CKD nutrition risk", "classification",
     "Estimate nutrition risk in patients with chronic kidney disease."),
    ("hospital-malnutrition-risk", "Hospital malnutrition risk", "classification",
     "Estimate malnutrition risk at hospital admission to support screening."),
    ("food-insecurity-prediction", "Food insecurity", "classification",
     "Estimate household food insecurity from socioeconomic and agricultural context."),
    ("nutritional-outcome-prediction", "Nutritional outcome", "regression",
     "Predict nutritional outcomes (for example weight change or recovery time) after intervention."),
]

SAFETY_NOTICE = (
    "Model outputs are risk estimates to support, not replace, clinical judgement. "
    "They are never confirmed diagnoses."
)
