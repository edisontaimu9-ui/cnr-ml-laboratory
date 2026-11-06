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
    ("metabolic-syndrome-screen", "Metabolic syndrome screen", "classification",
     "Screen for metabolic syndrome from cheap measurements (waist, BMI, age, diet) when blood tests are not available."),
    ("prediabetes-screen", "Prediabetes / raised HbA1c finder", "classification",
     "Identify people most likely to need confirmatory testing for raised HbA1c or prediabetes when lab results are unavailable."),
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
    "metabolic-syndrome-screen": (
        "Label from the harmonised definition (IDF/AHA/NHLBI 2009, or the definition the data owner agrees): three or more "
        "of raised waist circumference, raised triglycerides, low HDL cholesterol, raised blood pressure and raised fasting "
        "glucose, or treatment for them. Triglycerides, HDL, glucose, blood pressure and treatment or diagnosis columns "
        "must never be used as features. Waist circumference is itself one of the five criteria: it is allowed here because "
        "it is the cheap measurement the screen relies on, but results must be read with that overlap in mind (the run warns, "
        "and the same model without waist is the fairer check). A positive result means 'send for blood tests', never 'has "
        "metabolic syndrome'."
    ),
    "prediabetes-screen": (
        "Label from a measured HbA1c at or above 5.7% (ADA prediabetes threshold; this includes the diabetes range, because the aim "
        "is who needs confirmatory testing), or the definition the data owner agrees. Use a fasting glucose label only if that is "
        "agreed instead, never mixed in without saying so. HbA1c, glucose, diabetes diagnosis and treatment columns must never be used "
        "as features: the model is for when lab results are unavailable. People with already-known diabetes should be left out of the "
        "dataset or analysed separately. A positive result means 'arrange confirmatory testing', never 'has prediabetes'. This is a "
        "different target from the diabetes risk model, which estimates diabetes itself."
    ),
}

# Column-name words that may not be features for a model (they define or reveal its label). Whole words, split on
# anything that is not a letter or digit, plus the run-together forms in LEAKAGE_SUBSTRINGS.
LEAKAGE_TOKENS = {
    "prediabetes-screen": {"hba1c", "a1c", "fpg", "fbg", "fbs", "rbs", "ogtt", "dm", "t2dm", "ifg", "igt"},
    "metabolic-syndrome-screen": {"tg", "hdl", "fpg", "fbg", "fbs", "sbp", "dbp", "bp", "htn", "metsyn"},
    "hypertension-risk": {"systolic", "diastolic", "sbp", "dbp", "bp", "htn", "hypertension", "hypertensive", "antihypertensive"},
}
LEAKAGE_SUBSTRINGS = {
    "prediabetes-screen": ("hba1c", "a1c", "glycated", "glycosylated", "glucose", "glycaemia", "glycemia", "diabet", "prediab",
                           "metformin", "insulin", "sulfonylurea", "sulphonylurea", "ogtt"),
    "metabolic-syndrome-screen": ("triglycer", "hdl", "glucose", "bloodpressure", "systolic", "diastolic", "hypertens",
                                  "diabet", "metabolicsyndrome", "statin", "fibrate", "lipid"),
    "hypertension-risk": ("bloodpressure", "systolic", "diastolic", "hypertens"),
}

# Features that are part of the label definition but are allowed, with a warning on every run.
LEAKAGE_WARN_SUBSTRINGS = {
    "metabolic-syndrome-screen": ("waist",),
}

SAFETY_NOTICE = (
    "Model outputs are risk estimates to support, not replace, clinical judgement. "
    "They are never confirmed diagnoses."
)
