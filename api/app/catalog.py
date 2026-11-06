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
    ("low-birth-weight-risk", "Low birth weight risk", "classification",
     "Estimate the risk of a baby being born under 2,500 g from maternal and antenatal context, to prioritise extra antenatal and nutrition support."),
    ("preterm-growth-faltering", "Preterm infant growth faltering", "classification",
     "Estimate which preterm infants are likely to fall away from their growth trajectory (Fenton reference) in the neonatal unit, from early-life context."),
    ("school-age-overweight-risk", "Adolescent / school-age overweight risk", "classification",
     "Estimate which school-age children and adolescents (5 to 19 years) are at risk of overweight, from household, diet and activity context."),
    ("school-age-stunting-risk", "Adolescent / school-age stunting risk", "classification",
     "Estimate which school-age children and adolescents (5 to 19 years) are at risk of stunting, from household, diet and early-life context."),
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
    "low-birth-weight-risk": (
        "Label from the measured birth weight: under 2,500 g (low birth weight), taken within the first hour after birth. If the data owner "
        "wants very low (under 1,500 g) or small-for-gestational-age as the target, that is a different label and must be said so. Birth "
        "weight and everything derived from it or measured at or after birth (low-birth-weight and SGA flags, birth length, head "
        "circumference, Apgar) must never be used as features. Features are maternal and antenatal: age, height, BMI or MUAC, haemoglobin, "
        "parity, interval since last birth, antenatal visits, malaria prophylaxis, and similar. Gestational age at delivery and preterm "
        "birth are known only at delivery: they are allowed with a warning, but only make sense if the model is used at birth, not "
        "during pregnancy. A high estimate means 'offer extra support and plan the delivery', never 'the baby will be small'."
    ),
    "preterm-growth-faltering": (
        "Label from Fenton preterm growth reference values calculated by CNR or the Chakudya API's deterministic Fenton calculations, "
        "not re-implemented here: for example a fall in weight z-score of at least 1.0 from birth to discharge or 36 weeks postmenstrual "
        "age, or weight below the 10th percentile at that point. The data owner agrees the definition and the time point before any "
        "training, and every row must use the same one. Features must be known early (by about the end of the first week): gestational "
        "age, birth weight and birth z-score, sex, multiple birth, antenatal steroids, early illness, ventilation, time to first feed and "
        "early nutrition intake. Discharge or later weights, later z-scores or percentiles, z-score change, growth velocity, weight gain "
        "and any faltering or EUGR column must never be used as features. A baby born with a high z-score has further to fall, so birth "
        "z-score is expected to matter; read results with that in mind. A positive result means 'review feeding and growth closely', "
        "never a diagnosis."
    ),
    "school-age-overweight-risk": (
        "Label from BMI-for-age z-score above +1 SD (overweight, which includes obesity) on the WHO 2007 growth reference for 5 to 19 "
        "years, or above +2 SD if the data owner wants obesity only. Calculated by CNR or the Chakudya API's deterministic BMI-for-age "
        "classification from age in months, sex, weight and height, not re-implemented here. The reference covers 61 to 228 months; "
        "younger children need the under-five tools. The child's own weight, height, BMI, z-scores, percentiles and "
        "overweight/obesity/stunting/thinness columns must never be used as features. Parents', mother's and birth measurements are fine (name them maternal_ or parental_: the upload step drops columns named mother, father or caregiver as possible personal names). "
        "Features: age, sex, household and school context, diet (sugary drinks, fried food, school meals), physical activity, screen time, "
        "sleep, mother's or parents' BMI, birth weight. Waist and similar body-size measures are allowed with a warning. Pubertal timing "
        "affects BMI-for-age in adolescents, so results differ by age band. A positive result means 'review growth and diet', never "
        "'the child is overweight'."
    ),
    "school-age-stunting-risk": (
        "Label from height-for-age z-score below -2 SD on the WHO 2007 growth reference for 5 to 19 years, calculated by CNR or the "
        "Chakudya API's deterministic growth assessment from age in months, sex and height, not re-implemented here. The reference "
        "covers 61 to 228 months; younger children need the under-five tools. The child's own height, height-for-age z-score or "
        "percentile, and anything that stands in for height (knee height, ulna length, demi-span, arm span, stunted columns) must never "
        "be used as features. Mother's and parents' heights and birth measurements are fine (name them maternal_ or parental_: the upload step drops columns named mother, father or caregiver as possible personal names). Features: age, sex, household food "
        "security, dietary diversity, water and sanitation, maternal education, mother's height, birth weight, earlier undernutrition "
        "records. The child's own weight, BMI, MUAC and waist track height closely: they are allowed with a warning and only make sense "
        "if the model is meant for settings where height is not measured. A positive result means 'measure height and review diet', "
        "never 'the child is stunted'."
    ),
}

# Column-name words that may not be features for a model (they define or reveal its label). Whole words, split on
# anything that is not a letter or digit, plus the run-together forms in LEAKAGE_SUBSTRINGS.
LEAKAGE_TOKENS = {
    "school-age-overweight-risk": {"bmi", "haz", "baz", "waz", "whz", "bfa", "hfa"},
    "school-age-stunting-risk": {"haz", "hfa", "baz", "waz", "stunted"},
    "low-birth-weight-risk": {"bw", "bwt", "lbw", "vlbw", "elbw", "sga", "lga", "aga", "apgar"},
    "preterm-growth-faltering": {"pgf", "eugr", "ugr", "pgr"},
    "prediabetes-screen": {"hba1c", "a1c", "fpg", "fbg", "fbs", "rbs", "ogtt", "dm", "t2dm", "ifg", "igt"},
    "metabolic-syndrome-screen": {"tg", "hdl", "fpg", "fbg", "fbs", "sbp", "dbp", "bp", "htn", "metsyn"},
    "hypertension-risk": {"systolic", "diastolic", "sbp", "dbp", "bp", "htn", "hypertension", "hypertensive", "antihypertensive"},
}
LEAKAGE_SUBSTRINGS = {
    "school-age-overweight-risk": ("weight", "height", "stature", "zscore", "percentile", "centile", "overweight", "obes", "stunt",
                                   "thinness", "wasting", "underweight", "nutritionstatus", "skinfold", "bodyfat", "bodymass", "bmifor"),
    "school-age-stunting-risk": ("height", "stature", "kneeheight", "ulna", "demispan", "armspan", "zscore", "percentile", "centile",
                                 "stunt", "heightforage", "nutritionstatus", "overweight", "obes", "thinness", "wasting", "underweight"),
    "low-birth-weight-risk": ("birthweight", "weightatbirth", "wtatbirth", "lowbirth", "birthlength", "lengthatbirth", "birthhead",
                              "headcircumferenceatbirth", "apgar", "smallforgest", "neonataloutcome"),
    "preterm-growth-faltering": ("falter", "growthfailure", "growthrestrict", "extrauterine", "discharge", "followup", "36week",
                                 "zchange", "changeinz", "deltaz", "zscorechange", "velocity", "weightgain", "gainrate", "growthrate"),
    "prediabetes-screen": ("hba1c", "a1c", "glycated", "glycosylated", "glucose", "glycaemia", "glycemia", "diabet", "prediab",
                           "metformin", "insulin", "sulfonylurea", "sulphonylurea", "ogtt"),
    "metabolic-syndrome-screen": ("triglycer", "hdl", "cholesterol", "glucose", "bloodpressure", "systolic", "diastolic", "hypertens",
                                  "diabet", "metabolicsyndrome", "statin", "fibrate", "lipid"),
    "hypertension-risk": ("bloodpressure", "systolic", "diastolic", "hypertens"),
}

# Features that are part of the label definition but are allowed, with a warning on every run.
# Name parts that make a column NOT the child's own measurement, so the guards above do not apply to it.
LEAKAGE_EXEMPT = {
    "school-age-overweight-risk": ("mother", "maternal", "father", "paternal", "parent", "caregiver", "sibling", "birth"),
    "school-age-stunting-risk": ("mother", "maternal", "father", "paternal", "parent", "caregiver", "sibling", "birth"),
}

LEAKAGE_WARN_SUBSTRINGS = {
    "school-age-overweight-risk": ("waist", "hipcirc", "muac", "neckcirc", "armcirc"),
    "school-age-stunting-risk": ("weight", "bmi", "bodymass", "muac", "waist", "headcirc"),
    "low-birth-weight-risk": ("gestation", "preterm", "premat"),
    "metabolic-syndrome-screen": ("waist",),
}

# Reason shown for a warned feature, where the generic "part of the label definition" text does not fit. {col} is the column name.
LEAKAGE_WARN_NOTES = {
    "school-age-overweight-risk": (
        "'{col}' is a body-size measurement that tracks the label closely. It is allowed because it is cheap to take, but scores overstate "
        "how well the household, diet and activity features predict overweight. Train the same model with it excluded and compare."
    ),
    "school-age-stunting-risk": (
        "'{col}' tracks height closely, so it carries much of the signal about stunting. It is allowed only if the model is meant for "
        "settings where height is not measured. Train the same model with it excluded and compare before trusting the result."
    ),
    "low-birth-weight-risk": (
        "'{col}' is only known at or after delivery. It is allowed only if this model is meant to be used at birth. For antenatal use, "
        "leave it out: it carries a large part of the signal, so scores overstate how well the model works during pregnancy. "
        "Train the same model with it excluded and compare."
    ),
}

SAFETY_NOTICE = (
    "Model outputs are risk estimates to support, not replace, clinical judgement. "
    "They are never confirmed diagnoses."
)
