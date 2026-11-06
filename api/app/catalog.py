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
    ("refeeding-syndrome-risk", "Refeeding syndrome risk", "classification",
     "Estimate which patients are at risk of refeeding syndrome when nutrition support is started, from what is known before the first feed, "
     "so that electrolytes can be checked and feeding started with care."),
    ("malnutrition-prolonged-stay-risk", "Malnutrition-related prolonged hospital stay", "classification",
     "Estimate which admitted patients are likely to stay in hospital longer than an agreed threshold, from admission nutrition status and context, "
     "to prioritise early nutrition support."),
    ("malnutrition-readmission-risk", "Malnutrition-related readmission", "classification",
     "Estimate which patients are at risk of unplanned readmission within 30 days of discharge, from nutrition status and context known at discharge, "
     "to target nutrition follow-up."),
    ("weight-loss-trajectory", "Weight-loss trajectory", "classification",
     "Estimate which patients are likely to lose a clinically significant amount of weight over a fixed follow-up window, from baseline context, "
     "so that nutrition care can start early."),
    ("ckd-nutrition-risk", "CKD nutrition risk", "classification",
     "Estimate nutrition risk in patients with chronic kidney disease."),
    ("hospital-malnutrition-risk", "Hospital malnutrition risk", "classification",
     "Estimate malnutrition risk at hospital admission to support screening."),
    ("food-insecurity-prediction", "Food insecurity", "classification",
     "Estimate household food insecurity from socioeconomic and agricultural context."),
    ("food-name-matching", "Food-name matching", "matching",
     "Suggest registry food codes for free-text food names (English, Chichewa, local or colloquial). Suggestions for a person to confirm, never automatic assignments."),
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

LABEL_NOTES.update({
    "food-name-matching": (
        "Labels are the correct registry food code for each typed query, assigned by a person who knows the foods (for example a dietitian reviewing "
        "search logs). Queries generated from the registry's own names or aliases make scores look better than they are; use queries typed by people. "
        "This is not trained as an experiment: it is evaluated and registered under /v1/matching. Output is a ranked list of candidates for a person to "
        "confirm, with a similarity score that is not a probability. Nothing is assigned automatically."
    ),
    "refeeding-syndrome-risk": (
        "Label from measured electrolytes after nutrition support is started or substantially increased, using one definition for every row: "
        "for example the ASPEN 2020 consensus criteria (a fall in serum phosphate, potassium or magnesium of 10% or more from the pre-feeding value "
        "within 5 days of starting or increasing energy provision, graded by size of fall and organ dysfunction), or the definition the data owner "
        "agrees. Use one population per dataset: adults and children (including severe acute malnutrition on F-75 or RUTF) use different definitions "
        "and must not be mixed. Only patients whose electrolytes were rechecked have a label; leaving out those never rechecked removes the "
        "patients clinicians judged low-risk, so say how many were dropped. Features must be known before the first feed: BMI, weight loss over the "
        "last 3 to 6 months, days of little or no intake, alcohol or chemotherapy or diuretic or insulin use, HIV, TB, cancer, route of feeding, and "
        "electrolytes measured before feeding. Electrolytes must be named as baseline (for example baseline_phosphate): a column that does not say it "
        "was measured before feeding is refused, because the same measurement taken after feeding is the label. Values after feeding starts, the "
        "nadir or fall in any electrolyte, electrolyte replacement, the diagnosis itself and its complications must never be used as features. "
        "How fast feeding was started and thiamine given in advance are allowed with a warning: clinicians feed high-risk patients more slowly, so "
        "they hide the risk they respond to. A positive result means 'check electrolytes and start feeding with care', never 'has refeeding syndrome'."
    ),
    "malnutrition-prolonged-stay-risk": (
        "Label from the recorded length of stay (admission to discharge) being above a threshold fixed before training and applied to every row, for example "
        "more than 7 days or above the 75th percentile of the ward's earlier stays, not a cut-off worked out from this dataset. Decide how in-hospital "
        "deaths and transfers are handled (leaving them out of the cohort is the simplest, and the run reports how many rows were dropped): a patient who "
        "dies early has a short stay and would otherwise look like a good outcome. Stays differ by ward and diagnosis, so include them as features or "
        "train per ward. Features must be known at admission: age, sex, admission diagnosis group, route of admission, comorbidity, HIV, admission "
        "screening result (MUST, NRS-2002 or similar), BMI or MUAC, recent weight loss, albumin and intake before admission. Length of stay and the "
        "days it is counted in, discharge date, weight, destination or diagnosis, readmission, death and complications during the stay must never be used "
        "as features. Nutrition care given during the stay (dietitian review, supplements, tube or parenteral feeding) is allowed with a warning: a longer "
        "stay gives more time for it to happen, so it can look predictive only because it takes time. A positive result means 'review nutrition early', "
        "never 'the patient will stay long'."
    ),
    "malnutrition-readmission-risk": (
        "Label from an unplanned readmission within 30 days of the discharge date (any cause, or the cause the data owner agrees), counted the same way for "
        "every row. The cohort is patients discharged alive: in-hospital deaths cannot be readmitted and are left out, and planned readmissions are not "
        "counted. A patient with no readmission record is only a 'no' if follow-up is known to be complete for 30 days: readmission to another facility "
        "or loss to follow-up makes the label miss events, which flatters the model, so say how follow-up was ascertained. Features are known at "
        "discharge: length of stay, nutrition status and weight at discharge, diagnosis, comorbidity, HIV, earlier admissions, discharge destination, "
        "household food security, distance to the facility and the nutrition plan at discharge. Anything that happens after discharge (readmission, return "
        "or attendance, follow-up weights or visits, death after discharge, emergency visits) must never be used as features. A supplement, "
        "therapeutic food or review arranged at discharge is allowed with a warning: it is decided because the patient looked at risk, so treated patients can "
        "look low-risk. A positive result means 'arrange nutrition follow-up', never 'the patient will be readmitted'."
    ),
    "weight-loss-trajectory": (
        "Label from the percentage weight change between a baseline weight and a weight at the end of a fixed window, measured the same way and over the same "
        "window for every row: for example 5% or more lost within 3 months (yes or no), or bands such as under 5%, 5 to under 10% and 10% or more. "
        "The window and the cut-offs are agreed with the data owner before training. Patients who die or are lost before the end weight have no label: "
        "leaving them out removes the fastest losers, so the run reports how many rows were dropped and the result must be read with that in mind. "
        "Features must be known at baseline: age, sex, diagnosis and stage, HIV or TB status, treatment plan, appetite, symptoms such as nausea or "
        "difficulty swallowing, intake, food security and loss before baseline. Every weight, BMI, MUAC or cachexia column must say when it was taken "
        "(baseline, admission, enrolment, initial, prior or previous, for example weight_loss_prior_6m_pct): a column that does not is refused, because "
        "the same measurement at follow-up is the label. Follow-up and final weights, weight change, velocity or slope, and any loss or trajectory "
        "flag must never be used as features. Nutrition care given during the window is allowed with a warning. A positive result means 'review nutrition "
        "early and re-weigh soon', never a diagnosis."
    ),
})

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

LEAKAGE_TOKENS.update({
    "refeeding-syndrome-risk": {"rfs", "d1", "d2", "d3", "d4", "d5", "d7", "h24", "h48", "h72", "24h", "48h", "72h"},
    "malnutrition-prolonged-stay-risk": {"los", "alos", "dod", "died", "dead"},
    "malnutrition-readmission-risk": {"died", "dead"},
    "weight-loss-trajectory": {"died", "dead"},
})
LEAKAGE_SUBSTRINGS.update({
    "refeeding-syndrome-risk": ("refeedingsyndrome", "refeedinghypo", "refeedingcomplication", "refeedingevent", "refeedingoutcome", "refeedingdiagnos",
                                "nadir", "postfeed", "afterfeed", "duringfeed", "followup", "day1", "day2", "day3", "day4", "day5", "day7",
                                "replacement", "repletion", "replete", "ivphosphate", "ivpotassium", "ivmagnesium", "electrolytedrop",
                                "electrolytefall", "electrolytechange", "organdysfunction", "cardiacarrest", "arrhythm"),
    "malnutrition-prolonged-stay-risk": ("lengthofstay", "staylength", "stayduration", "daysinhospital", "daysadmitted", "hospitalstay", "prolonged",
                                         "longstay", "discharge", "readmi", "mortality", "death", "deceased", "complication", "nosocomial",
                                         "hospitalacquired", "duringstay", "instay", "inhospital"),
    "malnutrition-readmission-risk": ("readmi", "reattend", "returnvisit", "returnedto", "daystoreturn", "postdischarge", "afterdischarge",
                                      "followupweight", "followupvisit", "followupattend", "attendedfollowup", "followupdate", "followupoutcome",
                                      "followupresult", "followupstatus", "lostto", "mortality", "death", "deceased", "emergencyvisit"),
    "weight-loss-trajectory": ("followup", "finalweight", "endweight", "outcomeweight", "weightchange", "percentchange", "pctchange", "deltaweight",
                               "weightdelta", "velocity", "slope", "trajectory", "lossflag", "significantloss", "severeloss", "lostweight",
                               "mortality", "death", "deceased"),
})
LEAKAGE_WARN_SUBSTRINGS.update({
    "refeeding-syndrome-risk": ("thiamine", "prophyla", "startingenergy", "startingkcal", "initialenergy", "initialkcal", "kcalkg", "feedingrate",
                                "slowstart", "energyrestrict"),
    "malnutrition-prolonged-stay-risk": ("dietitian", "dietician", "nutritionreferral", "nutritionsupport", "nutritionalsupport", "nutritionconsult",
                                         "nutritionreview", "oralnutritionalsuppl", "tubefeed", "enteralfeed", "parenteral", "feedingstarted",
                                         "supplementstarted", "ons"),
    "malnutrition-readmission-risk": ("supplement", "rutf", "therapeuticfood", "nutritionplan", "nutritionreferral", "nutritionreview", "ons"),
    "weight-loss-trajectory": ("counsel", "intervention", "supplement", "dietitian", "dietician", "rutf", "appetitestimul", "ons"),
})
LEAKAGE_WARN_NOTES.update({
    "refeeding-syndrome-risk": (
        "'{col}' records a decision taken because a patient looked at risk (how fast feeding was started, or thiamine given in advance), and it also "
        "changes what happens next. Patients judged high-risk are fed more slowly, which lowers their observed risk, so the model can learn that "
        "cautious feeding means 'low risk'. Do not read it as a causal effect, and compare with a run that leaves it out."
    ),
    "malnutrition-prolonged-stay-risk": (
        "'{col}' records nutrition care that happens during the stay. A longer stay gives more time for it to happen, so it can look like a predictor "
        "of long stays simply because it takes time. It is only sound if it is always recorded at the same early point (for example within 24 hours "
        "of admission). Train the same model without it and compare."
    ),
    "malnutrition-readmission-risk": (
        "'{col}' records support arranged at discharge. It is decided because the patient looked at risk, and it also changes what happens next, so "
        "treated patients can look low-risk. Do not read it as the effect of that support, and compare with a run that leaves it out."
    ),
    "weight-loss-trajectory": (
        "'{col}' records nutrition care or treatment given during the follow-up window, after baseline. It is a response to expected weight loss and "
        "also changes it, so it makes the scores hard to interpret. Use it only if the model will run after that care has been decided, and compare "
        "with a run that leaves it out."
    ),
})

# Measurements that are a legitimate baseline feature or the label itself depending on when they were taken. For these models a column is refused
# unless its name states the timing (one of TIMING_MARKERS), so the user has to say so; short markers match whole words only.
TIMING_REQUIRED = {
    "refeeding-syndrome-risk": ("phosphate", "phosphorus", "po4", "potassium", "magnesium", "hypokal", "hypophos", "hypomag", "electrolyte"),
    "weight-loss-trajectory": ("weight", "bmi", "muac", "armcirc", "midarm", "skinfold", "cachexia", "sarcopenia", "wasting"),
}
TIMING_MARKERS = {
    "refeeding-syndrome-risk": ("baseline", "admission", "prefeed", "prefeeding", "beforefeed", "beforefeeding", "initial", "screening", "enrol", "pre"),
    "weight-loss-trajectory": ("baseline", "admission", "enrol", "initial", "prior", "previous", "usual", "screening", "entry", "pre"),
}
TIMING_NOTES = {
    "refeeding-syndrome-risk": (
        "'{col}' is an electrolyte value or condition and does not say when it was measured. Only values taken before feeding started can be features: "
        "the same measurement after feeding is the label. Name it so that it says so (baseline_, admission_ or prefeeding_, for example "
        "baseline_phosphate) or exclude it, and run again."
    ),
    "weight-loss-trajectory": (
        "'{col}' is a weight or body-size measure and does not say when it was taken. Only baseline values, or loss before baseline, can be features: "
        "the same measurement at follow-up is the label. Name it so that it says so (baseline_, admission_, enrolment_, prior_ or previous_, for example "
        "baseline_weight_kg or weight_loss_prior_6m_pct) or exclude it, and run again."
    ),
}

SAFETY_NOTICE = (
    "Model outputs are risk estimates to support, not replace, clinical judgement. "
    "They are never confirmed diagnoses."
)
