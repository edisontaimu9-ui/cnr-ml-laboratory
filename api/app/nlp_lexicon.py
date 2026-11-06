"""Terminology for the NLP laboratory: nutrition and clinical terms in English and Chichewa.

Every entry is `reviewed: False`. The lists were written by the developers, not validated by clinicians or Chichewa
speakers; they are meant to be reviewed and extended. Chichewa verbs change form (aku-, ama-, wa-, ...): only the forms listed here are found, and
spelling variants beyond apostrophes, hyphens and spaces must be added. A term that is not listed is simply not found."""

LABELS = {
    "FOOD": "A food, drink or prepared dish (with a food group)",
    "CONDITION": "A diagnosis or nutrition-related condition",
    "SYMPTOM": "A symptom or clinical sign",
    "INTERVENTION": "A nutrition treatment, product, supplement or feeding practice",
    "NUTRIENT": "A nutrient or dietary component",
    "POPULATION": "A patient group (age or physiological state)",
    "MEASUREMENT": "A number with a unit and a type (weight, MUAC, haemoglobin, age, portion, ...)",
    "PHI": "Something that identifies a person (email, phone, ID number, titled name)",
}

# (label, group, lang, case_sensitive, terms). A term may be "text" or "text=normalised form".
_DEFS: list[tuple[str, str, str, bool, list[str]]] = [
    # ---- foods
    ("FOOD", "cereal", "both", False, ["nsima", "maize", "maize flour", "rice", "sorghum", "millet", "bread", "wheat", "porridge",
                                       "mgaiwa=maize flour", "chimanga=maize", "mpunga=rice", "mapira=sorghum", "phala=porridge"]),
    ("FOOD", "cereal", "ny", False, ["ufa=flour"]),
    ("FOOD", "legume", "both", False, ["beans", "groundnuts", "peanuts", "pigeon peas", "cowpeas", "soya", "soya beans", "peanut butter",
                                       "nyemba=beans", "mtedza=groundnuts", "nandolo=pigeon peas", "chiponde=peanut butter"]),
    ("FOOD", "animal source", "both", False, ["fish", "meat", "beef", "chicken", "eggs", "kapenta", "usipa", "chambo",
                                              "nsomba=fish", "nyama=meat", "nkhuku=chicken", "mazira=eggs", "dzira=eggs"]),
    ("FOOD", "dairy", "both", False, ["milk", "yoghurt", "yogurt", "cheese", "breast milk", "infant formula",
                                      "mkaka=milk", "mkaka wa m'mawere=breast milk"]),
    ("FOOD", "vegetable", "both", False, ["vegetables", "pumpkin leaves", "amaranth", "okra", "pumpkin", "tomatoes", "onions", "cabbage",
                                          "masamba=vegetables", "nkhwani=pumpkin leaves", "bonongwe=amaranth", "therere=okra",
                                          "dzungu=pumpkin", "anyezi=onions", "kabichi=cabbage", "mpiru=mustard leaves"]),
    ("FOOD", "root or tuber", "both", False, ["cassava", "potatoes", "sweet potatoes", "chinangwa=cassava", "mbatata=potatoes"]),
    ("FOOD", "fruit", "both", False, ["banana", "mango", "oranges", "avocado", "lemon", "papaya", "masau",
                                      "nthochi=banana", "malalanje=oranges", "mapeyala=avocado", "mandimu=lemon"]),
    ("FOOD", "fat, sugar or salt", "both", False, ["cooking oil", "oil", "sugar", "salt", "honey", "mafuta=oil", "shuga=sugar",
                                                   "mchere=salt", "uchi=honey"]),
    ("FOOD", "drink", "both", False, ["water", "tea", "soft drink", "soda", "alcohol", "beer", "thobwa", "maheu", "kachasu=alcohol",
                                      "madzi=water"]),
    ("FOOD", "prepared or fortified", "both", False, ["Likuni Phala", "corn soya blend", "mandasi", "zitumbuwa", "chips"]),
    ("FOOD", "prepared or fortified", "both", True, ["CSB=corn soya blend"]),
    ("FOOD", "relish", "ny", False, ["ndiwo=relish"]),
    # ---- conditions
    ("CONDITION", "malnutrition", "en", False, ["severe acute malnutrition", "moderate acute malnutrition", "acute malnutrition",
                                                "malnutrition", "undernutrition", "protein-energy malnutrition", "kwashiorkor",
                                                "marasmus", "marasmic kwashiorkor", "stunting", "stunted=stunting", "wasting",
                                                "wasted=wasting", "underweight", "cachexia", "sarcopenia"]),
    ("CONDITION", "malnutrition", "en", True, ["SAM=severe acute malnutrition", "MAM=moderate acute malnutrition"]),
    ("CONDITION", "malnutrition", "ny", False, ["kuperewera kwa zakudya m'thupi=malnutrition"]),
    ("CONDITION", "overnutrition", "en", False, ["overweight", "obesity", "obese=obesity"]),
    ("CONDITION", "micronutrient", "en", False, ["anaemia", "anemia=anaemia", "iron deficiency", "vitamin A deficiency", "zinc deficiency",
                                                 "iodine deficiency", "folate deficiency", "pellagra", "rickets", "scurvy", "goitre"]),
    ("CONDITION", "micronutrient", "ny", False, ["kuchepa magazi=anaemia"]),
    ("CONDITION", "metabolic", "en", False, ["diabetes", "diabetes mellitus=diabetes", "type 2 diabetes", "type 1 diabetes",
                                             "gestational diabetes", "hypertension", "hypoglycaemia", "hypoglycemia=hypoglycaemia",
                                             "hyperglycaemia", "hyperglycemia=hyperglycaemia", "refeeding syndrome", "dehydration",
                                             "chronic kidney disease", "dysphagia"]),
    ("CONDITION", "metabolic", "en", True, ["CKD=chronic kidney disease"]),
    ("CONDITION", "metabolic", "ny", False, ["matenda a shuga=diabetes", "kuthamanga kwa magazi=hypertension"]),
    ("CONDITION", "infection", "en", False, ["tuberculosis", "malaria", "pneumonia", "sepsis"]),
    ("CONDITION", "infection", "en", True, ["HIV", "TB=tuberculosis"]),
    # ---- symptoms and signs
    ("SYMPTOM", "sign or symptom", "en", False, ["vomiting", "nausea", "diarrhoea", "diarrhea=diarrhoea", "loss of appetite", "poor appetite",
                                                 "anorexia", "weight loss", "fever", "cough", "fatigue", "weakness", "pitting oedema",
                                                 "oedema", "edema=oedema", "swelling", "pallor", "constipation", "difficulty swallowing"]),
    ("SYMPTOM", "sign or symptom", "ny", False, ["kusanza=vomiting", "akusanza=vomiting", "amasanza=vomiting", "wasanza=vomiting",
                                                 "kutsegula m'mimba=diarrhoea", "akutsegula m'mimba=diarrhoea",
                                                 "amatsegula m'mimba=diarrhoea", "watsegula m'mimba=diarrhoea",
                                                 "kuonda=weight loss", "akuonda=weight loss", "amaonda=weight loss",
                                                 "kutentha thupi=fever", "malungo=fever or malaria", "chifuwa=cough",
                                                 "kutopa=fatigue", "akutopa=fatigue", "kutupa=swelling", "akutupa=swelling",
                                                 "amatupa=swelling"]),
    # ---- patient groups
    ("POPULATION", "age or state", "en", False, ["infant", "child", "children=child", "toddler", "newborn", "neonate", "adolescent",
                                                 "under-five", "under five=under-five", "under 5=under-five", "pregnant", "lactating",
                                                 "breastfeeding mother", "elderly"]),
    ("POPULATION", "age or state", "ny", False, ["mwana=child", "ana=child", "mayi woyembekezera=pregnant", "mayi woyamwitsa=lactating"]),
    # ---- interventions
    ("INTERVENTION", "therapeutic food", "en", True, ["RUTF", "RUSF", "F-75", "F-100", "ReSoMal", "ORS=oral rehydration salts"]),
    ("INTERVENTION", "therapeutic food", "en", False, ["Plumpy'Nut=RUTF", "oral rehydration=oral rehydration salts"]),
    ("INTERVENTION", "feeding", "en", False, ["therapeutic feeding", "supplementary feeding", "exclusive breastfeeding", "breastfeeding",
                                              "complementary feeding", "enteral feeding", "nasogastric", "NG tube=nasogastric",
                                              "parenteral nutrition"]),
    ("INTERVENTION", "feeding", "ny", False, ["kuyamwitsa=breastfeeding", "akuyamwitsa=breastfeeding", "amayamwitsa=breastfeeding"]),
    ("INTERVENTION", "feeding", "en", True, ["TPN=parenteral nutrition", "ONS=oral nutritional supplement"]),
    ("INTERVENTION", "supplement or drug", "en", False, ["vitamin A supplementation", "iron supplementation", "iron and folic acid",
                                                         "zinc supplementation", "multivitamin", "deworming", "albendazole",
                                                         "oral nutritional supplement"]),
    ("INTERVENTION", "supplement or drug", "en", True, ["IFA=iron and folic acid"]),
    ("INTERVENTION", "counselling", "en", False, ["nutrition counselling", "nutrition counseling=nutrition counselling",
                                                  "dietary counselling", "dietary advice", "high-protein diet", "low-salt diet",
                                                  "diabetic diet"]),
    # ---- nutrients
    ("NUTRIENT", "nutrient", "en", False, ["protein", "carbohydrate", "fat", "fibre", "fiber=fibre", "energy", "calories=energy", "iron",
                                           "zinc", "vitamin A", "vitamin B12", "vitamin C", "vitamin D", "folate", "folic acid",
                                           "calcium", "iodine", "sodium", "potassium", "magnesium"]),
]


def entries() -> list[dict]:
    out = []
    for label, group, lang, cs, terms in _DEFS:
        for raw in terms:
            term, _, norm = raw.partition("=")
            out.append({"term": term, "norm": norm or term,
                        "label": label, "group": group, "lang": lang, "case_sensitive": cs, "reviewed": False})
    return out


# Words that mark a sentence as English or Chichewa (function words and very common words; used only to guess language).
EN_WORDS = set("""the and of with was were has had have is are for on in to at by from this that she he they his her patient child
mother father admitted presented reports denies history after before during daily twice weeks days months years old female male
ate drank eating weighed weighs measured given started continue discharged review plan""".split())
NY_WORDS = set("""ndi ndipo koma komanso chifukwa wa la za ya kwa mu pa ku ali anali ana mwana mayi bambo agogo akudya amadya adadya anadya
ndinadya tinadya akudwala anabwera lero dzulo tsiku masiku mlungu mwezi chaka palibe popanda alibe mwina anati ndikudya wakhanda
amayamwa sanadye sakudya amadwala""".split())

NEGATION_BEFORE = ["no", "not", "denies", "denied", "without", "absent", "negative for", "free of", "nil", "no evidence of", "no signs of",
                   "no history of", "never", "palibe", "popanda", "alibe"]
NEGATION_AFTER = ["absent", "ruled out", "not present", "negative", "was not seen"]
UNCERTAIN_BEFORE = ["possible", "probable", "suspected", "suspect", "likely", "query", "?", "r/o", "rule out", "maybe", "mwina", "cannot exclude"]
HISTORICAL_BEFORE = ["history of", "h/o", "previous", "previously", "past", "resolved"]
FAMILY_BEFORE = ["family history", "fh", "mother has", "father has", "mother had", "father had", "sibling"]
SCOPE_BREAKERS = {"but", "however", "although", "except", "koma", "while", "though"}


# ---------------------------------------------------------------- languages and speaker-supplied lists
import csv  # noqa: E402
from pathlib import Path  # noqa: E402

LEXICON_DIR = Path(__file__).parent / "lexicons"
LANGS = {
    "en": {"name": "English", "support": "Terminology lists and patterns."},
    "ny": {"name": "Chichewa", "support": "Terminology lists; only the listed word forms are found."},
    "tum": {"name": "Tumbuka", "support": "A small unreviewed starter list (foods and a few symptoms). Needs clinical terms from Tumbuka speakers (lexicons/tum.csv)."},
    "yao": {"name": "Yao (Chiyao)", "support": "Framework only, no terms yet. Needs terms from Yao speakers (lexicons/yao.csv)."},
}
ENTRY_LANGS = {"en", "ny", "both", "tum", "yao"}       # "both" means English and Chichewa
_BLOCKED = {"MEASUREMENT", "PHI"}


def load_csv_lexicons(directory: Path = LEXICON_DIR) -> tuple[list[dict], dict[str, set[str]], list[str]]:
    """Reads `*.csv` terminology files (columns: term,label,lang, optional norm,group,case_sensitive) and `markers.csv`
    (lang,word: common words that signal a language). Bad rows are skipped and reported, never fatal. Files starting with
    `_` are templates and are ignored; lines starting with `#` are comments."""
    entries: list[dict] = []
    markers: dict[str, set[str]] = {}
    errors: list[str] = []
    if not directory.is_dir():
        return entries, markers, errors
    for path in sorted(directory.glob("*.csv")):
        if path.name.startswith("_"):
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        for n, row in enumerate(csv.DictReader(lines), start=2):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            where = f"{path.name} row {n}"
            if path.name == "markers.csv":
                lang, word = row.get("lang", ""), row.get("word", "").lower()
                if lang not in LANGS or lang in ("en", "ny") or not word or " " in word:
                    errors.append(f"{where}: needs lang (tum or yao) and one word")
                else:
                    markers.setdefault(lang, set()).add(word)
                continue
            term, label, lang = row.get("term", ""), row.get("label", "").upper(), row.get("lang", "")
            if not term or len(term) > 60:
                errors.append(f"{where}: term is empty or longer than 60 characters")
            elif label not in LABELS or label in _BLOCKED:
                errors.append(f"{where}: label must be one of " + ", ".join(k for k in LABELS if k not in _BLOCKED))
            elif lang not in ENTRY_LANGS:
                errors.append(f"{where}: lang must be one of " + ", ".join(sorted(ENTRY_LANGS)))
            else:
                entries.append({"term": term, "norm": row.get("norm") or term, "label": label, "group": row.get("group") or "listed",
                                "lang": lang, "case_sensitive": row.get("case_sensitive", "").lower() in ("1", "true", "yes"),
                                "reviewed": False, "source": path.name})
    return entries, markers, errors


_CSV_ENTRIES, MARKERS, LOAD_ERRORS = load_csv_lexicons()
_base_entries = entries


def entries() -> list[dict]:                                         # noqa: F811
    """Built-in English and Chichewa terms plus any speaker-supplied terms from lexicons/*.csv."""
    return _base_entries() + [dict(e) for e in _CSV_ENTRIES]
