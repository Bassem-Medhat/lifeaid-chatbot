import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fuzzy matching for misspellings (optional — falls back gracefully if missing)
try:
    from rapidfuzz import fuzz as _rfuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

# Lemmatization for root-form matching (optional — falls back gracefully if missing)
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    _lemmatizer = WordNetLemmatizer()
    _LEMMATIZE_AVAILABLE = True
except (ImportError, Exception):
    _LEMMATIZE_AVAILABLE = False

_FUZZY_THRESHOLD = 80  # minimum similarity % to accept a fuzzy keyword match

# Single- or double-word inputs too vague to meaningfully search.
# When the entire query consists only of these words, ask for specifics.
_VAGUE_INPUTS = {
    'help', 'pain', 'emergency', 'hurts', 'hurt', 'sos', 'urgent',
    'quick', 'fast', 'please', 'ouch', 'ow', 'injured', 'injury',
    'bad', 'sick', 'ill', 'problem', 'trouble', 'issue', 'something',
    'accident', 'need', 'happening', 'wrong',
}

# Conversational filler words removed when building the core query.
# Only used in _extract_core_query — trigger detection always runs on the
# original text so no emergency keywords are accidentally stripped.
_QUERY_STOPWORDS = frozenset({
    # Personal pronouns
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
    'you', 'your', 'he', 'him', 'his', 'she', 'her',
    'they', 'them', 'their', 'it', 'its',
    # Articles
    'a', 'an', 'the',
    # Prepositions
    'of', 'in', 'on', 'at', 'to', 'for', 'with',
    'from', 'by', 'into', 'onto', 'upon',
    # Auxiliaries
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'will', 'would', 'shall',
    # Question starters
    'what', 'how', 'when', 'where', 'why', 'who', 'which',
    # Connectors / fillers
    'if', 'so', 'then', 'and', 'or', 'but',
    'that', 'this', 'these', 'those',
    'there', 'here', 'just', 'also', 'too',
    'very', 'really', 'quite', 'please',
    # Light verbs that add noise without meaning
    'do', 'does', 'did', 'get', 'got', 'go', 'went', 'going',
    'make', 'made', 'come', 'came', 'see', 'saw',
    'think', 'thought', 'want', 'need',
    # Quantifiers
    'some', 'any', 'all', 'much', 'many', 'lot', 'lots',
    'more', 'most', 'every', 'each',
})

# Maps a canonical emergency phrase to informal/synonym trigger words.
# When any trigger appears in the user's query the canonical phrase is
# appended before encoding, pulling the embedding toward the right entry.
_KEYWORD_EXPANSIONS = {
    'heart attack cardiac arrest': [
        'heart attack', 'cardiac arrest', 'cardiac', 'heart stopped',
        'chest pain', 'chest tightness', 'chest pressure', 'myocardial',
        'heart failure', 'palpitations', 'heart pain',
        'no heartbeat', 'no pulse', 'not responding', 'lifeless',
        'not moving', 'fell down not breathing', 'dropped dead',
        'not waking', 'wont wake',
        # Additional symptom descriptions
        'jaw pain', 'left arm pain', 'pain down arm',
        'pressure in chest', 'tightness in chest', 'squeezing in chest',
        'pain radiating', 'radiating to arm',
        # CPR / unresponsive natural phrases
        'not breathing at all', 'has no pulse', 'collapsed not responding',
        'lifeless on floor',
    ],
    'cpr cardiopulmonary resuscitation': [
        'cpr', 'cardiopulmonary resuscitation', 'perform cpr', 'do cpr',
        'chest compressions', 'rescue breathing', 'resuscitation',
        'how to revive', 'revive someone',
    ],
    'choking airway blocked cyanosis': [
        'choke', 'choking', 'cant breathe', "can't breathe", 'cannot breathe',
        'airway blocked', 'something stuck throat', 'swallowed wrong way',
        'food stuck', 'throat blocked', 'object in throat',
        'something stuck', 'face turning red', 'face is red', 'turning red',
        'cant speak', "can't speak", 'no air', 'gasping',
        # Blue skin / cyanosis — always a breathing or choking emergency
        'turning blue', 'turned blue', 'gone blue', 'going blue',
        'blue lips', 'lips are blue', 'lips turning blue',
        'blue face', 'face is blue', 'face turning blue',
        'skin is blue', 'skin turning blue', 'blue skin',
        'cyanosis', 'looks blue', 'appear blue',
        # Breathing difficulty descriptions
        'difficulty breathing', 'trouble breathing', 'hard to breathe',
        'struggling to breathe', 'labored breathing',
        'gagging', 'keeps gagging',
        # Natural conversational phrases
        'something stuck in', 'stuck in throat', 'blocked airway',
        'turning blue face',
    ],
    'severe bleeding wound': [
        'bleed', 'bleeding', 'blood', 'hemorrhage', 'hemorrhaging',
        'cut', 'wound', 'laceration', 'gash', 'slash', 'stabbed',
        'blood loss', 'blood everywhere', 'deep cut',
        'gushing blood', 'wont stop bleeding', "won't stop bleeding",
        'bleeding heavily', 'blood pouring', 'spurting blood',
        # Natural conversational phrases
        'bleeding badly', 'cut my', 'deep cut on',
        'knife cut', 'glass cut', 'bleeding from my',
        # High-priority first-person bleeding phrases (covers "im bleeding" after normalisation)
        'i am bleeding', "i'm bleeding", 'im bleeding',
    ],
    'burn injury scald': [
        'burn', 'burned', 'burning', 'burnt', 'scald', 'scalded',
        'hot water burn', 'fire burn', 'chemical burn', 'sunburn',
        'blister from heat',
        'boiling water', 'boiling', 'hot water', 'hot liquid',
        'spilled hot', 'spilled boiling', 'hot drink',
        'coffee burn', 'tea burn', 'cooking burn',
        'steam burn', 'hot surface', 'hot plate', 'oven burn', 'iron burn',
        'touched hot', 'contact burn',
        # Symptom-first descriptions
        'blistering', 'blistered', 'skin blistering', 'skin burned',
        'red and burning', 'burning skin', 'skin on fire',
        # Natural conversational phrases
        'hot water on', 'boiling water on', 'burned my', 'burn on my',
        'hot liquid on', 'scalded my', 'stove burned', 'iron burned',
    ],
    'broken bone fracture': [
        'fracture', 'fractured', 'broken bone', 'break bone', 'broke bone',
        'snapped bone', 'cracked bone', 'sprain', 'sprained',
        'bone sticking out', 'bone deformity', 'twisted',
        # Body-part specific broken descriptions
        'broken arm', 'broken leg', 'broken wrist', 'broken ankle', 'broken foot',
        'broke arm', 'broke leg',
        # Auditory cues
        'heard a crack', 'heard a snap', 'heard a pop', 'bone crack',
    ],
    'allergic reaction anaphylaxis': [
        'allergy', 'allergic', 'anaphylaxis', 'anaphylactic shock',
        'hives', 'swollen throat', 'throat closing', 'epipen',
        'bee sting', 'wasp sting', 'insect sting', 'nut allergy',
        'rash swelling',
        "can't swallow", 'cannot swallow', 'hives everywhere',
        'throat swelling', 'lips swelling', 'face swelling',
        'severe reaction', 'allergic emergency',
        # Standalone sting words
        'stung', 'sting on',
        # Swollen word-order variants (symptom word first)
        'swollen face', 'face swollen', 'swollen lips', 'lips swollen',
        'swollen throat', 'throat swollen', 'throat tight',
        # Additional triggers
        'itching all over', 'severe itching', 'peanut allergy',
        'food allergy reaction', 'allergic to food',
    ],
    'poisoning overdose': [
        'poison', 'poisoning', 'poisoned', 'toxic', 'toxin',
        'overdose', 'ingested', 'swallowed medication', 'ate chemicals',
        'drank bleach', 'cleaning product ingested', 'drug overdose',
    ],
    'unconscious unresponsive': [
        'unconscious', 'passed out', 'fainted', 'fainting', 'unresponsive',
        'knocked out', 'not waking up', 'wont wake up', 'collapse',
        'collapsed', 'blackout', 'black out', 'lost consciousness',
        # Limp / not moving / unresponsive descriptions
        'limp', 'went limp', 'body limp', 'gone limp',
        'not moving', 'stopped moving',
        'not responding to', 'not reacting',
        # Natural conversational phrases
        'fell and wont', 'not responding to me', 'eyes rolled back',
    ],
    'seizure convulsion': [
        'seizure', 'convulsion', 'convulsing', 'fitting', 'fits',
        'shaking uncontrollably', 'trembling', 'epilepsy', 'epileptic',
        'twitching', 'body jerking',
        'epileptic fit', 'body shaking', 'shaking all over', 'grand mal',
    ],
    'stroke': [
        'stroke', 'paralysis', 'face drooping', 'facial droop',
        'slurred speech', 'speech slurred', 'sudden weakness', 'arm weakness',
        'trouble speaking', 'sudden numbness', 'face numb',
        'sudden confusion', 'fast test', 'face arm speech time',
        'one side weak', 'cant lift arm', "can't lift arm",
    ],
    'diabetic emergency hypoglycemia': [
        'diabetic', 'diabetes', 'low blood sugar', 'hypoglycemia',
        'hyperglycemia', 'insulin', 'sugar crash', 'diabetic shock',
    ],
    'eye injury': [
        'eye injury', 'eye pain', 'something in eye', 'object in eye',
        'chemical in eye', 'eye bleeding', 'hit in eye', 'eye damage',
    ],
    'drowning near drowning': [
        'drowning', 'drowned', 'near drowning', 'water rescue',
        'pulled from water', 'underwater', 'inhaled water',
    ],
    'electric shock electrocution': [
        'electric shock', 'electrocuted', 'electrocution',
        'lightning strike', 'touched live wire', 'shocked by electricity',
    ],
    'head injury concussion': [
        'head injury', 'concussion', 'head trauma', 'hit head',
        'head wound', 'skull fracture', 'brain injury', 'hit in head',
        'fell on head', 'blow to head',
    ],
    'nosebleed': [
        'nosebleed', 'nose bleed', 'nose bleeding', 'blood from nose',
        'bloody nose',
    ],
    'sprain strain': [
        'sprain', 'sprained', 'strain', 'strained', 'twisted ankle',
        'twisted wrist', 'rolled ankle',
    ],
    'animal bite dog cat': [
        'dog bite', 'dog bit', 'dog bit me', 'a dog bit me',
        'bitten by dog', 'bitten by a dog', 'dog attacked', 'dog attack',
        'cat bite', 'cat bit', 'cat bit me', 'bitten by cat',
        'cat scratch', 'cat scratched',
        'animal bite', 'animal bit', 'bitten by animal', 'animal attacked',
        'animal attack', 'bitten by an animal',
    ],
}


# ─── Priority keyword boosting ───────────────────────────────────────────────

# Maps a critical trigger phrase to the canonical emergency category it belongs
# to (must be a key of _KEYWORD_EXPANSIONS).  When any trigger is found in the
# user query, TF-IDF similarity scores for that emergency category are doubled
# so the right answer always wins over weaker, less-specific matches.
_PRIORITY_KEYWORDS = {
    'choking':           'choking airway blocked cyanosis',
    'not breathing':     'heart attack cardiac arrest',
    'unconscious':       'unconscious unresponsive',
    'no pulse':          'heart attack cardiac arrest',
    'cardiac arrest':    'heart attack cardiac arrest',
    'cpr':               'cpr cardiopulmonary resuscitation',
    'chest compressions':'cpr cardiopulmonary resuscitation',
    'severe bleeding':   'severe bleeding wound',
    'bleeding':          'severe bleeding wound',
    'im bleeding':       'severe bleeding wound',
    'i am bleeding':     'severe bleeding wound',
    "i'm bleeding":      'severe bleeding wound',
    'anaphylaxis':       'allergic reaction anaphylaxis',
    'stroke':            'stroke',
    'seizure':           'seizure convulsion',
    'drowning':          'drowning near drowning',
    'electrocuted':      'electric shock electrocution',
    'overdose':          'poisoning overdose',
    'poisoned':          'poisoning overdose',
    'collapsed':         'unconscious unresponsive',
    'turning blue':      'choking airway blocked cyanosis',
    'heart attack':      'heart attack cardiac arrest',
    'allergic reaction': 'allergic reaction anaphylaxis',
    'broken bone':       'broken bone fracture',
    # Burns — missing from original; 'burn' substring matches burning/burned/burnt/scald
    'burn':              'burn injury scald',
    'scald':             'burn injury scald',
    # Head trauma
    'head injury':       'head injury concussion',
    'concussion':        'head injury concussion',
    # Diabetic emergencies
    'diabetic':          'diabetic emergency hypoglycemia',
    'hypoglycemia':      'diabetic emergency hypoglycemia',
    # Animal / dog / cat bites — must beat spider bite and other bite scenarios
    'dog bite':          'animal bite dog cat',
    'dog bit':           'animal bite dog cat',
    'dog attacked':      'animal bite dog cat',
    'dog attack':        'animal bite dog cat',
    'cat bite':          'animal bite dog cat',
    'cat bit':           'animal bite dog cat',
    'cat scratch':       'animal bite dog cat',
    'animal bite':       'animal bite dog cat',
    'animal bit':        'animal bite dog cat',
    'bitten by dog':     'animal bite dog cat',
    'bitten by cat':     'animal bite dog cat',
    'bitten by animal':  'animal bite dog cat',
    'animal attacked':   'animal bite dog cat',
    'animal attack':     'animal bite dog cat',
}


def _apply_priority_boost(similarities, user_question, vectorizer, question_embeddings):
    """Multiply similarity scores by 2× for entries matching a detected priority keyword.

    When a critical emergency keyword is found in the user message the entries
    most relevant to that emergency get a score boost so they always rank above
    less-specific matches.

    Args:
        similarities: 1-D numpy array of cosine similarity scores (will be copied).
        user_question: Raw user query text.
        vectorizer: The fitted TF-IDF vectorizer.
        question_embeddings: Sparse matrix of all question embeddings.

    Returns:
        numpy array with boosted scores (original array is not modified).
    """
    lower = user_question.lower()
    boosted = set()
    result = similarities.copy()
    for keyword, canonical in _PRIORITY_KEYWORDS.items():
        if keyword in lower and canonical not in boosted:
            canonical_vec = vectorizer.transform([canonical])
            canonical_sims = cosine_similarity(canonical_vec, question_embeddings)[0]
            boost_mask = canonical_sims >= 0.20
            result[boost_mask] *= 2.0
            boosted.add(canonical)
    return result


def _detect_emergency_categories(text):
    """Return the set of canonical emergency category keys detected in *text*.

    Uses the same trigger lists as _KEYWORD_EXPANSIONS so the mapping is
    always consistent.  Used by the interactive chatbot to decide whether
    a user message is about the SAME or a DIFFERENT emergency than the
    one currently being discussed.
    """
    lower = text.lower()
    cats = set()
    for canonical, triggers in _KEYWORD_EXPANSIONS.items():
        if any(t in lower for t in triggers):
            cats.add(canonical)
    return cats


# Words that, when appearing alongside "blue" in a message, indicate cyanosis
# (oxygen deprivation) rather than a tight bandage or bruise.  This pattern
# MUST always route to a choking/breathing emergency — never to bandage advice.
_BLUE_CYANOSIS_CONTEXT = frozenset({
    'person', 'face', 'lips', 'lip', 'skin', 'turning', 'turned',
    'going', 'gone', 'looks', 'appears', 'baby', 'child', 'body',
    'color', 'colour', 'nails', 'fingernails', 'he', 'she',
    'they', 'his', 'her', 'their', 'someone', 'him', 'mouth',
})

# If ANY of these appear in the same message as "blue", it is a bandage/wrap
# issue — NOT cyanosis.  These words suppress the cyanosis override.
_BLUE_BANDAGE_EXCLUSIONS = frozenset({
    'wrap', 'bandage', 'cast', 'splint', 'tourniquet', 'compression',
    'wrapped', 'tight', 'tightly', 'dressing', 'brace', 'strap',
})


# ─── Spell correction ────────────────────────────────────────────────────────

# Medical / first-aid terms that must never be "corrected".
_MEDICAL_TERMS = {
    'aed', 'airway', 'amputation', 'anaphylactic', 'anaphylaxis', 'artery',
    'asthma', 'avulsion', 'bronchitis', 'capillary', 'cardiac', 'concussion',
    'contusion', 'convulsion', 'cpr', 'cyanosis', 'defibrillator', 'diastolic',
    'dislocation', 'epilepsy', 'epileptic', 'epipen', 'fracture', 'heimlich',
    'hemorrhage', 'hemorrhaging', 'hyperglycemia', 'hyperthermia',
    'hypoglycemia', 'hypothermia', 'inhaler', 'insulin', 'laceration',
    'myocardial', 'paralysis', 'pneumonia', 'pulse', 'resuscitation',
    'seizure', 'sprain', 'strain', 'systolic', 'tourniquet', 'triage',
    'unconscious', 'unresponsive', 'vein',
}

# Custom word-level overrides for cases where the general spell checker picks
# a more common but medically wrong word (e.g. "brning"→"bring", "hart"→"hart").
# Keys are lowercase misspellings; values are the intended corrections.
_CUSTOM_CORRECTIONS = {
    'brning':    'burning',
    'buring':    'burning',
    'burining':  'burning',
    'poisend':   'poisoned',
    'poisond':   'poisoned',
    'poisond':   'poisoned',
    'hart':      'heart',   # "hart" is a valid word (deer) — force medical meaning
    'chokng':    'choking',
    'bleding':   'bleeding',
    'unconcious':'unconscious',
    'unconcius': 'unconscious',
    'siezure':   'seizure',
    'seziure':   'seizure',
    'fracure':   'fracture',
    'fractuer':  'fracture',
    'alergic':   'allergic',
    'alergick':  'allergic',
    'dizy':      'dizzy',
    'disy':      'dizzy',
}

try:
    from spellchecker import SpellChecker as _SpellChecker
    _spell = _SpellChecker()
    _spell.word_frequency.load_words(_MEDICAL_TERMS)
    _SPELL_AVAILABLE = True
except ImportError:
    _SPELL_AVAILABLE = False


def _is_latin_script(text):
    """Return True when text is primarily Latin-alphabet (not Arabic/CJK/etc.)."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return True
    # Characters above U+024F are outside the Latin/Latin-Extended blocks.
    non_latin = sum(1 for c in alpha_chars if ord(c) > 0x024F)
    return (non_latin / len(alpha_chars)) < 0.15


def _correct_spelling(text):
    """Return text with obvious misspellings silently fixed.

    Rules:
    - Only runs on Latin-script input (Arabic/CJK etc. passed through unchanged).
    - Skips words of 3 characters or fewer (too risky to auto-correct).
    - Skips words that start with an uppercase letter (likely proper nouns).
    - Skips words already recognised as correct or in the medical whitelist.
    - Preserves original casing of the corrected word.
    """
    if not _SPELL_AVAILABLE or not _is_latin_script(text):
        return text

    # Split into alternating word / non-word tokens to preserve spacing/punctuation.
    tokens = re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)
    out = []
    for token in tokens:
        if not token[0].isalpha():
            out.append(token)
            continue

        # Preserve likely proper nouns (capitalised mid-sentence handling is
        # imperfect, but erring on the side of caution is correct here).
        if token[0].isupper():
            out.append(token)
            continue

        lower = token.lower().strip("'")
        if len(lower) <= 3:
            out.append(token)
            continue

        # Custom overrides take priority — handles cases where the general
        # spell checker would pick a more common but medically wrong word.
        if lower in _CUSTOM_CORRECTIONS:
            out.append(_CUSTOM_CORRECTIONS[lower])
            continue

        # If the word is already known (including our medical whitelist), keep it.
        if _spell.known([lower]):
            out.append(token)
            continue

        best = _spell.correction(lower)
        if best and best != lower:
            out.append(best)
        else:
            out.append(token)

    return ''.join(out)


def _lemmatize_text(text):
    """Reduce each word to its root form using verb-form lemmatization.

    Converts inflected forms so that TF-IDF matching works on shared root
    vocabulary across both user queries and knowledge-base documents:
      "spilled" → "spill",  "burning"/"burned" → "burn",
      "bleeding"/"bled"    → "bleed",  "choking"/"choked" → "choke"

    Applied to BOTH user queries (in find_best_match) and KB documents
    (in _build_doc_text) so they are encoded on identical root forms.
    Falls back to returning the original text when NLTK is unavailable.
    """
    if not _LEMMATIZE_AVAILABLE:
        return text
    tokens = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
    out = []
    for token in tokens:
        if token.isalpha():
            out.append(_lemmatizer.lemmatize(token.lower(), pos='v'))
        else:
            out.append(token)
    return ''.join(out)


def _expand_query(user_question):
    """Append canonical emergency terms when trigger keywords are detected.

    Phase 1 — exact/substring matching: if any trigger phrase from
    _KEYWORD_EXPANSIONS appears in the query, its canonical category is appended.

    Phase 2 — fuzzy matching (requires rapidfuzz): each word in the query that
    is ≥4 characters long is compared against every single-word trigger at ≥80%
    similarity.  This catches common misspellings such as:
      "bleding"  → "bleeding"  → appends "severe bleeding wound"
      "chokng"   → "choking"   → appends "choking airway blocked cyanosis"
      "siezure"  → "seizure"   → appends "seizure convulsion"
    """
    # Normalise informal "im" → "i am" so "im bleeding" is treated the same as
    # "I am bleeding" in all three matching phases below.
    lower = re.sub(r'\bim\b', 'i am', user_question.lower())
    additions = []
    already_matched = set()

    # Phase 1: exact / substring matching (original behaviour)
    for canonical, triggers in _KEYWORD_EXPANSIONS.items():
        if any(t in lower for t in triggers):
            additions.append(canonical)
            already_matched.add(canonical)

    # Phase 2: fuzzy matching for misspelled words
    if _RAPIDFUZZ_AVAILABLE:
        query_words = re.findall(r'\b[a-z]{4,}\b', lower)
        for word in query_words:
            for canonical, triggers in _KEYWORD_EXPANSIONS.items():
                if canonical in already_matched:
                    continue
                for trigger in triggers:
                    # Only fuzzy-compare single-word triggers of meaningful length
                    if ' ' not in trigger and len(trigger) >= 4:
                        if _rfuzz.ratio(word, trigger) >= _FUZZY_THRESHOLD:
                            additions.append(canonical)
                            already_matched.add(canonical)
                            break

    # Phase 3: Word-overlap matching — all words of a multi-word trigger appear
    # in the query regardless of order or intervening words.
    # Handles reordered phrases:
    #   "pressure in my chest"    → {"chest","pressure"} ⊆ query → cardiac
    #   "throat is swelling"      → {"throat","swelling"} ⊆ query → anaphylaxis
    #   "she hit her head"        → {"hit","head"} ⊆ query → head injury
    query_word_set = set(re.findall(r'\b[a-z]+\b', lower))
    for canonical, triggers in _KEYWORD_EXPANSIONS.items():
        if canonical in already_matched:
            continue
        for trigger in triggers:
            trigger_words = set(re.findall(r'\b[a-z]+\b', trigger.lower()))
            if len(trigger_words) >= 2 and trigger_words.issubset(query_word_set):
                additions.append(canonical)
                already_matched.add(canonical)
                break

    if additions:
        return user_question + ' ' + ' '.join(additions)
    return user_question


def _extract_core_query(text):
    """Strip conversational filler, returning only emergency-relevant tokens.

    Converts natural-language descriptions into focused vocabulary that scores
    better in TF-IDF matching.  Used alongside the full query — both are scored
    and the element-wise maximum is taken so neither representation is lost.

    Examples:
      "I spilled boiling water on my hand"   → "spilled boiling water hand"
      "what do I do if someone is choking?"  → "choking"
      "she hit her head on the table"        → "hit head table"
      "there is a lot of blood from my arm"  → "blood arm"
    """
    words = re.findall(r'\b[A-Za-z]+\b', text)
    core = [w for w in words if w.lower() not in _QUERY_STOPWORDS]
    return ' '.join(core) if core else text


def _build_doc_text(item):
    """Return the text used to represent a knowledge-base entry in TF-IDF space.

    Combines ALL text fields so that synonyms, symptom descriptions, and the
    actual first-aid steps vocabulary all contribute to similarity scoring:
      - question        (scenario title)
      - keywords        (synonyms / symptom phrases; may contain newline-joined
                         multi-term strings from the original dataset)
      - answer          (first-aid steps text — stripped of markdown formatting
                         so symbols like ** and ## don't pollute the vocabulary)

    Note: the vectorizer must be initialised with max_features=8000 to prevent
    the longer answer texts from diluting cosine-similarity scores (longer
    documents raise the L2 norm and lower similarity).  The cap keeps only the
    most informative terms, eliminating that length penalty.
    """
    parts = []

    # 1. Scenario title
    q = item.get('question', '')
    if q:
        parts.append(q)

    # 2. Keywords / synonym list
    kw_list = item.get('keywords', [])
    if kw_list:
        kw_text = ' '.join(
            kw.replace('\n', ' ') for kw in kw_list if isinstance(kw, str)
        ).strip()
        if kw_text:
            parts.append(kw_text)

    # 3. First-aid steps / answer text (was previously omitted — this was the bug)
    answer = item.get('answer', '')
    if answer and isinstance(answer, str):
        answer = answer.strip()
        # Strip wrapping quotes occasionally present in the raw dataset
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        # Remove markdown formatting so ** __ ## etc. don't enter the vocabulary
        answer = re.sub(r'\*\*|__|##|#', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        parts.append(answer)

    return _lemmatize_text(' '.join(parts))


class FirstAidChatbot:
    def __init__(self, data_file='processed_data.json'):
        """
        Initialize the chatbot with processed data
        """
        print("Initializing First Aid Chatbot...")

        # Load the processed data
        self.data = self.load_data(data_file)

        # [DEBUG] Confirm how many scenarios loaded and what keywords look like
        print(f"[DEBUG] Total scenarios loaded: {len(self.data)}")
        print(f"[DEBUG] Sample keywords from first scenario: {self.data[0].get('keywords', []) if self.data else []}")

        # Initialize the TF-IDF vectorizer.
        # max_features=8000 keeps only the most informative vocabulary terms.
        # This is required because answer texts are long; without the cap their
        # large L2 norms suppress cosine-similarity scores for short user queries.
        print("Loading AI model - this may take a moment...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), analyzer='word', min_df=1, max_features=8000
        )
        print("AI model loaded successfully")

        # Create TF-IDF matrix — each document combines question + keywords + answer
        # so that synonyms, symptom descriptions, AND first-aid step vocabulary all
        # contribute to similarity scoring.
        print("Creating knowledge base...")
        self.questions = [item['question'] for item in self.data]
        self.answers = [item['answer'] for item in self.data]
        docs = [_build_doc_text(item) for item in self.data]
        self.question_embeddings = self.vectorizer.fit_transform(docs)

        # [DEBUG] Confirm vectorizer was fitted on all scenarios
        print(f"[DEBUG] TF-IDF matrix shape: {self.question_embeddings.shape} "
              f"— fitted on all {len(self.data)} scenarios")
        print(f"Knowledge base ready with {len(self.questions)} first aid scenarios")

    def load_data(self, data_file):
        """
        Load processed data from JSON file
        """
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} question-answer pairs")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []

    def find_best_match(self, user_question, threshold=0.05, clarification_threshold=0.10):
        """
        Find the best matching answer for user's question.

        Scores are divided into three zones:
          < threshold              → no useful match found       (0.05)
          threshold – clarification_threshold → low confidence   (0.05–0.10)
          >= clarification_threshold           → good match       (0.10+)

        Args:
            user_question: The question asked by the user
            threshold: Minimum similarity score to consider any match
            clarification_threshold: Minimum score to return an answer directly

        Returns:
            dict: Contains answer, confidence, matched_question, found, low_confidence
        """
        # --- Path A: full query (original behaviour) -------------------------
        expanded_full = _expand_query(user_question)
        lemmatized_full = _lemmatize_text(expanded_full)
        emb_full = self.vectorizer.transform([lemmatized_full])
        sims_full = cosine_similarity(emb_full, self.question_embeddings)[0]

        # --- Path B: core query (filler stripped, then expanded) -------------
        # Removing pronouns/articles/auxiliaries tightens the L2 norm around
        # the emergency-relevant terms, raising cosine scores for natural
        # language like "I spilled boiling water on my hand".
        core_text = _extract_core_query(user_question)
        expanded_core = _expand_query(core_text)
        lemmatized_core = _lemmatize_text(expanded_core)
        emb_core = self.vectorizer.transform([lemmatized_core])
        sims_core = cosine_similarity(emb_core, self.question_embeddings)[0]

        # Element-wise max: whichever representation scored higher wins
        similarities = np.maximum(sims_full, sims_core)

        # Apply 2× priority boost for critical emergency keywords
        similarities = _apply_priority_boost(
            similarities, user_question, self.vectorizer, self.question_embeddings
        )

        # Find the best match
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]

        print(f"Best match score: {best_score:.3f}")
        print(f"Matched question: {self.questions[best_match_idx]}")

        # No useful match
        if best_score < threshold:
            return {
                'answer': (
                    "I'm sorry, I couldn't find a good match for your question. "
                    "Please try rephrasing or ask about common first aid situations "
                    "like cuts, burns, choking, CPR, fractures, or allergic reactions."
                ),
                'confidence': best_score,
                'matched_question': None,
                'found': False,
                'low_confidence': False,
            }

        # Low confidence — found something but not certain enough
        if best_score < clarification_threshold:
            return {
                'answer': (
                    "I'm not entirely sure what situation you're describing. "
                    "Could you give a bit more detail — for example, mention what happened, "
                    "where on the body, or how severe it seems?\n\n"
                    "I can help with: bleeding, burns, choking, CPR, fractures, "
                    "allergic reactions, seizures, poisoning, and more."
                ),
                'confidence': best_score,
                'matched_question': self.questions[best_match_idx],
                'found': False,
                'low_confidence': True,
            }

        return {
            'answer': self.answers[best_match_idx],
            'confidence': best_score,
            'matched_question': self.questions[best_match_idx],
            'found': True,
            'low_confidence': False,
        }

    def _get_cyanosis_answer(self):
        """Return the answer for the dedicated blue-skin/cyanosis entry."""
        for item in self.data:
            q = item.get('question', '').lower()
            if 'turning blue' in q and item.get('severity') == 'CRITICAL':
                answer = item.get('answer', '').strip()
                if answer.startswith('"') and answer.endswith('"'):
                    answer = answer[1:-1]
                return answer
        # Fallback: find any CRITICAL choking entry
        for item in self.data:
            q = item.get('question', '').lower()
            if "can't breathe" in q and 'choking' in q and item.get('severity') == 'CRITICAL':
                answer = item.get('answer', '').strip()
                if answer.startswith('"') and answer.endswith('"'):
                    answer = answer[1:-1]
                return answer
        return None

    def get_response(self, user_question):
        """
        Get response for user's question

        Args:
            user_question: The question asked by the user

        Returns:
            str: The answer to the question
        """
        if not user_question or user_question.strip() == "":
            return "Please ask me a first aid or emergency question."

        user_question = _correct_spelling(user_question)
        lower_q = user_question.lower()
        words = lower_q.strip().split()
        words_set = set(words)

        # ── Priority safety override: blue skin/lips/face = oxygen emergency ──
        # This check runs before ANY embedding work.  "blue" next to body-part or
        # state words always means cyanosis (oxygen deprivation), never a tight
        # bandage.  A wrong answer here could cost a life.
        # Exception: if bandage/wrap words are present, it's a circulation question.
        if ('blue' in words_set
                and (_BLUE_CYANOSIS_CONTEXT & words_set)
                and not (_BLUE_BANDAGE_EXCLUSIONS & words_set)):
            print("Blue-skin/cyanosis override triggered")
            override = self._get_cyanosis_answer()
            if override:
                return "🚨 CRITICAL EMERGENCY - IMMEDIATE ACTION NEEDED\n\n" + override

        # Reject vague single/double-word queries before doing any embedding work
        if len(words) <= 2 and all(w in _VAGUE_INPUTS for w in words):
            return (
                "I'm here to help! Please describe what happened specifically — "
                "for example: 'someone is choking', 'deep cut on arm', "
                "'possible broken leg', 'person collapsed and not breathing'. "
                "What is the emergency?"
            )

        result = self.find_best_match(user_question)

        if result['found']:
            answer = result['answer']
            answer = answer.strip()
            if answer.startswith('"') and answer.endswith('"'):
                answer = answer[1:-1]
            return answer
        elif result['low_confidence']:
            return result['answer']
        else:
            return "I'm sorry, I couldn't find information about that. Could you rephrase your question or ask about common emergencies like bleeding, burns, choking, CPR, fractures, or poisoning?"

    def chat(self):
        """
        Interactive chat mode for testing
        """
        print("\n" + "=" * 80)
        print("First Aid Chatbot - Interactive Mode")
        print("=" * 80)
        print("Ask me anything about first aid and emergencies!")
        print("Type 'quit', 'exit', or 'bye' to stop.\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye', 'stop']:
                print("Stay safe! Goodbye!")
                break

            if not user_input:
                continue

            response = self.get_response(user_input)
            print(f"\nBot: {response}\n")
            print("-" * 80 + "\n")


# Test the chatbot
if __name__ == "__main__":
    print("=" * 80)
    print("Starting First Aid Chatbot Engine Test")
    print("=" * 80 + "\n")

    try:
        # Initialize the chatbot
        bot = FirstAidChatbot()

        # Test with sample questions
        print("\n" + "=" * 80)
        print("Testing with sample questions:")
        print("=" * 80 + "\n")

        test_questions = [
            "What should I do if someone is bleeding?",
            "How to help a choking person?",
            "Someone has a burn, what to do?",
        ]

        for question in test_questions:
            print(f"Question: {question}")
            response = bot.get_response(question)
            print(f"Response: {response[:200]}...\n")
            print("-" * 80 + "\n")

        # ── Targeted regression tests for natural-language matching ──────────
        print("=" * 80)
        print("[TEST] Natural-language matching scores")
        print("=" * 80)
        _natural_tests = [
            "I spilled boiling water on my hand",
            "my hand got burned",
            "there is blood everywhere",
            "he wont wake up",
        ]
        for _test_query in _natural_tests:
            _expanded = _expand_query(_test_query)
            _lemmatized = _lemmatize_text(_expanded)
            _emb = bot.vectorizer.transform([_lemmatized])
            _sims = cosine_similarity(_emb, bot.question_embeddings)[0]
            _sims = _apply_priority_boost(_sims, _test_query, bot.vectorizer, bot.question_embeddings)
            _best_idx = int(np.argmax(_sims))
            _best_score = _sims[_best_idx]
            print(f"\n  Input : {_test_query!r}")
            print(f"  Score : {_best_score:.4f}  |  Matched Q: {bot.questions[_best_idx]}")
            _response = bot.get_response(_test_query)
            _label = "FOUND" if _best_score >= 0.10 else ("LOW-CONF" if _best_score >= 0.05 else "NO MATCH")
            print(f"  Result: [{_label}]  Response preview: {_response[:80]}...")
        print("\n" + "=" * 80 + "\n")

        # Start interactive mode
        print("\nStarting interactive chat mode...")
        bot.chat()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()