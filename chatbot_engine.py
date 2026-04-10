import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Single- or double-word inputs too vague to meaningfully search.
# When the entire query consists only of these words, ask for specifics.
_VAGUE_INPUTS = {
    'help', 'pain', 'emergency', 'hurts', 'hurt', 'sos', 'urgent',
    'quick', 'fast', 'please', 'ouch', 'ow', 'injured', 'injury',
    'bad', 'sick', 'ill', 'problem', 'trouble', 'issue', 'something',
    'accident', 'need', 'happening', 'wrong',
}

# Maps a canonical emergency phrase to informal/synonym trigger words.
# When any trigger appears in the user's query the canonical phrase is
# appended before encoding, pulling the embedding toward the right entry.
_KEYWORD_EXPANSIONS = {
    'heart attack cardiac arrest': [
        'heart attack', 'cardiac arrest', 'cardiac', 'heart stopped',
        'chest pain', 'chest tightness', 'chest pressure', 'myocardial',
        'heart failure', 'palpitations', 'heart pain',
    ],
    'choking airway blocked cyanosis': [
        'choke', 'choking', 'cant breathe', "can't breathe", 'cannot breathe',
        'airway blocked', 'something stuck throat', 'swallowed wrong way',
        'food stuck', 'throat blocked', 'object in throat',
        # Blue skin / cyanosis — always a breathing or choking emergency
        'turning blue', 'turned blue', 'gone blue', 'going blue',
        'blue lips', 'lips are blue', 'lips turning blue',
        'blue face', 'face is blue', 'face turning blue',
        'skin is blue', 'skin turning blue', 'blue skin',
        'cyanosis', 'looks blue', 'appear blue',
    ],
    'severe bleeding wound': [
        'bleed', 'bleeding', 'blood', 'hemorrhage', 'hemorrhaging',
        'cut', 'wound', 'laceration', 'gash', 'slash', 'stabbed',
        'blood loss', 'blood everywhere', 'deep cut',
    ],
    'burn injury scald': [
        'burn', 'burned', 'burning', 'burnt', 'scald', 'scalded',
        'hot water burn', 'fire burn', 'chemical burn', 'sunburn',
        'blister from heat',
    ],
    'broken bone fracture': [
        'fracture', 'fractured', 'broken bone', 'break bone', 'broke bone',
        'snapped bone', 'cracked bone', 'sprain', 'sprained',
        'bone sticking out', 'bone deformity', 'twisted',
    ],
    'allergic reaction anaphylaxis': [
        'allergy', 'allergic', 'anaphylaxis', 'anaphylactic shock',
        'hives', 'swollen throat', 'throat closing', 'epipen',
        'bee sting', 'wasp sting', 'insect sting', 'nut allergy',
        'rash swelling',
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
    ],
    'seizure convulsion': [
        'seizure', 'convulsion', 'convulsing', 'fitting', 'fits',
        'shaking uncontrollably', 'trembling', 'epilepsy', 'epileptic',
        'twitching', 'body jerking',
    ],
    'stroke': [
        'stroke', 'paralysis', 'face drooping', 'facial droop',
        'slurred speech', 'sudden weakness', 'arm weakness',
        'trouble speaking', 'sudden numbness', 'face numb',
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
}


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


def _expand_query(user_question):
    """Append canonical emergency terms when trigger keywords are detected.

    This gives the sentence-transformer a much better chance of matching
    the right knowledge-base entry when the user uses informal phrasing
    or synonyms (e.g. 'blood everywhere' → appends 'severe bleeding wound').
    """
    lower = user_question.lower()
    additions = []
    for canonical, triggers in _KEYWORD_EXPANSIONS.items():
        if any(t in lower for t in triggers):
            additions.append(canonical)
    if additions:
        return user_question + ' ' + ' '.join(additions)
    return user_question


class FirstAidChatbot:
    def __init__(self, data_file='processed_data.json'):
        """
        Initialize the chatbot with processed data
        """
        print("Initializing First Aid Chatbot...")

        # Load the processed data
        self.data = self.load_data(data_file)

        # Initialize the sentence transformer model
        print("Loading AI model - this may take a moment...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("AI model loaded successfully")

        # Create embeddings for all questions in the dataset
        print("Creating knowledge base...")
        self.questions = [item['question'] for item in self.data]
        self.answers = [item['answer'] for item in self.data]
        self.question_embeddings = self.model.encode(self.questions)
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

    def find_best_match(self, user_question, threshold=0.25, clarification_threshold=0.40):
        """
        Find the best matching answer for user's question.

        Scores are divided into three zones:
          < threshold              → no useful match found
          threshold – clarification_threshold → low confidence; ask for more detail
          >= clarification_threshold           → good match; return the answer

        Args:
            user_question: The question asked by the user
            threshold: Minimum similarity score to consider any match
            clarification_threshold: Minimum score to return an answer directly

        Returns:
            dict: Contains answer, confidence, matched_question, found, low_confidence
        """
        # Expand query with synonym/canonical terms before encoding
        expanded = _expand_query(user_question)
        user_embedding = self.model.encode([expanded])

        # Calculate similarity with all questions in database
        similarities = cosine_similarity(user_embedding, self.question_embeddings)[0]

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

        # Start interactive mode
        print("\nStarting interactive chat mode...")
        bot.chat()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()