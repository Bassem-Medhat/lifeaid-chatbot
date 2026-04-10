import re
from interactive_chatbot import InteractiveFirstAidChatbot
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException, DetectorFactory

# Make language detection deterministic across all calls and reruns
DetectorFactory.seed = 0

# Per-language signature words for Latin-script language detection.
# These are words that are highly distinctive to each language and very
# unlikely to appear in casual English text.  English is NOT in this dict
# because English is the default — if no foreign signatures are found the
# text is treated as English.  That is the key design choice that prevents
# langdetect's Dutch/Afrikaans mis-classifications from ever affecting English.
#
# Non-Latin-script languages (Arabic, Chinese, Hebrew, Russian, Hindi, etc.)
# are NOT listed here — they are handled by langdetect which is highly
# reliable for non-Latin scripts.
_LANG_SIGNATURES = {
    'fr': {
        'je', 'vous', 'nous', 'une', 'les', 'des', 'bonjour', 'merci',
        'oui', 'salut', 'aussi', 'mais', 'très', 'avec', 'dans', 'pour',
        'sur', 'par', 'qui', 'que', 'quoi', 'quand', 'comment', 'pourquoi',
        'avoir', 'être', 'faire', 'savoir', 'pouvoir', 'aller',
        'douleur', 'brûlure', 'blessure', 'urgence', 'sang', 'saignement',
    },
    'es': {
        'yo', 'hola', 'gracias', 'está', 'estoy', 'están', 'hay',
        'tiene', 'tengo', 'tenemos', 'los', 'las', 'del', 'con',
        'también', 'cómo', 'qué', 'dónde', 'cuándo', 'por', 'favor',
        'muy', 'más', 'cuando', 'porque', 'pero', 'sin',
        'dolor', 'quemadura', 'sangrado', 'emergencia', 'ayuda',
    },
    'de': {
        'ich', 'nicht', 'kein', 'keine', 'auch', 'noch', 'schon',
        'ein', 'eine', 'dem', 'den', 'des',
        'mit', 'bei', 'von', 'aus', 'nach', 'über', 'unter',
        'und', 'oder', 'aber', 'wenn', 'weil', 'dass',
        'ist', 'sind', 'war', 'haben', 'wird', 'werden',
        'hilfe', 'schmerzen', 'blutung', 'verbrennung', 'notfall',
    },
    'it': {
        'sono', 'siamo', 'stai', 'della', 'dello', 'degli', 'delle',
        'nel', 'nella', 'nei', 'nelle',
        'grazie', 'ciao', 'prego',
        'dolore', 'bruciatura', 'sanguinamento', 'emergenza', 'aiuto',
    },
    'pt': {
        'você', 'nós', 'eles', 'elas', 'vocês',
        'está', 'estou', 'são', 'tem', 'temos',
        'obrigado', 'obrigada', 'olá', 'não',
        'dor', 'queimadura', 'sangramento', 'emergência', 'ajuda',
    },
    'nl': {
        'ik', 'jij', 'wij', 'jullie',
        'ben', 'bent', 'zijn', 'heb', 'heeft', 'hebben',
        'niet', 'geen', 'naar', 'over', 'door',
        'maar', 'want', 'dus', 'toch',
        'wat', 'wie', 'hoe', 'wanneer', 'waar', 'waarom',
        'pijn', 'brand', 'bloeding', 'nood',
    },
    'tr': {
        'bir', 'için', 'ile', 'gibi', 'ama', 'veya', 'değil', 'çok',
        'merhaba', 'teşekkür', 'evet', 'hayır',
        'ağrı', 'yardım', 'acil',
    },
    'sv': {
        'jag', 'han', 'hon', 'vi', 'ni',
        'är', 'var', 'har', 'hade', 'ska', 'skulle',
        'inte', 'också', 'från', 'till', 'på',
        'och', 'eller', 'men', 'när',
        'hej', 'tack', 'smärta', 'hjälp',
    },
    'pl': {
        'jest', 'są', 'mam', 'mamy', 'jestem',
        'nie', 'też', 'już', 'tylko',
        'przez', 'przy', 'lub', 'że', 'bo',
        'ból', 'pomoc',
    },
}

# Single words / short phrases that signal vague unlocated pain.
# Checked against the English (post-translation) input.
_VAGUE_PAIN_TRIGGERS = {
    'hurts', 'hurt', 'pain', 'painful', 'ache', 'aching', 'sore', 'aches',
}
_VAGUE_PAIN_PHRASES = [
    'it hurts', 'i feel pain', 'something hurts', 'i have pain',
    'in pain', 'feeling pain', 'hurts a lot', 'lot of pain',
    'so much pain', 'really hurts', 'hurts bad', 'hurts badly',
    'i am in pain', "i'm in pain", 'feel pain',
]

# If ANY of these appear alongside a pain word, the input is specific
# enough to route normally — don't return the clarifying question.
_PAIN_CONTEXT_WORDS = {
    # body parts
    'chest', 'stomach', 'head', 'arm', 'leg', 'back', 'neck', 'shoulder',
    'knee', 'ankle', 'hand', 'foot', 'finger', 'eye', 'ear', 'throat',
    'hip', 'wrist', 'elbow', 'toe', 'jaw', 'teeth', 'tooth', 'nose',
    'belly', 'abdomen', 'ribs', 'spine', 'side', 'groin', 'pelvis',
    # emergency causes
    'cut', 'burn', 'burned', 'bleeding', 'blood', 'broke', 'broken',
    'fell', 'fall', 'fallen', 'hit', 'bite', 'bitten', 'sting', 'stung',
    'choke', 'choking', 'swallowed', 'overdose', 'poison', 'allergic',
    'seizure', 'breath', 'breathe', 'breathing', 'fracture', 'fractured',
    # descriptors that give enough clinical context
    'sharp', 'stabbing', 'crushing', 'throbbing', 'burning',
    'sudden', 'severe', 'intense', 'radiating', 'pressure',
}

_VAGUE_PAIN_RESPONSE = (
    "I'm sorry to hear you're in pain. Can you tell me more so I can help?\n\n"
    "- Where does it hurt? (chest, stomach, head, arm, leg…)\n"
    "- Did something happen? (fall, cut, burn, hit…)\n"
    "- How severe is the pain on a scale of 1–10?"
)


class MultilingualInteractiveFirstAidChatbot:
    def __init__(self, data_file='processed_data.json'):
        """
        Initialize multilingual interactive chatbot
        Combines translation with interactive follow-ups
        """
        print("Initializing Multilingual Interactive First Aid Chatbot...")

        # Initialize the base English interactive chatbot
        self.chatbot = InteractiveFirstAidChatbot(data_file)

        # Language names for display
        self.language_names = {
            'en': 'English',
            'ar': 'Arabic',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh-cn': 'Chinese (Simplified)',
            'zh-tw': 'Chinese (Traditional)',
            'ja': 'Japanese',
            'ko': 'Korean',
            'hi': 'Hindi',
            'tr': 'Turkish',
            'nl': 'Dutch',
            'pl': 'Polish',
            'sv': 'Swedish',
            'he': 'Hebrew'
        }

        # Track user's language
        self.user_language = 'en'

        # Track which follow-up questions have already been asked this session
        # so the same question is never repeated even if state is reset
        self._asked_followup_questions = set()

        print("Multilingual interactive chatbot ready")
        print(f"Supported languages: {len(self.language_names)}")

    def detect_language(self, text):
        """Detect the language of text using a two-path approach.

        Path A — Non-Latin script (Arabic, Chinese, Hebrew, Russian, Hindi…):
            langdetect is highly reliable when the script itself is distinctive.

        Path B — Latin-script text (French, Spanish, German, English…):
            Check the message against per-language signature words from
            _LANG_SIGNATURES.  If at least one signature word is found, return
            that language.  If none are found, return 'en' (English).

            This is the key change: English is the *default* for Latin-script
            text.  We never try to *detect* English — we only detect non-English.
            This completely eliminates the Dutch/Afrikaans mis-classification that
            langdetect produces for short English phrases like "my dog bit me".
        """
        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            return self.user_language

        # Determine script: ord < 0x250 covers Basic Latin + Latin Extended A/B
        latin_count = sum(1 for c in alpha_chars if ord(c) < 0x250)
        latin_ratio = latin_count / len(alpha_chars)

        # ── Path A: non-Latin script ──────────────────────────────────────
        if latin_ratio < 0.5:
            try:
                detected = detect(text)
                if detected == 'fa':   # Persian → treat as Arabic
                    detected = 'ar'
                print(f"Non-Latin detection: {detected}")
                return detected
            except Exception as e:
                print(f"Language detection error: {e}")
                return self.user_language

        # ── Path B: Latin-script — signature-word lookup ──────────────────
        clean_words = {w.strip(".,!?;:'\"()[]") for w in text.lower().split()}
        clean_words.discard('')

        best_lang = None
        best_count = 0
        for lang, signatures in _LANG_SIGNATURES.items():
            count = sum(1 for w in clean_words if w in signatures)
            if count > best_count:
                best_count = count
                best_lang = lang

        if best_lang and best_count >= 1:
            print(f"Signature detection: {best_lang} ({best_count} match(es))")
            return best_lang

        # No foreign signatures found → English
        print("No foreign signatures — defaulting to English")
        return 'en'

    def translate_to_english(self, text, source_lang):
        """
        Translate text to English
        """
        if source_lang == 'en':
            return text

        try:
            translator = GoogleTranslator(source=source_lang, target='en')
            translated = translator.translate(text)
            return translated
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def translate_from_english(self, text, target_lang):
        """
        Translate English text to target language
        Optimized: batch translation instead of line-by-line
        """
        if target_lang == 'en':
            return text

        try:
            translator = GoogleTranslator(source='en', target=target_lang)

            # Translate the entire text at once (MUCH faster)
            translated_text = translator.translate(text)

            return translated_text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def reset_conversation(self):
        """
        Reset conversation state and language
        """
        self.user_language = 'en'
        self._asked_followup_questions = set()
        self.chatbot.conversation_state = {
            'current_emergency': None,
            'current_followup_index': 0,
            'waiting_for_followup': False
        }

    def get_response(self, user_input):
        """
        Get response in the same language as the input.

        Translate-first flow:
          1. Detect language from the user's message
          2. Translate the message to English
          3. Run all detection and chatbot logic on English text
          4. Translate the English response back to the user's language

        This means every language is handled correctly automatically —
        no per-language keyword lists are needed anywhere.
        """
        if not user_input or not user_input.strip():
            msg = "Please describe your emergency or ask a first aid question."
            return self.translate_from_english(msg, self.user_language) if self.user_language != 'en' else msg

        user_lower = user_input.lower().strip()

        # ── Step 1: Detect language ───────────────────────────────────────
        detected_lang = self.detect_language(user_input)

        if detected_lang != self.user_language:
            self.user_language = detected_lang
            lang_name = self.language_names.get(detected_lang, detected_lang)
            print(f"Language: {lang_name} ({detected_lang})")

        # ── Step 2: Translate to English ──────────────────────────────────
        if detected_lang != 'en':
            english_input = self.translate_to_english(user_input, detected_lang)
            print(f"Translated to English: {english_input}")
        else:
            english_input = user_input

        english_lower = english_input.lower().strip()

        # ── Step 3: New question vs follow-up detection (English only) ────
        # Now that the input is in English, no multilingual keyword lists
        # are needed — the same checks work for every input language.
        in_conversation = self.chatbot.conversation_state.get('waiting_for_followup', False)

        if in_conversation:
            # Use whole-word matching so "show" doesn't match "how",
            # "shoulder" doesn't match "should", etc.
            # Emergency topic words are intentionally excluded: the user's
            # answer to "Is the bleeding heavy?" naturally contains "bleeding",
            # but that does NOT make it a new question.  The base chatbot
            # (interactive_chatbot.py) already detects genuine new emergencies
            # via embedding similarity, so we don't need to duplicate that here.
            _new_q_words = {'what', 'how', 'when', 'where', 'why', 'should'}
            _eng_word_set = set(re.findall(r'\b\w+\b', english_lower))
            is_new_question = (
                '?' in english_input or
                bool(_new_q_words & _eng_word_set) or
                'do i' in english_lower
            )
            if is_new_question:
                print("Detected new question — resetting conversation state")
                self.chatbot.conversation_state = {
                    'current_emergency': None,
                    'current_followup_index': 0,
                    'waiting_for_followup': False,
                }
                self._asked_followup_questions = set()
                in_conversation = False

        # ── Step 4: Vague pain detection ──────────────────────────────────
        if not in_conversation:
            _eng_words = set(english_lower.split())
            _has_pain = (
                any(w in _eng_words for w in _VAGUE_PAIN_TRIGGERS) or
                any(phrase in english_lower for phrase in _VAGUE_PAIN_PHRASES)
            )
            _has_context = any(w in english_lower for w in _PAIN_CONTEXT_WORDS)
            if _has_pain and not _has_context:
                print("Vague pain input detected — returning clarifying question")
                if self.user_language != 'en':
                    return self.translate_from_english(_VAGUE_PAIN_RESPONSE, self.user_language)
                return _VAGUE_PAIN_RESPONSE

        # ── Step 5: Get English response from chatbot ─────────────────────
        english_response = self.chatbot.get_response(english_input)

        # ── Step 5.5: Prevent follow-up question repetition ───────────────
        # Even after the is_new_question fix above, keep this as a safety net:
        # if the base chatbot is about to ask a question already asked this
        # session, suppress the follow-up state so it won't be repeated.
        if self.chatbot.conversation_state.get('waiting_for_followup'):
            _curr_emerg = self.chatbot.conversation_state.get('current_emergency')
            _curr_idx   = self.chatbot.conversation_state.get('current_followup_index', 0)
            if _curr_emerg:
                _fups = _curr_emerg.get('follow_up_qa', [])
                if _curr_idx < len(_fups):
                    _q_text = _fups[_curr_idx].get('question', '')
                    if _q_text:
                        if _q_text in self._asked_followup_questions:
                            print(f"[Dedup] Suppressing already-asked follow-up")
                            self.chatbot.conversation_state['waiting_for_followup'] = False
                        else:
                            self._asked_followup_questions.add(_q_text)

        # ── Step 6: Translate response back to user's language ────────────
        is_conversation_end = any(
            phrase in english_response
            for phrase in ["You're welcome!", "Stay safe! Take care.", "Alright. If you need anything else"]
        )

        if self.user_language != 'en':
            translated_response = self.translate_from_english(english_response, self.user_language)
            if is_conversation_end:
                self.user_language = 'en'
            return translated_response
        return english_response


# Test the multilingual interactive chatbot
if __name__ == "__main__":
    print("=" * 80)
    print("Starting Multilingual Interactive First Aid Chatbot Test")
    print("=" * 80 + "\n")

    try:
        # Initialize multilingual interactive chatbot
        bot = MultilingualInteractiveFirstAidChatbot()

        # Test with questions in different languages
        print("\n" + "=" * 80)
        print("Testing with multiple languages:")
        print("=" * 80 + "\n")

        test_questions = [
            ("What to do if someone is bleeding badly?", "English"),
            ("ماذا أفعل إذا كان شخص ما يختنق؟", "Arabic"),
            ("Que faire en cas de brûlure?", "French"),
        ]

        for question, lang in test_questions:
            print(f"Testing {lang} question:")
            print(f"Input: {question}")
            response = bot.get_response(question)
            print(f"Response: {response[:300]}...")
            print("-" * 80 + "\n")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()