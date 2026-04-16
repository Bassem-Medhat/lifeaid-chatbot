import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from chatbot_engine import (
    _VAGUE_INPUTS, _expand_query, _extract_core_query,
    _BLUE_CYANOSIS_CONTEXT, _BLUE_BANDAGE_EXCLUSIONS,
    _apply_priority_boost, _detect_emergency_categories,
    _build_doc_text,
)


class InteractiveFirstAidChatbot:
    def __init__(self, data_file='processed_data.json'):

        print("Initializing Smart First Aid Chatbot...")

        self.data = self.load_data(data_file)

        print("Loading AI model...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), analyzer='word', min_df=1, max_features=8000
        )

        print("Creating knowledge base...")
        self.questions = [item['question'] for item in self.data]
        docs = [_build_doc_text(item) for item in self.data]
        self.question_embeddings = self.vectorizer.fit_transform(docs)

        self.conversation_state = {
            'current_emergency': None,
            'current_followup_index': 0,
            'waiting_for_followup': False,
            'last_bot_message': '',
            '_details_emergency': None,  # holds emergency for pending "more details?" prompt
        }

        self.last_matched_emergency = None

        print(f"Smart chatbot ready with {len(self.questions)} scenarios")

    def load_data(self, data_file):
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} entries")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []

    def find_best_match(self, user_question, threshold=0.25):

        if not self.questions:
            return None, 0

        # --- Path A: full query -----------------------------------------------
        expanded_full = _expand_query(user_question)
        emb_full = self.vectorizer.transform([expanded_full])
        sims_full = cosine_similarity(emb_full, self.question_embeddings)[0]

        # --- Path B: core query (filler stripped, then expanded) ---------------
        core_text = _extract_core_query(user_question)
        expanded_core = _expand_query(core_text)
        emb_core = self.vectorizer.transform([expanded_core])
        sims_core = cosine_similarity(emb_core, self.question_embeddings)[0]

        # Element-wise max: whichever representation scored higher wins
        similarities = np.maximum(sims_full, sims_core)

        # Apply 2× priority boost for critical emergency keywords
        similarities = _apply_priority_boost(
            similarities, user_question, self.vectorizer, self.question_embeddings
        )

        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]

        print(f"[InteractiveChatbot] query={user_question!r:.60}  score={best_score:.4f}  threshold={threshold}  matched={self.questions[best_match_idx]!r:.60}")

        if best_score < threshold:
            return None, 0

        return self.data[best_match_idx], best_score

    def is_question_specific(self, user_input):

        user_lower = user_input.lower()

        location_words = ['arm', 'leg', 'head', 'chest', 'hand', 'foot', 'face', 'neck', 'back']
        severity_words = ['severe', 'heavy', 'light', 'minor', 'major', 'deep', 'small', 'large']
        detail_words = ['how much', 'how long', 'what kind', 'what type']

        has_location = any(word in user_lower for word in location_words)
        has_severity = any(word in user_lower for word in severity_words)
        has_detail = any(phrase in user_lower for phrase in detail_words)

        details_count = sum([has_location, has_severity, has_detail])

        return details_count >= 1 or len(user_input.split()) > 8

    def format_answer_smart(self, emergency_data, user_input, confidence):

        self.last_matched_emergency = emergency_data

        severity = emergency_data.get('severity', 'MILD')
        full_answer = emergency_data.get('answer', '')
        follow_up_qa = emergency_data.get('follow_up_qa', [])

        response = ""

        is_specific = self.is_question_specific(user_input)

        # CRITICAL
        if severity == 'CRITICAL':
            response += "🚨 CRITICAL EMERGENCY - IMMEDIATE ACTION NEEDED\n\n"
            response += full_answer + "\n\n"

        # URGENT
        elif severity == 'URGENT':
            response += "⚠️ URGENT - Act Quickly\n\n"
            response += full_answer + "\n\n"

        # MODERATE
        elif severity == 'MODERATE':
            response += full_answer + "\n\n"

        # MILD
        else:
            response += self.format_concise_answer(full_answer)

        # Add follow-up only if available
        first_question = follow_up_qa[0].get('question', '') if follow_up_qa else ''
        if follow_up_qa and first_question:
            self.conversation_state['current_emergency'] = emergency_data
            self.conversation_state['current_followup_index'] = 0
            self.conversation_state['waiting_for_followup'] = True

            response += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
            response += "📋 To give you more specific guidance:\n"
            response += first_question

        return response

    def format_concise_answer(self, full_answer):

        if '1.' in full_answer and '2.' in full_answer:
            return full_answer

        paragraphs = full_answer.split('\n\n')

        if len(paragraphs) > 3:
            return '\n\n'.join(paragraphs[:3]) + "\n\n(Ask if you need more details)"

        return full_answer

    def match_followup_response(self, user_answer, qa_item):
        """
        Match user's answer to appropriate response option.
        Strips 'Conversational Response:' artifact from any bot_response field.
        Uses whole-word matching to avoid false positives.
        """
        user_lower = user_answer.lower().strip()
        responses = qa_item.get('responses', [])

        def get_bot_response(response_item):
            if isinstance(response_item, dict):
                text = response_item.get('bot_response', str(response_item))
            else:
                text = str(response_item)
            # Strip the "Conversational Response:" section — must never reach user
            marker = 'Conversational Response:'
            if marker in text:
                text = text[:text.index(marker)]
            # Strip trailing separator lines
            text = re.sub(r'\s*={3,}\s*$', '', text).strip()
            return text

        def _has_word(word, text):
            return bool(re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE))

        yes_words = ['yes', 'yeah', 'yep', 'sure', 'ok', 'correct', 'right', 'نعم', 'اه', 'ايوه']
        no_words  = ['no', 'nope', 'not', 'none', 'لا', 'لأ']

        if isinstance(responses, list):
            for response_item in responses:
                if not isinstance(response_item, dict):
                    continue
                stored_answer = response_item.get('user_answer', '').lower()

                if any(_has_word(w, user_lower) for w in yes_words):
                    if any(_has_word(w, stored_answer) for w in yes_words + ['can', 'conscious', 'breathing']):
                        return get_bot_response(response_item)
                elif any(_has_word(w, user_lower) for w in no_words):
                    if any(_has_word(w, stored_answer) for w in no_words + ["can't", 'cannot', 'not', 'unconscious']):
                        return get_bot_response(response_item)

                # Keyword fallback: whole-word match on any meaningful user word
                user_words = re.findall(r'\b\w+\b', user_lower)
                if any(_has_word(w, stored_answer) for w in user_words if len(w) > 2):
                    return get_bot_response(response_item)

            if responses:
                return get_bot_response(responses[0])
            return "Understood."

        else:
            if any(_has_word(w, user_lower) for w in yes_words):
                response = responses.get('yes', responses.get('if_yes', responses.get('default', 'Understood.')))
                return get_bot_response(response)
            elif any(_has_word(w, user_lower) for w in no_words):
                response = responses.get('no', responses.get('if_no', responses.get('default', 'Understood.')))
                return get_bot_response(response)
            else:
                for key, response in responses.items():
                    if isinstance(key, str) and (key.lower() in user_lower or user_lower in key.lower()):
                        return get_bot_response(response)
                default = responses.get('default', list(responses.values())[0] if responses else 'Understood.')
                return get_bot_response(default)
    def _get_cyanosis_answer(self):
        """Return the dedicated blue-skin/cyanosis entry from the knowledge base."""
        for item in self.data:
            q = item.get('question', '').lower()
            if 'turning blue' in q and item.get('severity') == 'CRITICAL':
                return item
        # Fallback: first CRITICAL choking entry
        for item in self.data:
            q = item.get('question', '').lower()
            if "can't breathe" in q and 'choking' in q and item.get('severity') == 'CRITICAL':
                return item
        return None

    def get_response(self, user_input):
        """Get chatbot response with follow-up question handling.

        Thin public wrapper that records the reply in conversation_state so the
        smarter follow-up detector can inspect whether the last bot message was a
        question.
        """
        print("\n" + "="*60)
        print(f"[DEBUG] USER INPUT: {user_input!r}")
        print(f"[DEBUG] STATE BEFORE:")
        print(f"  waiting_for_followup : {self.conversation_state['waiting_for_followup']}")
        print(f"  current_followup_idx : {self.conversation_state['current_followup_index']}")
        current_emerg = self.conversation_state.get('current_emergency')
        print(f"  current_emergency    : {current_emerg.get('question','?')!r:.80}" if current_emerg else "  current_emergency    : None")
        print(f"  last_matched_emerg   : {self.last_matched_emergency.get('question','?')!r:.80}" if self.last_matched_emergency else "  last_matched_emerg   : None")
        print(f"  last_bot_message ends: {self.conversation_state.get('last_bot_message','')[-80:]!r}")
        print("="*60)

        response = self._get_response(user_input)
        self.conversation_state['last_bot_message'] = response

        print(f"[DEBUG] STATE AFTER:")
        print(f"  waiting_for_followup : {self.conversation_state['waiting_for_followup']}")
        print(f"  current_followup_idx : {self.conversation_state['current_followup_index']}")
        after_emerg = self.conversation_state.get('current_emergency')
        print(f"  current_emergency    : {after_emerg.get('question','?')!r:.80}" if after_emerg else "  current_emergency    : None")
        print(f"[DEBUG] RESPONSE (first 200 chars): {response[:200]!r}")
        print("="*60 + "\n")

        return response

    def _get_response(self, user_input):
        """Internal implementation of get_response."""
        user_input = user_input.strip()

        if not user_input:
            return "Please describe your emergency or ask a first aid question."

        user_lower = user_input.lower()

        # ── Priority safety override: blue skin/lips/face = oxygen emergency ──
        # Runs before conversation state, vague checks, and embedding.
        # Exception: bandage/wrap words in the message mean it's a circulation
        # question about a tight dressing, not a breathing emergency.
        words_set = set(user_lower.split())
        if ('blue' in words_set
                and (_BLUE_CYANOSIS_CONTEXT & words_set)
                and not (_BLUE_BANDAGE_EXCLUSIONS & words_set)):
            print("Blue-skin/cyanosis override triggered")
            # Reset conversation state — this is a fresh critical emergency
            self.conversation_state = {
                'current_emergency': None,
                'current_followup_index': 0,
                'waiting_for_followup': False,
                'last_bot_message': '',
                '_details_emergency': None,
            }
            override_data = self._get_cyanosis_answer()
            if override_data:
                return self.format_answer_smart(override_data, user_input, 1.0)

        # Reject vague single/double-word queries before any embedding work
        words = user_lower.split()
        if len(words) <= 2 and all(w in _VAGUE_INPUTS for w in words):
            return (
                "I'm here to help! Please describe what happened specifically — "
                "for example: 'someone is choking', 'deep cut on arm', "
                "'possible broken leg', 'person collapsed and not breathing'. "
                "What is the emergency?"
            )

        # Handle common pleasantries
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon']
        if user_lower in greetings:
            return "Hello! I'm here to help with first aid and emergency situations. What emergency are you dealing with?"

        thanks = ['thank', 'thanks', 'thx', 'appreciate']
        goodbyes = ['bye', 'goodbye', 'exit', 'quit']

        # Only treat as thanks if it's clearly a thank you message
        if any(word in user_lower for word in thanks) and len(user_input.split()) <= 5:
            self.conversation_state['current_emergency'] = None
            self.conversation_state['waiting_for_followup'] = False
            self.conversation_state['current_followup_index'] = 0
            self.conversation_state['_details_emergency'] = None
            return "You're welcome! Stay safe. If you need any more help, I'm here."

        # Only treat as goodbye if clearly leaving
        if any(word in user_lower for word in goodbyes) and len(user_input.split()) <= 4:
            self.conversation_state['current_emergency'] = None
            self.conversation_state['waiting_for_followup'] = False
            self.conversation_state['current_followup_index'] = 0
            self.conversation_state['_details_emergency'] = None
            return "Stay safe! Take care."

        # ── Smarter follow-up detection ──────────────────────────────────────
        # Four ordered rules determine whether the current message is a follow-up
        # answer or the start of a new emergency question.
        if self.conversation_state['waiting_for_followup']:
            word_count = len(user_input.split())
            last_bot_msg = self.conversation_state.get('last_bot_message', '')

            # Rule A (highest priority): user mentions a DIFFERENT emergency category
            # → always treat as new question, regardless of message length.
            user_cats = _detect_emergency_categories(user_input)
            current_emergency_entry = self.conversation_state.get('current_emergency')
            if user_cats and current_emergency_entry:
                current_q = current_emergency_entry.get('question', '')
                current_cats = _detect_emergency_categories(current_q)
                if not (user_cats & current_cats):
                    # Completely different emergency → reset and fall through
                    self.conversation_state['current_emergency'] = None
                    self.conversation_state['waiting_for_followup'] = False
                    self.conversation_state['current_followup_index'] = 0

            # Only continue as a follow-up if state is still active after Rule A
            if self.conversation_state['waiting_for_followup']:
                _clear_followup_starters = [
                    'yes', 'no', 'yeah', 'nope', 'sure', 'ok', 'okay', 'right',
                    'correct', 'fine', 'alright', 'yep', 'yup', 'got it',
                    'understood', 'not really', 'kind of', 'sort of', 'i think',
                    'maybe', 'possibly', 'not sure', 'i guess',
                    'نعم', 'لا', 'اه', 'ايوه',
                ]

                # Rule B: short answers (< 5 words) or pure numbers → always follow-up
                _is_short = (
                    word_count < 5
                    or user_lower.strip().rstrip('.!?').isdigit()
                )
                # Rule C: starts with a typical yes/no/qualifier word → follow-up
                _starts_with_followup = any(
                    user_lower.startswith(w) for w in _clear_followup_starters
                )
                # Rule D: last bot message was a question and reply is ≤8 words → likely follow-up
                _last_was_question = last_bot_msg.rstrip().endswith('?')
                _is_clear_followup = (
                    _is_short
                    or _starts_with_followup
                    or (_last_was_question and word_count <= 8)
                )

                if not _is_clear_followup and word_count >= 2:
                    # Score-based fallback for longer messages with no clear signal
                    _candidate, _score = self.find_best_match(user_input, threshold=0.0)
                    _current = self.conversation_state.get('current_emergency')
                    if (_candidate is not None and _score >= 0.40
                            and _current is not _candidate):
                        self.conversation_state['current_emergency'] = None
                        self.conversation_state['waiting_for_followup'] = False
                        self.conversation_state['current_followup_index'] = 0

        # Handle follow-up answers
        if self.conversation_state['waiting_for_followup']:
            current_emergency = self.conversation_state['current_emergency']
            followup_index = self.conversation_state['current_followup_index']
            follow_up_qa = current_emergency.get('follow_up_qa', [])

            if followup_index < len(follow_up_qa):
                current_qa = follow_up_qa[followup_index]

                # Get the appropriate conditional response
                response_given = self.match_followup_response(user_input, current_qa)

                # Move to next follow-up question
                self.conversation_state['current_followup_index'] += 1

                # Check if there are more follow-ups
                if self.conversation_state['current_followup_index'] < len(follow_up_qa):
                    next_qa = follow_up_qa[self.conversation_state['current_followup_index']]
                    return f"{response_given}\n\nNow, {next_qa['question']}"
                else:
                    # No more follow-ups - ask if they want detailed steps.
                    # Bug 1 fix: reset current_emergency and waiting_for_followup together
                    # so there is never a zombie current_emergency.  The emergency is
                    # saved to _details_emergency so the "more details?" prompt still works.
                    self.conversation_state['_details_emergency'] = self.conversation_state['current_emergency']
                    self.conversation_state['current_emergency'] = None
                    self.conversation_state['waiting_for_followup'] = False
                    self.conversation_state['current_followup_index'] = 0
                    return f"{response_given}\n\nWould you like more detailed step-by-step instructions?"

        # Check if asking for more details after follow-ups completed.
        # Bug 3 fix: only fires when the ENTIRE message is a bare yes/no word
        # so messages like "yes but what about burns?" fall through to matching.
        # Uses _details_emergency (Bug 1 fix) so current_emergency is never
        # left set when waiting_for_followup is False.
        if self.conversation_state.get('_details_emergency') and not self.conversation_state['waiting_for_followup']:
            yes_words = {'yes', 'yeah', 'yep', 'sure', 'please', 'ok', 'okay'}
            no_words  = {'no', 'nope', 'not really', 'im good', "i'm good"}

            _bare = user_lower.strip().rstrip('.!?')
            if _bare in yes_words:
                pending = self.conversation_state['_details_emergency']
                self.conversation_state['_details_emergency'] = None
                answer = pending.get('answer', '')
                answer = answer.replace('CRITICAL EMERGENCY - IMMEDIATE ACTION NEEDED\n\n', '')
                answer = answer.replace('URGENT - Act Quickly\n\n', '')
                return "Here are the detailed steps:\n\n" + answer

            elif _bare in no_words:
                self.conversation_state['_details_emergency'] = None
                return "Alright. If you need anything else, just ask. Stay safe!"

        # Clear pending-details state for any new question that contains '?'
        # so it is never intercepted on a subsequent bare-yes answer.
        if '?' in user_input and len(user_input.split()) > 3:
            if self.conversation_state.get('_details_emergency'):
                self.conversation_state['_details_emergency'] = None

        # New query — TF-IDF matching runs here for every message that was not
        # handled as a follow-up answer or a bare yes/no above.
        emergency_data, confidence = self.find_best_match(user_input)

        if emergency_data is None:
            return "I'm not sure about that. Could you rephrase? Try asking about bleeding, choking, burns, CPR, fractures, or other common emergencies."

        if confidence < 0.40:
            return (
                "I found something related, but I'm not confident it matches your situation. "
                "Could you describe the emergency in more detail?\n\n"
                "For example: 'severe bleeding from arm', 'child choking on food', "
                "'burn from hot water', 'person collapsed and unresponsive'."
            )

        response = self.format_answer_smart(emergency_data, user_input, confidence)
        return response

    def chat(self):

        print("\nSmart First Aid Chatbot\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Stay safe! Goodbye!")
                break

            if not user_input:
                continue

            response = self.get_response(user_input)
            print(f"\nBot: {response}\n")


if __name__ == "__main__":

    try:
        bot = InteractiveFirstAidChatbot()
        bot.chat()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
