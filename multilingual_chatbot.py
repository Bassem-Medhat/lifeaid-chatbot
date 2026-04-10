from chatbot_engine import FirstAidChatbot
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException


class MultilingualFirstAidChatbot:
    def __init__(self, data_file='processed_data.json'):
        """
        Initialize multilingual chatbot
        """
        print("Initializing Multilingual First Aid Chatbot...")

        # Initialize the base English chatbot
        self.chatbot = FirstAidChatbot(data_file)

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
            'zh-cn': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'hi': 'Hindi',
            'tr': 'Turkish'
        }

        print("Multilingual chatbot ready")
        print(f"Supported languages: {len(self.language_names)}")

    def detect_language(self, text):
        """
        Detect the language of input text

        Args:
            text: Input text

        Returns:
            str: Language code (e.g., 'ar', 'fr', 'en')
        """
        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            # Default to English if detection fails
            return 'en'

    def translate_to_english(self, text, source_lang):
        """
        Translate text to English

        Args:
            text: Text to translate
            source_lang: Source language code

        Returns:
            str: Translated text in English
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

        Args:
            text: English text
            target_lang: Target language code

        Returns:
            str: Translated text
        """
        if target_lang == 'en':
            return text

        try:
            translator = GoogleTranslator(source='en', target=target_lang)
            translated = translator.translate(text)
            return translated
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def get_response(self, user_question):
        """
        Get response in the same language as the question

        Args:
            user_question: Question in any language

        Returns:
            str: Answer in the same language
        """
        # Detect language
        detected_lang = self.detect_language(user_question)
        lang_name = self.language_names.get(detected_lang, detected_lang)

        print(f"Detected language: {lang_name} ({detected_lang})")

        # Translate question to English if needed
        if detected_lang != 'en':
            english_question = self.translate_to_english(user_question, detected_lang)
            print(f"Translated to English: {english_question}")
        else:
            english_question = user_question

        # Get answer from base chatbot in English
        english_answer = self.chatbot.get_response(english_question)

        # Translate answer back to original language if needed
        if detected_lang != 'en':
            translated_answer = self.translate_from_english(english_answer, detected_lang)
            return translated_answer
        else:
            return english_answer

    def chat(self):
        """
        Interactive multilingual chat mode
        """
        print("\n" + "=" * 80)
        print("Multilingual First Aid Chatbot - Interactive Mode")
        print("=" * 80)
        print("Ask me anything in ANY language!")
        print("Supported languages: English, Arabic, French, Spanish, and more")
        print("Type 'quit', 'exit', or 'bye' in any language to stop.\n")

        while True:
            user_input = input("You: ").strip()

            # Check for quit commands in multiple languages
            quit_commands = ['quit', 'exit', 'bye', 'stop', 'خروج', 'توقف', 'sortir', 'salir']
            if user_input.lower() in quit_commands:
                print("Stay safe! Goodbye!")
                break

            if not user_input:
                continue

            try:
                response = self.get_response(user_input)
                print(f"\nBot: {response}\n")
                print("-" * 80 + "\n")
            except Exception as e:
                print(f"Error processing your question: {e}")
                print("Please try again.\n")


# Test the multilingual chatbot
if __name__ == "__main__":
    print("=" * 80)
    print("Starting Multilingual First Aid Chatbot Test")
    print("=" * 80 + "\n")

    try:
        # Initialize multilingual chatbot
        bot = MultilingualFirstAidChatbot()

        # Test with questions in different languages
        print("\n" + "=" * 80)
        print("Testing with multiple languages:")
        print("=" * 80 + "\n")

        test_questions = [
            ("What to do for bleeding?", "English"),
            ("ماذا أفعل في حالة النزيف؟", "Arabic"),
            ("Que faire en cas de saignement?", "French"),
            ("Qué hacer en caso de sangrado?", "Spanish"),
        ]

        for question, lang in test_questions:
            print(f"Testing {lang} question:")
            print(f"Input: {question}")
            response = bot.get_response(question)
            print(f"Response: {response[:200]}...")
            print("-" * 80 + "\n")

        # Start interactive mode
        print("\nStarting interactive multilingual chat mode...")
        bot.chat()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()