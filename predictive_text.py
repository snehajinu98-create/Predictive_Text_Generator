import re
from collections import defaultdict, Counter

class ContextAwarePredictiveText:
    def __init__(self, n=3):
        self.n = n  # n-gram size, 3 = trigram (2 words context)
        self.model = defaultdict(Counter)
        self.custom_dictionary = set()

    # -------------------- Text Preprocessing --------------------
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        return words

    # -------------------- Train N-gram Model --------------------
    def train(self, text):
        words = self.preprocess(text)
        for i in range(len(words) - self.n + 1):
            context = tuple(words[i:i+self.n-1])
            next_word = words[i+self.n-1]
            self.model[context][next_word] += 1
        print("✅ Model trained with context awareness!")

    # -------------------- Add Custom Words --------------------
    def add_custom_word(self, word):
        self.custom_dictionary.add(word.lower())
        print(f"✅ Custom word '{word}' added!")

    # -------------------- Predict Next Words --------------------
    def predict(self, sentence, top_n=5):
        words = self.preprocess(sentence)

        if len(words) < self.n - 1:
            return ["⚠️ Please type more words for context-aware prediction"]

        context = tuple(words[-(self.n-1):])
        predictions = []

        # Predict from N-gram model
        if context in self.model:
            predictions = [w for w, _ in self.model[context].most_common(top_n)]

        # Add matches from custom dictionary
        last_word = words[-1]
        custom_matches = [w for w in self.custom_dictionary if w.startswith(last_word)]
        predictions.extend(custom_matches)

        # Remove duplicates
        return list(dict.fromkeys(predictions))

    # -------------------- Continuous Learning --------------------
    def learn_from_sentence(self, sentence):
        self.train(sentence)
        print("🤖 Model updated with new sentence!")

# -------------------- MAIN PROGRAM --------------------
if __name__ == "__main__":
    ptg = ContextAwarePredictiveText(n=3)

    # Sample Training Dataset
    sample_text = """
    i love python programming
    python is easy to learn
    python is used in machine learning
    machine learning is part of artificial intelligence
    artificial intelligence is the future
    predictive text helps people type faster
    """

    ptg.train(sample_text)

    while True:
        print("\n==============================")
        print(" CONTEXT-AWARE PREDICTIVE TEXT ")
        print("==============================")
        print("1. Predict next word")
        print("2. Add custom word")
        print("3. Train model with new sentence")
        print("4. Exit")

        choice = input("Enter choice (1-4): ")

        if choice == "1":
            sentence = input("Type your sentence: ")
            predictions = ptg.predict(sentence)
            print("🔮 Predicted words:", predictions)

        elif choice == "2":
            word = input("Enter custom word: ")
            ptg.add_custom_word(word)

        elif choice == "3":
            new_sentence = input("Enter sentence to train model: ")
            ptg.learn_from_sentence(new_sentence)

        elif choice == "4":
            print("👋 Exiting program...")
            break

        else:
            print("❌ Invalid choice! Try again.")