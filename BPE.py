from collections import defaultdict
import re

class BytePairEncoding:
    def __init__(self, num_merges, verbose=False):
        self.num_merges = num_merges  # Number of merge steps
        self.vocab = {}  # Vocabulary storing current tokens
        self.merges = []  # List of all merge operations
        self.verbose = verbose  # Verbose flag for printing steps

    # Pre-tokenize the input similar to WordPiece
    def pre_tokenize(self, text):
        return re.findall(r'\b\w+\b|[,.]', text.lower())

    # Training the BPE model
    def train(self, corpus):
        # Pre-tokenize the corpus (similar to WordPiece)
        tokenized_corpus = [list(word) for word in self.pre_tokenize(corpus)]

        # Initialize the vocabulary with all unique characters in the corpus
        self.vocab = set()
        for word in tokenized_corpus:
            for char in word:
                self.vocab.add(char)

        # Convert the vocab set to a dictionary with initial tokens
        self.vocab = {char: char for char in self.vocab}

        if self.verbose:
            print(f"Initial Vocabulary: {self.vocab}")
        
        # Perform the merging process num_merges times
        for merge_step in range(1, self.num_merges + 1):
            # Get frequencies of consecutive token pairs
            pairs = self.get_pair_frequencies(tokenized_corpus)
            if not pairs:
                break

            # Find the most frequent pair of tokens
            most_frequent_pair = max(pairs, key=pairs.get)

            # Create a new token by concatenating the most frequent pair
            new_token = ''.join(most_frequent_pair)

            # Update vocabulary and record the merge operation
            self.vocab[new_token] = new_token
            self.merges.append(most_frequent_pair)

            if self.verbose:
                print(f"\nStep {merge_step}:")
                print(f"Most Frequent Pair: {most_frequent_pair}")
                print(f"New Token: {new_token}")

            # Replace every occurrence of the most frequent pair in the tokenized corpus
            tokenized_corpus = self.replace_pairs_in_corpus(tokenized_corpus, most_frequent_pair, new_token)

            if self.verbose:
                print(f"Updated Corpus: {tokenized_corpus}")
                print(f"Updated Vocabulary: {self.vocab}")

    # Calculate frequencies of consecutive token pairs
    def get_pair_frequencies(self, tokenized_corpus):
        pairs = {}
        for tokens in tokenized_corpus:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in pairs:
                    pairs[pair] += 1
                else:
                    pairs[pair] = 1
        return pairs

    # Replace occurrences of a token pair with the new token in the corpus
    def replace_pairs_in_corpus(self, tokenized_corpus, pair, new_token):
        new_corpus = []
        for tokens in tokenized_corpus:
            new_word = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_word.append(new_token)  # Merge the pair into a new token
                    i += 2  # Skip the merged pair
                else:
                    new_word.append(tokens[i])
                    i += 1
            new_corpus.append(new_word)
        return new_corpus

    # Tokenize a single word using the learned BPE merges
    def tokenize(self, word):
        tokens = list(word)  # Initially, treat each character as a token
        for merge in self.merges:
            new_token = ''.join(merge)
            merged_word = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == merge:
                    merged_word.append(new_token)  # Merge the pair into a new token
                    i += 2  # Skip merged tokens
                else:
                    merged_word.append(tokens[i])
                    i += 1
            tokens = merged_word
        return tokens

    # Tokenize a full text based on the learned BPE model
    def tokenize_BPE(self, text):
        words = self.pre_tokenize(text)  # Pre-tokenize the text like in WordPiece
        encoded_words = [self.tokenize(word) for word in words]  # Tokenize each word
        return sum(encoded_words, [])  # Flatten the list of tokenized words

    # Get the current vocabulary
    def get_vocabulary(self):
        return self.vocab
