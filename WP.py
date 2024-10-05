from collections import defaultdict
import re

class WordPiece:
    def __init__(self, corpus: str, k: int = 40, verbose: bool = False):
        # iniciamos el corpus
        self.corpus = corpus
        # pre-tokenizamos el corpus
        self.pre_tokenized_corpus = self.pre_tokenize(corpus)
        # creamos el vocabulario inicial
        self.vocab = self.start_vocab()
        # Creamos un diccionario de divisiones por palabra
        self.word_pieces = self.start_word_pieces()

        # Numero de mezclas
        self.k = k
        # Calculamos las frecuencias de cada palabra
        self.pre_word_freq = self.get_pre_word_freq()

        # Modo verbose para imprimir los pasos
        self.verbose = verbose
    
    def get_pre_word_freq(self):
        word_freq = {}
        for word in self.pre_tokenized_corpus:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        return word_freq

    def start_word_pieces(self):
        word_pieces = {
            # 2. Para cada palabra en el corpus pre-tokenizado
            palabra: [ c if i == 0 else f'##{c}' for i, c in enumerate(palabra) ]
            # 1. Para cada palabra en el corpus pre-tokenizado
            for palabra in self.pre_tokenized_corpus
        }
        return word_pieces
    
    def start_vocab(self):
        vocab = []
        # Recorremos cada palabra del corpus (pre-tokenizado)
        for word in self.pre_tokenized_corpus:
            # La letra inicial se agrega al vocabulario
            if word[0] not in vocab:
                vocab.append(word[0])
            # Todas las demás letras se les añade '##' al vocabulario
            for letter in word[1:]:
                if '##' + letter not in vocab:
                    vocab.append('##' + letter)
        
        # Agregamos los tokens especiales
        tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        for token in tokens:
            vocab.append(token)

        return vocab
             
    def pre_tokenize(self, text):
        return re.findall(r'\b\w+\b|[,.]', text.lower())
    
    # Función para calcular los puntajes de los pares
    def compute_pair_scores(self):
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        
        # Calcular frecuencias de letras y pares
        for word, freq in self.pre_word_freq.items():
            split = self.word_pieces[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        # Calcular puntajes de los pares
        scores = {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
        return scores
    
    # Función para hacer la mezcla del par seleccionado
    def merge_pair(self, a, b, splits):
        # Recorrer cada palabra en el corpus
        for word in self.pre_word_freq:
            # Obtener la división de la palabra
            split = splits[word]
            # Si la palabra tiene solo una división, continuar
            if len(split) == 1:
                continue
            # Recorrer cada par de divisiones
            i = 0
            while i < len(split) - 1:
                # Si el par actual es igual al par seleccionado
                if split[i] == a and split[i + 1] == b:
                    # Realizar la mezcla
                    merge = a + b[2:] if b.startswith("##") else a + b
                    # Actualizar la división de la palabra
                    split = split[:i] + [merge] + split[i + 2:]                    
                else:
                    # Incrementar el índice
                    i += 1
            # Actualizar la división de la palabra
            splits[word] = split
        
        return splits

    # Paso de entrenamiento: seleccionar el mejor par y actualizar las divisiones de palabra
    def train(self):
        # Iterar sobre el número de mezclas
        for i in range(self.k):
            if self.verbose:
                print(f"\n=== Iteración {i + 1} ===")
                print("Vocabulario actual:", self.vocab)
                print("Divisiones actuales (word_pieces):", self.word_pieces)

            # Calcular los puntajes de los pares
            scores = self.compute_pair_scores()
            if self.verbose:
                print("Puntajes calculados:", scores)

            if scores == {}:
                if self.verbose:
                    print("No hay más pares para procesar. Finalizando entrenamiento.")
                break

            # Seleccionar el par con el puntaje más alto
            best_pair = max(scores, key=scores.get)
            if self.verbose:
                print(f"Mejor par seleccionado: {best_pair} con puntaje: {scores[best_pair]}")

            # Actualizar las divisiones de palabra
            self.word_pieces = self.merge_pair(best_pair[0], best_pair[1], self.word_pieces)
            if self.verbose:
                print("Nuevas divisiones (word_pieces):", self.word_pieces)

            # Agregar el nuevo token al vocabulario
            new_token = best_pair[0] + best_pair[1][2:] if best_pair[1].startswith("##") else best_pair[0] + best_pair[1]
            self.vocab.append(new_token)
            if self.verbose:
                print(f"Nuevo token añadido al vocabulario: {new_token}")

    def tokenize(self, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def tokenize_WordPiece(self, text):
        words = self.pre_tokenize(text)
        encoded_words = [self.tokenize(word) for word in words] 

        return sum(encoded_words, [])
