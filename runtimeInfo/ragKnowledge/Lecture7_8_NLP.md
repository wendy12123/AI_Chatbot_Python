# Natural Language Processing (NLP) Knowledge Base

## 1. Overview of NLP

Natural Language Processing combines linguistics, statistics, and machine learning to enable computers to "understand" and process human language.

* **The Challenge**: Human language is unstructured and highly ambiguous.
* **Types of Ambiguity**:
  * **Lexical Ambiguity**: A single word has multiple meanings (e.g., "Bank").
  * **Syntactic Ambiguity**: A sentence can be parsed in different ways (e.g., "The man saw the girl with the telescope").
  * **Referential Ambiguity**: Unclear what a pronoun refers to (e.g., "The car hit the tank while it was moving").

---

## 2. Text Preprocessing (Traditional NLP)

Before feeding text into a model, it must be cleaned and structured. Key concepts include:

### Tokenization

The process of breaking down a body of text into smaller units called **tokens** (usually individual words or sentences).

* **spaCy Containers**: A `Doc` object contains `Sent` (sentences), `Token` (words), and `Span` (groups of words).

### Normalization: Stemming vs. Lemmatization

* **Stemming**: Removing the end of a word to reach a "stem," even if the result isn't a real dictionary word (e.g., "running" $\rightarrow$ "runn").
* **Lemmatization**: A more informed approach that considers context and converts a word to its dictionary form, or **lemma** (e.g., "better" $\rightarrow$ "good").

### Bag-of-Words (BoW)

A method of representing text data as numerical vectors. It creates a vocabulary of all unique words in a corpus and tracks the occurrence of those words in specific documents, ignoring grammar and word order.

---

## 3. NLP Software Ecosystem

Modern NLP utilizes several specialized libraries:

* **NLTK (Natural Language Toolkit)**: A leading platform for building Python programs to work with human language data.
* **spaCy**: Designed for production use, offering efficient tokenization and containerization.
* **TensorFlow & Keras**: Used for building Deep Learning models (ANNs, CNNs, RNNs) to classify or generate text.

---

## 4. Text-to-Sequence Pipeline (TensorFlow/Keras)

To process text using Neural Networks, data must follow a specific pipeline:

1. **Tokenization**: Assigning a unique integer index to every word in the vocabulary.
2. **Out of Vocabulary (OOV) Tokens**: Using a special tag (e.g., `<OOV>`) for words not seen during training to maintain sentence structure.
3. **Sequencing**: Converting sentences into lists of integers based on the tokenizer's word index.
4. **Padding**: Ensuring all sequences are the same length by adding zeros to the beginning or end, which is a requirement for neural network input layers.

---

## 5. Text Classification Workflow

1. **Data Loading**: Importing text data (e.g., from a `.json` or `.csv` file).
2. **Preprocessing**: Applying tokenization, sequencing, and padding.
3. **Model Architecture**:
    * **Embedding Layer**: Maps word indices to dense vectors representing semantic meaning.
    * **Global Average Pooling**: Reduces dimensionality.
    * **Dense Layers**: Fully connected layers with **ReLU** activation for learning patterns.
    * **Output Layer**: Typically uses **Sigmoid** (for binary classification) or **Softmax** (for multi-class) to provide a final prediction.
4. **Evaluation**: Testing the model on unseen text to check for accuracy in categorizing sentiment or topics.
