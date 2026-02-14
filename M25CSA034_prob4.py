
"""
Sports vs Politics Text Classification System

This system implements and compares three different machine learning approaches
for binary text classification:
1. Naive Bayes (Multinomial)
2. Logistic Regression
3. Support Vector Machine (SVM)

Features used:
- Bag of Words (BoW)
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Character and Word N-grams

Author: Sandesh Suman
Roll Number:M25CSA034
Course: CSL 7640 - Natural Language Understanding
"""

import re
import math
import json
from collections import defaultdict, Counter
import random


class TextPreprocessor:
    
    def __init__(self):
        # Common English stopwords for filtering
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
    
    def clean_text(self, text):
    
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        
        #Tokenize text into words.                
        return text.split()
    
    def remove_stopwords(self, tokens):
    
        #remove common stopwords from token list.
        return [token for token in tokens if token not in self.stopwords]
    
    def get_ngrams(self, tokens, n=2):
        
        #generate n-grams from token list.
       
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return ngrams


class FeatureExtractor:
    #Extracts numerical features from text documents.
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.vocabulary = set()
        self.idf_scores = {}
        self.document_count = 0
    
    def build_vocabulary(self, documents, use_ngrams=False, n=2, remove_stopwords=True):
        
        #Build vocabulary from document collection.

        self.vocabulary = set()
        
        for doc in documents:
            cleaned = self.preprocessor.clean_text(doc)
            tokens = self.preprocessor.tokenize(cleaned)
            
            if remove_stopwords:
                tokens = self.preprocessor.remove_stopwords(tokens)
            
            # Add unigrams
            self.vocabulary.update(tokens)
            
            # Add n-grams if requested
            if use_ngrams:
                ngrams = self.preprocessor.get_ngrams(tokens, n)
                self.vocabulary.update(['_'.join(ng) for ng in ngrams])
    
    def calculate_idf(self, documents, remove_stopwords=True):
        """
        Calculate IDF scores for vocabulary.
        
        IDF(term) = log(total_documents / documents_containing_term)
        
        """
        self.document_count = len(documents)
        document_frequency = defaultdict(int)
        
        for doc in documents:
            cleaned = self.preprocessor.clean_text(doc)
            tokens = self.preprocessor.tokenize(cleaned)
            
            if remove_stopwords:
                tokens = self.preprocessor.remove_stopwords(tokens)
            
            # Count unique terms per document
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token in self.vocabulary:
                    document_frequency[token] += 1
        
        # Calculate IDF scores
        for term in self.vocabulary:
            df = document_frequency.get(term, 0)
            if df > 0:
                self.idf_scores[term] = math.log(self.document_count / df)
            else:
                self.idf_scores[term] = 0
    
    def extract_bow(self, document, remove_stopwords=True):
        cleaned = self.preprocessor.clean_text(document)
        tokens = self.preprocessor.tokenize(cleaned)
        
        if remove_stopwords:
            tokens = self.preprocessor.remove_stopwords(tokens)
        
        bow = defaultdict(int)
        for token in tokens:
            if token in self.vocabulary:
                bow[token] += 1
        
        return dict(bow)
    
    def extract_tfidf(self, document, remove_stopwords=True):
        """
        Extract TF-IDF features.
        
        TF-IDF = TF × IDF
        where TF = (count of term in doc) / (total terms in doc)        
        """
        cleaned = self.preprocessor.clean_text(document)
        tokens = self.preprocessor.tokenize(cleaned)
        
        if remove_stopwords:
            tokens = self.preprocessor.remove_stopwords(tokens)
        
        # Calculate term frequency
        tf = defaultdict(int)
        for token in tokens:
            if token in self.vocabulary:
                tf[token] += 1
        
        # Normalize by document length
        doc_length = len(tokens) if tokens else 1
        
        # Calculate TF-IDF
        tfidf = {}
        for term, count in tf.items():
            tf_score = count / doc_length
            idf_score = self.idf_scores.get(term, 0)
            tfidf[term] = tf_score * idf_score
        
        return tfidf


class NaiveBayesClassifier:
    #Multinomial Naive Bayes classifier for text.
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = {}
        self.word_probs = defaultdict(dict)
        self.classes = set()
        self.vocabulary = set()
    
    def train(self, features_list, labels):
        
        #Train the Naive Bayes model.
        # Count documents per class
        class_counts = Counter(labels)
        total_docs = len(labels)
        
        # Calculate priors
        for cls in class_counts:
            self.classes.add(cls)
            self.class_priors[cls] = class_counts[cls] / total_docs
        
        # Build vocabulary
        for features in features_list:
            self.vocabulary.update(features.keys())
        
        # Calculate word probabilities per class
        class_word_counts = defaultdict(lambda: defaultdict(int))
        class_total_words = defaultdict(int)
        
        for features, label in zip(features_list, labels):
            for word, count in features.items():
                class_word_counts[label][word] += count
                class_total_words[label] += count
        
        # Apply Laplace smoothing
        vocab_size = len(self.vocabulary)
        for cls in self.classes:
            for word in self.vocabulary:
                word_count = class_word_counts[cls][word]
                total_count = class_total_words[cls]
                self.word_probs[cls][word] = (word_count + self.alpha) / \
                                             (total_count + self.alpha * vocab_size)
    
    def predict(self, features):        #Predict class for given features.                
        scores = {}
        
        for cls in self.classes:
            score = math.log(self.class_priors[cls])
            
            for word, count in features.items():
                if word in self.word_probs[cls]:
                    score += count * math.log(self.word_probs[cls][word])
            
            scores[cls] = score
        
        return max(scores, key=scores.get)


class LogisticRegressionClassifier:
    """Binary Logistic Regression classifier using gradient descent."""
    
    def __init__(self, learning_rate=0.01, iterations=1000, regularization=0.01):

        self.lr = learning_rate
        self.iterations = iterations
        self.reg = regularization
        self.weights = {}
        self.bias = 0
        self.vocabulary = set()
    
    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + math.exp(-max(min(z, 500), -500)))  # Clip to prevent overflow
    
    def train(self, features_list, labels): #Train using gradient descent.

        # Build vocabulary
        for features in features_list:
            self.vocabulary.update(features.keys())
        
        # Initialize weights
        self.weights = {word: 0.0 for word in self.vocabulary}
        self.bias = 0.0
        
        n = len(labels)
        
        # Gradient descent
        for iteration in range(self.iterations):
            total_loss = 0
            
            for features, y in zip(features_list, labels):
                # Calculate prediction
                z = self.bias
                for word, value in features.items():
                    z += self.weights.get(word, 0) * value
                
                prediction = self.sigmoid(z)
                error = prediction - y
                
                # Update weights
                self.bias -= self.lr * error
                
                for word, value in features.items():
                    self.weights[word] -= self.lr * (error * value + self.reg * self.weights[word])
                
                # Track loss
                total_loss += -(y * math.log(prediction + 1e-10) + 
                               (1 - y) * math.log(1 - prediction + 1e-10))
    
    def predict(self, features):#Predict class (0 or 1)
        z = self.bias
        for word, value in features.items():
            z += self.weights.get(word, 0) * value
        
        probability = self.sigmoid(z)
        return 1 if probability >= 0.5 else 0


class SVMClassifier:
    #Support Vector Machine classifier using simplified SMO algorithm.
    
    def __init__(self, C=1.0, iterations=100, kernel='linear'): #Initialize SVM.
        self.C = C
        self.iterations = iterations
        self.kernel_type = kernel
        self.weights = {}
        self.bias = 0
        self.vocabulary = set()
    
    def kernel(self, x1, x2):
        if self.kernel_type == 'linear':
            # Linear kernel: dot product
            result = 0
            for word in set(x1.keys()) | set(x2.keys()):
                result += x1.get(word, 0) * x2.get(word, 0)
            return result
        else:
            # RBF kernel
            diff = 0
            for word in set(x1.keys()) | set(x2.keys()):
                diff += (x1.get(word, 0) - x2.get(word, 0)) ** 2
            return math.exp(-0.1 * diff)
    
    def train(self, features_list, labels): #Train SVM using simplified approach.
        # Build vocabulary
        for features in features_list:
            self.vocabulary.update(features.keys())
        
        # Initialize weights
        self.weights = {word: 0.0 for word in self.vocabulary}
        self.bias = 0.0
        
        # Convert labels to -1, 1
        y = [1 if label == 1 else -1 for label in labels]        
        for iteration in range(self.iterations):
            for i, (features, yi) in enumerate(zip(features_list, y)):
                # Calculate prediction
                score = self.bias
                for word, value in features.items():
                    score += self.weights.get(word, 0) * value
                
                # Update if misclassified or within margin
                if yi * score < 1:
                    # Update weights
                    for word, value in features.items():
                        self.weights[word] += self.C * yi * value
                    self.bias += self.C * yi
    
    def predict(self, features):
        score = self.bias
        for word, value in features.items():
            score += self.weights.get(word, 0) * value
        
        return 1 if score >= 0 else 0


def load_dataset(filename):#    Load dataset from JSON file.
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    
    return documents, labels


def split_dataset(documents, labels, train_ratio=0.8, shuffle=True):
    """
    Split dataset into train and test sets.
    """
    # Combine and shuffle
    data = list(zip(documents, labels))
    if shuffle:
        random.seed(42)  # For reproducibility
        random.shuffle(data)
    
    # Split
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    train_docs = [d[0] for d in train_data]
    train_labels = [d[1] for d in train_data]
    test_docs = [d[0] for d in test_data]
    test_labels = [d[1] for d in test_data]
    
    return train_docs, train_labels, test_docs, test_labels


def evaluate_classifier(predictions, true_labels):#Calculate classification metrics

    # For binary classification
    tp = fp = tn = fn = 0
    
    for pred, true in zip(predictions, true_labels):
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 0:
            tn += 1
        else:
            fn += 1
    
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def print_results(method_name, feature_name, metrics):
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print(f"{method_name} with {feature_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"Confusion Matrix: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")


def main():
    """Main execution function."""
    print("Sports vs Politics Text Classification")
    print("="*60)
    
    # Load dataset
    print("\nLoading dataset...")
    documents, labels = load_dataset('dataset.json')
    
    # Convert string labels to binary
    label_map = {'sports': 1, 'politics': 0}
    binary_labels = [label_map[label] for label in labels]
    
    print(f"Total documents: {len(documents)}")
    print(f"Sports: {labels.count('sports')}, Politics: {labels.count('politics')}")
    
    # Split dataset
    train_docs, train_labels, test_docs, test_labels = split_dataset(
        documents, binary_labels, train_ratio=0.8
    )
    
    print(f"\nTraining set: {len(train_docs)} documents")
    print(f"Test set: {len(test_docs)} documents")
    
    # Initialize preprocessor and feature extractor
    preprocessor = TextPreprocessor()
    feature_extractor = FeatureExtractor(preprocessor)
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    feature_extractor.build_vocabulary(train_docs, use_ngrams=True, n=2)
    feature_extractor.calculate_idf(train_docs)
    print(f"Vocabulary size: {len(feature_extractor.vocabulary)}")
    
    # Extract features
    print("\nExtracting features...")
    
    # Bag of Words
    train_bow = [feature_extractor.extract_bow(doc) for doc in train_docs]
    test_bow = [feature_extractor.extract_bow(doc) for doc in test_docs]
    
    # TF-IDF
    train_tfidf = [feature_extractor.extract_tfidf(doc) for doc in train_docs]
    test_tfidf = [feature_extractor.extract_tfidf(doc) for doc in test_docs]
    
    results = {}
    
    # Test Naive Bayes with BoW
    print("\n" + "="*60)
    print("Training Naive Bayes with Bag of Words...")
    print("="*60)
    nb_bow = NaiveBayesClassifier()
    nb_bow.train(train_bow, train_labels)
    predictions = [nb_bow.predict(feat) for feat in test_bow]
    metrics = evaluate_classifier(predictions, test_labels)
    print_results("Naive Bayes", "Bag of Words", metrics)
    results['NB_BoW'] = metrics
    
    # Test Naive Bayes with TF-IDF
    print("\n" + "="*60)
    print("Training Naive Bayes with TF-IDF...")
    print("="*60)
    nb_tfidf = NaiveBayesClassifier()
    nb_tfidf.train(train_tfidf, train_labels)
    predictions = [nb_tfidf.predict(feat) for feat in test_tfidf]
    metrics = evaluate_classifier(predictions, test_labels)
    print_results("Naive Bayes", "TF-IDF", metrics)
    results['NB_TFIDF'] = metrics
    
    # Test Logistic Regression with TF-IDF
    print("\n" + "="*60)
    print("Training Logistic Regression with TF-IDF...")
    print("="*60)
    lr = LogisticRegressionClassifier(learning_rate=0.1, iterations=500)
    lr.train(train_tfidf, train_labels)
    predictions = [lr.predict(feat) for feat in test_tfidf]
    metrics = evaluate_classifier(predictions, test_labels)
    print_results("Logistic Regression", "TF-IDF", metrics)
    results['LR_TFIDF'] = metrics
    
    # Test SVM with TF-IDF
    print("\n" + "="*60)
    print("Training SVM with TF-IDF...")
    print("="*60)
    svm = SVMClassifier(C=1.0, iterations=100)
    svm.train(train_tfidf, train_labels)
    predictions = [svm.predict(feat) for feat in test_tfidf]
    metrics = evaluate_classifier(predictions, test_labels)
    print_results("SVM", "TF-IDF", metrics)
    results['SVM_TFIDF'] = metrics
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    for method, metrics in results.items():
        print(f"{method:20s} - Accuracy: {metrics['accuracy']*100:6.2f}%, F1: {metrics['f1_score']*100:6.2f}%")
    
    # Find best method
    best_method = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest Method: {best_method[0]} with {best_method[1]['accuracy']*100:.2f}% accuracy")


if __name__ == "__main__":
    main()