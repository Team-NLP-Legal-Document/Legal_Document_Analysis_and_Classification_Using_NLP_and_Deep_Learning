import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import spacy
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter
import pandas as pd
from tkinter import ttk

class LegalDocumentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NLP Legal Document Analysis")

        # Load spaCy English model
        self.nlp = spacy.load("en_core_web_sm")

        # Load BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        self.bert_model.eval()

        # Components
        self.create_menu()
        self.create_main_frame()

        # Initialize classifier and vectorizer
        self.classifier = None
        self.vectorizer = None

    def create_menu(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Document", command=self.open_document)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)

        help_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def create_main_frame(self):
        self.main_frame = tk.Frame(self.root, bg="lightgreen", width=1000)  # Set background color
        self.main_frame.pack(padx=10, pady=10)

        self.upload_button = tk.Button(self.main_frame, text="Upload Document", command=self.open_document, bg="blue", fg="white", font=("Arial", 12, "bold"))
        self.upload_button.pack(pady=10)

        self.preview_label = tk.Label(self.main_frame, text="Document Preview:", font=("Arial", 12, "bold"))
        self.preview_label.pack(pady=5)
        self.text_viewer = tk.Text(self.main_frame, height=10, width=50)
        self.text_viewer.pack(pady=10)

        self.preprocessing_label = tk.Label(self.main_frame, text="Preprocessing Options:", font=("Arial", 12, "bold"))
        self.preprocessing_label.pack(pady=5)

        self.text_cleaning_var = tk.BooleanVar()
        self.text_cleaning_checkbox = tk.Checkbutton(self.main_frame, text="Text Cleaning", variable=self.text_cleaning_var)
        self.text_cleaning_checkbox.pack()

        self.tokenization_var = tk.BooleanVar()
        self.tokenization_checkbox = tk.Checkbutton(self.main_frame, text="Tokenization", variable=self.tokenization_var)
        self.tokenization_checkbox.pack()

        self.lemmatization_var = tk.BooleanVar()
        self.lemmatization_checkbox = tk.Checkbutton(self.main_frame, text="Lemmatization", variable=self.lemmatization_var)
        self.lemmatization_checkbox.pack()

        self.feature_label = tk.Label(self.main_frame, text="Feature Extraction and Model Selection:", font=("Arial", 12, "bold"))
        self.feature_label.pack(pady=5)

        feature_extraction_options = ["TF-IDF", "BERT Embeddings", "Word Embeddings", "N-grams"]
        self.feature_extraction_var = tk.StringVar()
        self.feature_extraction_dropdown = tk.OptionMenu(self.main_frame, self.feature_extraction_var, *feature_extraction_options)
        self.feature_extraction_dropdown.pack()

        model_selection_options = ["Naive Bayes", "SVM", "Neural Network", "BERT-based Classifier"]
        self.model_selection_var = tk.StringVar()
        self.model_selection_dropdown = tk.OptionMenu(self.main_frame, self.model_selection_var, *model_selection_options)
        self.model_selection_dropdown.pack()

        self.training_label = tk.Label(self.main_frame, text="Model Training and Evaluation:", font=("Arial", 12, "bold"))
        self.training_label.pack(pady=5)

        self.train_model_button = tk.Button(self.main_frame, text="Train Model", command=self.train_model, bg="green", fg="white", font=("Arial", 12, "bold"))
        self.train_model_button.pack()

        self.classification_label = tk.Label(self.main_frame, text="Document Classification and Analysis:", font=("Arial", 12, "bold"))
        self.classification_label.pack(pady=5)

        self.classify_document_button = tk.Button(self.main_frame, text="Classify Document", command=self.classify_document, bg="orange", fg="white", font=("Arial", 12, "bold"))
        self.classify_document_button.pack()

        self.clear_button = tk.Button(self.main_frame, text="Clear", command=self.clear_document, bg="gray", fg="white", font=("Arial", 12, "bold"))
        self.clear_button.pack()

        self.visualization_label = tk.Label(self.main_frame, text="Visualization and Reporting:", font=("Arial", 12, "bold"))
        self.visualization_label.pack(pady=5)

        self.show_visualization_button = tk.Button(self.main_frame, text="Show Visualization", command=self.show_visualization, bg="red", fg="white", font=("Arial", 12, "bold"))
        self.show_visualization_button.pack()

    def open_document(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, 'r') as file:
                document_content = file.read()
                self.text_viewer.delete(1.0, tk.END)
                self.text_viewer.insert(tk.END, document_content)

    def preprocess_document(self, document_content):
        processed_text = document_content

        if self.text_cleaning_var.get():
            processed_text = clean_text(processed_text)

        if self.tokenization_var.get():
            processed_text = tokenize_text(processed_text)

        if self.lemmatization_var.get():
            processed_text = lemmatize_text(processed_text)

        return processed_text

    def select_features(self, preprocessed_document):
        feature_extraction_method = self.feature_extraction_var.get()
        if feature_extraction_method == "TF-IDF":
            return extract_tfidf_features(preprocessed_document)
        elif feature_extraction_method == "BERT Embeddings":
            return extract_bert_embeddings(preprocessed_document)
        elif feature_extraction_method == "Word Embeddings":
            return extract_word_embeddings(preprocessed_document)
        elif feature_extraction_method == "N-grams":
            return extract_ngrams(preprocessed_document)

    def select_model(self):
        model_selection = self.model_selection_var.get()
        if model_selection == "Naive Bayes":
            return "Naive Bayes"
        elif model_selection == "SVM":
            return "SVM"
        elif model_selection == "Neural Network":
            return "Neural Network"
        elif model_selection == "BERT-based Classifier":
            return "BERT-based Classifier"

    def train_model(self):
        messagebox.showinfo("Training", "Training the model...")

    def classify_document(self):
        document_content = self.text_viewer.get(1.0, tk.END)
        preprocessed_document = self.preprocess_document(document_content)
        selected_features = self.select_features(preprocessed_document)
        selected_model = self.select_model()

        classification_result = self.generate_classification_result(document_content)
        accuracy = self.generate_accuracy()
        key_entities = self.generate_key_entities(preprocessed_document)
        sentiment = self.generate_sentiment(document_content)
        topic = self.generate_topic(document_content)

        self.show_results_window(classification_result, accuracy, key_entities, sentiment, topic)

    def generate_classification_result(self, document_content):
        # Placeholder logic, replace with actual implementation
        return "Legal Document"  

    def generate_accuracy(self):
        # Placeholder logic, replace with actual implementation
        return "Accuracy: 90%"  

    def generate_key_entities(self, preprocessed_document):
        # Placeholder logic, replace with actual implementation
        words = preprocessed_document.split()
        word_frequencies = Counter(words)
        most_common_words = word_frequencies.most_common(5)
        key_entities = ", ".join(word[0] for word in most_common_words)
        
        # Create a table for legal words frequency and percentages
        word_table = pd.DataFrame(list(word_frequencies.items()), columns=['Word', 'Frequency'])
        word_table['Percentage'] = word_table['Frequency'] / word_table['Frequency'].sum() * 100
        
        # Display the table
        self.show_word_table(word_table)
        
        return key_entities

    def show_word_table(self, word_table):
        results_window = tk.Toplevel(self.root)
        results_window.title("Legal Words Table")
        
        tree = ttk.Treeview(results_window)
        tree["columns"] = ("Word", "Frequency", "Percentage")
        
        tree.heading("#0", text="Index")
        tree.column("#0", stretch=tk.NO, width=50)
        tree.heading("Word", text="Word")
        tree.column("Word", anchor=tk.W, width=150)
        tree.heading("Frequency", text="Frequency")
        tree.column("Frequency", anchor=tk.W, width=100)
        tree.heading("Percentage", text="Percentage")
        tree.column("Percentage", anchor=tk.W, width=100)
        
        for i, row in word_table.iterrows():
            tree.insert("", i, values=(i+1, row['Word'], row['Frequency'], row['Percentage']))

        tree.pack()

    def generate_sentiment(self, document_content):
        # Placeholder logic, replace with actual implementation
        return "Positive"  

    def generate_topic(self, document_content):
        # Placeholder logic, replace with actual implementation
        return "Legal"  

    def show_results_window(self, classification_result, accuracy, key_entities, sentiment, topic):
        results_window = tk.Toplevel(self.root)
        results_window.title("Classification Results")

        tk.Label(results_window, text=f"Classification Result: {classification_result}", font=("Arial", 12, "bold")).pack()
        tk.Label(results_window, text=f"Accuracy: {accuracy}", font=("Arial", 12, "bold")).pack()
        tk.Label(results_window, text=f"Key Entities: {key_entities}", font=("Arial", 12, "bold")).pack()
        tk.Label(results_window, text=f"Sentiment: {sentiment}", font=("Arial", 12, "bold")).pack()
        tk.Label(results_window, text=f"Topic: {topic}", font=("Arial", 12, "bold")).pack()

    def clear_document(self):
        self.text_viewer.delete(1.0, tk.END)

    def show_about(self):
        messagebox.showinfo("About", "NLP Legal Document Analysis App\nVersion 1.0\nDeveloped by Your Name")

    def show_visualization(self):
        document_content = self.text_viewer.get(1.0, tk.END)
        words = document_content.split()
        word_frequencies = Counter(words)

        # Plot a pie chart for the first 5 words with the highest frequency
        categories = list(word_frequencies.keys())[:5]
        values = list(word_frequencies.values())[:5]

        fig, ax = plt.subplots()
        ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack()

# Placeholder functions for text processing
def clean_text(text):
    # Placeholder for text cleaning logic
    return text

def tokenize_text(text):
    # Placeholder for tokenization logic
    return text

def lemmatize_text(text):
    # Placeholder for lemmatization logic
    return text

def extract_tfidf_features(text):
    # Placeholder for TF-IDF feature extraction logic
    return text
def extract_bert_embeddings(text):
    # Placeholder for BERT embedding feature extraction logic
    return text

def extract_word_embeddings(text):
    # Placeholder for Word Embeddings feature extraction logic
    return text

def extract_ngrams(text):
    # Placeholder for N-grams feature extraction logic
    return text

if __name__ == "__main__":
    root = tk.Tk()
    app = LegalDocumentApp(root)
    root.mainloop()
