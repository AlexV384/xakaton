import os
import shutil
import tkinter as tk
from tkinter import ttk, filedialog
import nltk
from nltk.corpus import stopwords
from main import TextProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(doc, topic, vectorizer):
    # Рассчитываем косинусную схожесть между документом и темой с использованием TF-IDF векторизатора.
    tfidf_matrix = vectorizer.transform([doc, topic])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity
def load_texts_from_folder(folder_path):
    # Загружаем тексты из указанной папки и возвращаем словарь с именем файла и его содержимым.
    texts = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            texts[filename] = file.read()
    return texts
def classify_documents(documents, topics, vectorizer, similarity_threshold=0.035):
    # Классифицируем документы на основе их сходства с темами.
    result = []
    topics_folder = "themes/utf8"
    for doc_name, doc_content in documents.items():
        max_similarity = 0
        assigned_topic = 0  # Начальное значение - тема 0 (неопознанная)
        # Извлекаем числовую часть из имен тем.
        numeric_topic_names = [int(name.split('.')[0]) for name in topics.keys() if name.split('.')[0].isdigit()]
        # Находим максимальное числовое имя темы.
        last_topic_name = str(max(numeric_topic_names, default=0))
        for topic_name, topic_content in topics.items():
            similarity = calculate_similarity(doc_content, topic_content, vectorizer)
            if similarity > max_similarity and similarity >= similarity_threshold:
                max_similarity = similarity
                assigned_topic = topic_name
        # Если текст не соответствует ни одной теме, сохраняем его как новую тему.
        if assigned_topic == 0:
            new_topic_name = str(int(last_topic_name) + 1)
            new_topic_path = os.path.join(topics_folder, f"{new_topic_name}.txt")
            with open(new_topic_path, "w", encoding="utf-8") as new_topic_file:
                new_topic_file.write(doc_content)
            assigned_topic = new_topic_name
            topics[new_topic_name] = doc_content
        result.append((doc_name, assigned_topic))
    return result


def classify_documents_async(text_processor, documents_folder, output_folder, topics_folder, result_label):
    # Классифицируем документы асинхронно.
    output_themes_folder = "output_themes"
    text_processor.process_folder(documents_folder, output_folder)
    text_processor.process_folder(topics_folder, output_themes_folder)
    documents = load_texts_from_folder(output_folder)
    topics = load_texts_from_folder(output_themes_folder)
    all_texts = list(documents.values()) + list(topics.values())
    stop_words = list(set(stopwords.words('russian')))
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    vectorizer.fit(all_texts)
    classified_documents = classify_documents(documents, topics, vectorizer)
    result_file_path = os.path.abspath("classification.txt")
    with open(result_file_path, "w", encoding='utf-8') as output_file:
        for doc_name, assigned_topic in classified_documents:
            assigned_topic_without_extension, _ = os.path.splitext(assigned_topic)
            output_file.write(f"{doc_name}\t{assigned_topic_without_extension}\n")
    result_label.config(text=f"Классификация завершена. Результат сохранен в:\n{result_file_path}")
def select_documents_folder(text_processor, result_label):
    # Запрашиваем у пользователя выбор папки с документами и запускаем асинхронную классификацию.
    global output_folder, topics_folder
    documents_folder = filedialog.askdirectory(title="Выберите папку с документами")
    result_label.config(text="Выполняется классификация...")
    root.update()
    root.after(100, classify_documents_async, text_processor, documents_folder, output_folder, topics_folder,
               result_label)
def add_topic_file(result_label):
    # Запрашиваем у пользователя выбор файла для добавления в темы и копируем его в соответствующую папку.
    topic_file = filedialog.askopenfilename(title="Выберите файл для добавления в темы")
    if topic_file:
        topics_folder = "themes/utf8"
        shutil.copy(topic_file, topics_folder)
        result_label.config(
            text=f"Файл скопирован в папку тем: {os.path.join(topics_folder, os.path.basename(topic_file))}")
def remove_topic_file(result_label):
    # Запрашиваем у пользователя выбор файла для удаления из тем и удаляем его.
    file_path = filedialog.askopenfilename(title="Выберите файл для удаления из тем")
    os.remove(file_path)
    result_label.config(text="Файл удален из тем.")
def on_add_topic(result_label):
    # Обработчик события добавления файла в темы.
    add_topic_file(result_label)
def on_remove_topic(result_label):
    # Обработчик события удаления файла из тем.
    remove_topic_file(result_label)
def main():
    # Основная функция программы.
    global root, output_folder, topics_folder
    nltk.download('punkt')
    nltk.download('stopwords')
    root = tk.Tk()
    root.title("Управление документами и темами")
    root.geometry("400x250")  # Задаем начальный размер окна
    root.resizable(True, True)  # Разрешаем изменение размеров в обоих направлениях
    # Настраиваем стиль для тематических виджетов
    style = ttk.Style()
    style.configure("TButton", padding=6, relief="flat", background="#ccc")
    style.map("TButton", background=[("active", "#ddd")])
    # Создаем и размещаем кнопки и метку
    text_processor = TextProcessor()
    output_folder = "output"
    output_themes_folder = "output_themes"
    topics_folder = "themes/utf8"
    button_select_documents = ttk.Button(root, text="Выбрать папку с документами",
                                         command=lambda: select_documents_folder(text_processor, result_label))
    button_add_topic = ttk.Button(root, text="Добавить файл в темы",
                                  command=lambda: on_add_topic(result_label))
    button_remove_topic = ttk.Button(root, text="Удалить файл из тем",
                                     command=lambda: on_remove_topic(result_label))
    result_label = tk.Label(root, text="")
    # Упаковываем виджеты и добавляем эффект наведения
    button_select_documents.pack(pady=10)
    button_add_topic.pack(pady=10)
    button_remove_topic.pack(pady=10)
    result_label.pack(pady=10)
    root.mainloop()
if __name__ == "__main__":
    main()