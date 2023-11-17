import os
import concurrent.futures
import nltk
import pymorphy2
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

class TextProcessor:
    def __init__(self):
        # Загружаем необходимые ресурсы для обработки текста с использованием библиотеки NLTK.
        nltk.download('punkt')
        nltk.download('stopwords')
        # Создаем экземпляр морфологического анализатора Pymorphy2.
        self.morph = pymorphy2.MorphAnalyzer()
    def preprocess_text(self, text):
        # Удаляем знаки препинания и цифры из текста.
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\b\w*\d\w*\b', '', text)
        # Токенизируем текст на слова.
        words = word_tokenize(text, language='russian')
        # Приводим слова к нормальной форме и приводим их к нижнему регистру.
        normalized_words = [self.morph.parse(word)[0].normal_form.lower() for word in words]
        return normalized_words
    def remove_specific_words(self, words, words_to_remove):
        # Удаляем определенные слова из списка слов.
        return [word for word in words if word not in words_to_remove]
    def extract_keywords(self, text):
        # Проводим предварительную обработку текста.
        normalized_words = self.preprocess_text(text)
        # Загружаем стоп-слова для русского языка из библиотеки NLTK.
        stop_words = set(stopwords.words('russian'))
        # Фильтруем слова согласно заданным условиям.
        filtered_words = [word for word in normalized_words if len(word) > 1 and
                           word.isalnum() and
                           word not in stop_words and
                           'VERB' not in self.morph.parse(word)[0].tag and
                           'NPRO' not in self.morph.parse(word)[0].tag and
                           'INFN' not in self.morph.parse(word)[0].tag and
                           'INTJ' not in self.morph.parse(word)[0].tag and
                           'PRTF' not in self.morph.parse(word)[0].tag and
                           'PRCL' not in self.morph.parse(word)[0].tag and
                           'CONJ' not in self.morph.parse(word)[0].tag and
                           'PRED' not in self.morph.parse(word)[0].tag and
                           'PREP' not in self.morph.parse(word)[0].tag]
        # Удаляем специфичные слова из списка.
        specific_words_to_remove = [
            'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
            'меня', 'тебя', 'его', 'её', 'нас', 'вас', 'их',
            'себя', 'себе', 'собой', 'мой', 'твой', 'его', 'её',
            'наш', 'ваш', 'их', 'свой', 'кто', 'что', 'какой',
            'который', 'чей', 'где', 'когда', 'куда', 'откуда',
            'зачем', 'почему', 'сколько', 'тот', 'та', 'то',
            'те', 'этот', 'эта', 'это', 'эти', 'такой', 'такая',
            'такое', 'такие', 'столько', 'весь', 'вся', 'всё',
            'все', 'каждый', 'каждая', 'каждое', 'каждые', 'другой',
            'другая', 'другое', 'другие', 'сам', 'сама', 'само',
            'сами', 'самый', 'самая', 'самое', 'самые', 'всякий',
            'всякая', 'всякое', 'всякие', 'кто-то', 'что-то',
            'какой-то', 'который-то', 'чей-то', 'где-то', 'когда-то',
            'куда-то', 'откуда-то', 'зачем-то', 'почему-то', 'сколько-то', 'мочь'
        ]
        filtered_words = self.remove_specific_words(filtered_words, specific_words_to_remove)
        # Подсчитываем количество каждого слова.
        word_count = Counter(filtered_words)
        # Сортируем слова по частоте их встречаемости в обратном порядке.
        sorted_word_count = dict(sorted(word_count.items(), key=lambda item: (item[1], item[0]), reverse=True))
        return sorted_word_count
    def process_file(self, file_path, output_folder):
        try:
            # Читаем содержимое файла.
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            print(f"Ошибка при чтении документа {file_path}: {e}")
            return
        # Извлекаем ключевые слова из текста.
        keywords_count = self.extract_keywords(text)
        # Формируем путь для сохранения результатов обработки.
        output_path = os.path.join(output_folder, os.path.basename(file_path))
        # Записываем результаты обработки в файл.
        with open(output_path, 'w', encoding='utf-8') as output_file:
            for word, count in keywords_count.items():
                output_file.write(f"{word.upper()} {count}\n")
    def process_folder(self, input_folder, output_folder):
        # Проверяем наличие папки для сохранения результатов обработки и создаем ее, если она отсутствует.
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Формируем список файлов в указанной папке.
        files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
        # Используем многозадачность для параллельной обработки файлов.
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.process_file, file_path, output_folder) for file_path in files]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Ошибка при обработке файла: {e}")
if __name__ == "__main__":
    # Создаем экземпляр класса TextProcessor.
    text_processor = TextProcessor()
    # Обрабатываем указанную папку с документами и сохраняем результаты в указанную папку.
    text_processor.process_folder('docs/utf8', 'output')