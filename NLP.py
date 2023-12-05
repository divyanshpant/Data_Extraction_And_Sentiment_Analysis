import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import cmudict
import os
import logging
import string
import re

def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def remove_element(driver, xpath):
    try:
        element = driver.find_element(By.XPATH, xpath)
        driver.execute_script("arguments[0].parentNode.removeChild(arguments[0]);", element)
    except NoSuchElementException:
        logging.warning(f"Element not found: {xpath}")

def load_stop_words(stop_words_file):
    
    with open(stop_words_file, 'r', encoding='ISO-8859-1') as file:
        stop_words = [word.strip().lower() for word in file]
    return stop_words

def clean_data(read_file, stop_words, positive_dict, negative_dict):
    words = word_tokenize(read_file)
    cleaned_words=[]
    
    for word in words:
        lowercase_word = word.lower()

        # Check if the word is in stop_words, but not if it's a part of another word
        if lowercase_word not in stop_words:
            for stop_word in stop_words:
                if lowercase_word.startswith(stop_word + '-') or lowercase_word.endswith('-' + stop_word):
                    break
            else:
                # If the loop did not break, add the word to cleaned_words
                cleaned_words.append(lowercase_word)
                
    cleaned_content = ' '.join(cleaned_words)
    
    positive_score = sum(1 for word in cleaned_words if word in positive_dict)
    negative_score = -1 * sum(-1 for word in cleaned_words if word in negative_dict)

    # Calculate Polarity Score
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

    # Calculate Subjectivity Score
    subjectivity_score = (positive_score + negative_score) / (len(cleaned_words) + 0.000001)

    return cleaned_content, positive_score, negative_score, polarity_score, subjectivity_score

def load_positive_negative_words(positive_file, negative_file):
    with open(positive_file, 'r', encoding = "ISO-8859-1") as file:
        positive_words = {word.strip().lower(): word.strip() for word in file}

    with open(negative_file, 'r', encoding = "ISO-8859-1") as file:
        negative_words = {word.strip().lower(): word.strip() for word in file}

    return positive_words, negative_words

def create_positive_negative_dictionaries(positive_words, negative_words, stop_words):
    positive_dict = {word.lower(): positive_words[word] for word in positive_words if word.lower() not in stop_words}
    negative_dict = {word.lower(): negative_words[word] for word in negative_words if word.lower() not in stop_words}
    return positive_dict, negative_dict

def create_excel_file(output_data, final_output_data):
    final_file_path = os.path.join(output_data, "Final_output.xlsx")

    df = pd.DataFrame(final_output_data)
    if os.path.exists(final_file_path):
        # If the file exists, read the existing data and append the new data in the next row
        existing_data = pd.read_excel(final_file_path)
        df = pd.concat([existing_data, df], ignore_index=True).drop_duplicates()

    df.to_excel(final_file_path, index=False)

    logging.info(f"Sentiment analysis results saved to {final_file_path}")
    
    
def find_element_by_xpath(driver, xpaths):
    for xpath in xpaths:
        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, xpath)))
            if element:
            
                return element.text
                break

        except Exception:
            print("Xpath not found trying another one")

    return None

def calculate_readability_metrics(cleaned_content):
    sentences = sent_tokenize(cleaned_content)

    final_sentence=""
    # Print the split sentences
    for i, sentence in enumerate(sentences, 1):
        stripped_sentence = sentence.strip()
        if stripped_sentence:
            final_sentence += f"{stripped_sentence}\n"

    # Count the number of sentences based on the final_sentence variable
    num_sentences = len([s for s in final_sentence.split("\n") if s])

    total_words = len(word_tokenize(cleaned_content))
    total_sentences = num_sentences
    #print(f"The provided text contains {total_sentences} sentences.")
    #print(f"The provided text contains {total_words} words.")
    average_sentence_length = total_words / total_sentences
    
    pronouncing_dict = cmudict.dict()
    identifying_complex_word = [
        word
        for word in word_tokenize(cleaned_content)
        if pronouncing_dict.get(word.lower()) is not None and max(
            [len(list(y for y in x if y[-1].isdigit())) for x in pronouncing_dict[word.lower()]]
        )
    ]
    
    complex_words = len(identifying_complex_word)
    #print("Total Number of Complex Words:", complex_words)
    percentage_complex_words = complex_words / total_words
    fog_index = 0.4 * (average_sentence_length + percentage_complex_words)
    
    complex_word_2 = [
        word
        for word in word_tokenize(cleaned_content)
        if pronouncing_dict.get(word.lower()) is not None and max(
            [len(list(y for y in x if y[-1].isdigit())) for x in pronouncing_dict[word.lower()]]
        ) > 2
    ]
    
    complex_words_more_than_2 = len(complex_word_2)

    return average_sentence_length, percentage_complex_words, fog_index, complex_words_more_than_2


def count_syllables(word):
    # Remove punctuation and convert to lowercase
    cleaned_word = word.strip(string.punctuation).lower()
    
    # Handle exceptions for words ending with 'es' or 'ed'
    if cleaned_word.endswith('es') or cleaned_word.endswith('ed'):
        return 0
    
    # Count vowels in the word
    vowels = 'aeiou'
    count = sum(1 for char in cleaned_word if char in vowels)
    
    for i in range(len(cleaned_word) - 1):
        if cleaned_word[i:i+2] in ['ae', 'ai', 'ei', 'oi', 'ou', 'ea']:
            count -= 1
    
    return max(1, count)  # Ensure at least one syllable

def cal_personal_pronouns(read_file):
    text = word_tokenize(read_file)
    text2 = ' '.join(text)
    pronoun_re = re.compile(r'\b(I|we|ours|my|mine|(?-i:us))\b', re.I) #?-i modifier, which makes the "us" case-sensitive
    matches = pronoun_re.findall(text2)
    return matches

def avg_word_len(read_file):
    text = word_tokenize(read_file)
    text2 = ' '.join(text)
    text3 =text2.translate(str.maketrans('', '', string.punctuation))
    cleaned_text = re.sub(' +', ' ', text3)
    words = cleaned_text.split()
    word_lengths = [len(word) for word in words]
    word_len=sum(word_lengths) / len(words)
    return word_len




def scrape_website(url, url_id, stop_words_files, article_folder, positive_words, negative_words, output_data):
    logging.info(f"Scraping data from {url}")

    chrome_options = Options()
    chrome_options.add_argument("incognito")
    chrome_options.add_argument("headless")
    driver = webdriver.Chrome(options=chrome_options)
    
    final_output_data = []

    try:
        driver.get(url)

        # Extract article title
        try:
            title_element = driver.find_element(By.XPATH, "//h1")
            article_title = title_element.text
        except NoSuchElementException:
            logging.warning(f"Title element not found for {url}. Skipping.")
            return
        
        """

        # Remove unwanted elements
        remove_element(driver, "//footer")
        remove_element(driver, "//*[@id='tdi_20']")
        remove_element(driver, "//div[contains(@class,'td-full-screen-header-image-wrap')]")
        remove_element(driver, "//article[contains(@class,'td-post-template-7')]//div[contains(@class,'td-pb-row')]//pre[contains(@class,'wp-block-preformatted')]")
        remove_element(driver, "//article[contains(@class,'td-post-template-7')]//div[contains(@class,'td_block_related_posts')]")
        remove_element(driver, "//article[contains(@class,'td-post-template-7')]//div[contains(@class,'td-pb-span4')]")
        remove_element(driver, "//*[@id='tdi_119']/div")
        remove_element(driver, "//*[@id='tdi_157']")

        # Extract article content
        article_content = driver.find_element(By.XPATH, "//body").text
        
        """
        
        remove_element(driver, "//div[contains(@class,'td-pb-row')]//pre[contains(@class,'wp-block-preformatted')]")
        
        xpaths_to_try = ["//article[contains(@class,'td-post-template-7')]//div[contains(@class,'td-post-content')]", "//*[@id='tdi_117']/div/div[1]/div/div[11]/div"]
        
        article_content = find_element_by_xpath(driver, xpaths_to_try)
        
        file_name = f"{article_folder}/{url_id}.txt"
        
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(f"{article_title}\n\n")
            file.write(f"{article_content}")

        with open(file_name, 'r', encoding="utf-8") as f:
            read_file = f.read()
            cleaned_content= read_file
            
        for stop_words_file in stop_words_files:
            
            try:
                
                # Load stop words from the current file
                stop_words = load_stop_words(stop_words_file)
                cleaned_content,positive_score, negative_score, polarity_score, subjectivity_score = clean_data(cleaned_content, stop_words, positive_words, negative_words)
            except Exception as e:
                print(f"Error in processing stop words file {stop_words_file}: {e}")

        # Save to text file
        #article_folder_name = f"{article_folder}/{url_id}.txt"
        
        # Calculate Readability Metrics
        average_sentence_length, percentage_complex_words, fog_index, complex_words_more_than_2 = calculate_readability_metrics(cleaned_content)
        
        

        #with open(article_folder_name, "w", encoding="utf-8") as file1:
            #file1.write(f"{cleaned_content}\n\n")
        
        cleaned_text_without_punctuation = cleaned_content.translate(str.maketrans('', '', string.punctuation))
        cleaned_word_count=len(nltk.word_tokenize(cleaned_text_without_punctuation))
        
        
        syllable_words = word_tokenize(cleaned_content)
        syllable_count = {}
        for syllable_word in syllable_words:
            syllable_count[syllable_word] = count_syllables(syllable_word)
        total_syllables = sum(syllable_count.values())

        # Print individual counts and the total count
        #for syllable_word, syllable_counts in syllable_count.items():
            #print(f"{syllable_word}: {syllable_counts} syllable(s)")
        #print(f"Total Syllables: {total_syllables}")
        
        ##finding pronouns
        matches = cal_personal_pronouns(read_file)
        pronoun_count = len(matches)
        
        ##Average Word Length
        word_len = avg_word_len(read_file)
                
        # Append results to the output_data list
        final_output_data.append({
            'URL_ID': url_id,
            'URL': url,
            'Positive Score': positive_score,
            'Negative Score': negative_score,
            'Polarity Score': polarity_score,
            'Subjectivity Score':subjectivity_score,
            'Average Sentence Length': average_sentence_length,
            'Percentage of Complex Words': percentage_complex_words,
            'Fog Index': fog_index,
            'Avg Number Of Words Per Sentence':average_sentence_length,
            'Complex Word Count': complex_words_more_than_2,
            'Word Count': cleaned_word_count,
            'Syllable Per Word': total_syllables,
            'Personal Pronouns': pronoun_count,
            'Average Word Length': word_len
        })

        #logging.info(f"Article from {url} saved to {article_folder_name} with sentiment scores.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        

    finally:
        create_excel_file(output_data, final_output_data)
        driver.quit()

def scrape_from_excel(input_file_path, stop_words_folder, article_folder, positive_file, negative_file, output_data):
    # Load positive and negative words
    positive_words, negative_words = load_positive_negative_words(positive_file, negative_file)

    df = pd.read_excel(input_file_path)
    stop_words_files = [os.path.join(stop_words_folder, file) for file in os.listdir(stop_words_folder) if file.endswith('.txt')]

    for index, row in df.iterrows():
        url = row['URL']
        url_id = row['URL_ID']
        scrape_website(url, url_id, stop_words_files, article_folder, positive_words, negative_words, output_data)

if __name__ == "__main__":
    configure_logging()
    
    output_data= 'C:/Users/ddivy/OneDrive/Documents/VS CODE/Data_Extraction_And_Sentiment_Analysis/'
    
    positive_file='C:/Users/ddivy/OneDrive/Documents/VS CODE/Data_Extraction_And_Sentiment_Analysis/MasterDictionary/positive-words.txt'
    
    negative_file='C:/Users/ddivy/OneDrive/Documents/VS CODE/Data_Extraction_And_Sentiment_Analysis/MasterDictionary/negative-words.txt'

    stop_words_folder = 'C:/Users/ddivy/OneDrive/Documents/VS CODE/Data_Extraction_And_Sentiment_Analysis/StopWords/'

    article_folder = 'C:/Users/ddivy/OneDrive/Documents/VS CODE/Data_Extraction_And_Sentiment_Analysis/ArticleExtract'
    
    input_file_path= 'C:/Users/ddivy/OneDrive/Documents/VS CODE/Data_Extraction_And_Sentiment_Analysis/input.xlsx'

    scrape_from_excel(input_file_path, stop_words_folder, article_folder, positive_file, negative_file, output_data)
