import glob
import os

import evaluate
import pandas as pd
from tqdm import tqdm

from service import QAService
from utils import load

reference_data_path = 'data/qa.csv'

articles_dir = 'webpages'
faq_dir = 'webpages_faq'

# Load articles
articles = [load(p) for p in glob.glob(os.path.join(articles_dir, '*.html'))]
articles_faq = [load(p) for p in glob.glob(os.path.join(faq_dir, '*.html'))]

# Load reference data
reference_data = pd.read_csv(reference_data_path)

# Create service and index articles
service = QAService.from_default()

for article in articles:
    service.add_html_article(article)

for article_faq in articles_faq:
    service.add_html_faq(article_faq, index_answers=True)

# Conduct experiments
metric = evaluate.load('rouge')
experiments = []

# Without finding similar questions in faq vectorstore
generated_answers = []
for query in tqdm(reference_data['Q']):
    generated_answers.append(service.query(query)[0].text)

metrics = metric.compute(predictions=generated_answers, references=reference_data['A'])
metrics['experiment'] = f'no cache'
metrics['threshold_val'] = 0
experiments.append(metrics)

# Try different threshold values for faq similar questions
for threshold in tqdm([0.85, 0.95]):
    generated_answers = []
    for query in tqdm(reference_data['Q']):
        generated_answers.append(service.find_best_answer(query, faq_threshold=threshold))

    metrics = metric.compute(predictions=generated_answers, references=reference_data['A'])

    metrics['experiment'] = f'cache {threshold}'
    metrics['threshold_val'] = threshold

    experiments.append(metrics)

results_df = pd.DataFrame(experiments)
