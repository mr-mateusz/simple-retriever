import argparse
import glob
import os

from tqdm import tqdm

from service import QAService
from utils import load


def run_service(articles_dir: str, limit: int) -> None:
    paths = glob.glob(os.path.join(articles_dir, '*.html'))

    if not paths:
        print(f'Nie znaleziono artykułów w katalogu {articles_dir}')
        exit()
    print(f'Znaleziono {len(paths)} artykułów w katalogu {articles_dir}')

    print('Trwa indeksowanie dokumentów...')

    texts = [load(p) for p in paths]

    service = QAService.from_default()

    for t in tqdm(texts):
        service.add_html_article(t)

    print(f'Zaindeksowano łacznie {service.vectorstore.total_chunks} fragmentów dokumentów.')

    while True:
        try:
            query = input('Zadaj pytanie: ')
        except KeyboardInterrupt:
            break

        res = service.query(query, limit)

        for r in res:
            print('-' * 30)
            print(f'Dokument: {r.metadata["title"]}')
            print(r.text)
            print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Simple QA Service')
    parser.add_argument('articles_dir', help='Path to a directory that contains list of webpages to download')
    parser.add_argument('-l', '--limit', help='Number of document chunks to return', default=3, type=int)
    args = parser.parse_args()

    run_service(args.articles_dir, args.limit)
