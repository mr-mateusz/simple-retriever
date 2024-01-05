import argparse
import os
import time
from collections.abc import Iterable

import requests
from loguru import logger

from utils import save, load


def get_webpage(url: str) -> str | None:
    # noinspection PyBroadException
    try:
        r = requests.get(url)
        if r.status_code != 200:
            raise Exception()

        return r.text
    except Exception:
        return None


def create_filename(url: str) -> str:
    return url.split('/')[-1] + '.html'


def download_and_save(url: str, output_dir: str | None) -> bool:
    content = get_webpage(url)

    if not content:
        return False

    name = create_filename(url)
    filepath = name if not output_dir else os.path.join(output_dir, name)

    save(content, filepath)
    return True


def download_webpages(urls: Iterable[str], output_dir: str | None, sleep_time: int = 1) -> None:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for url in urls:
        downloaded_ok = download_and_save(url, output_dir)

        if not downloaded_ok:
            logger.warning(f'Cannot download. {{"url": "{url}"}}')

        time.sleep(sleep_time)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='Webpage downloader',
        description='Download webpages listed in the input file')
    parser.add_argument('input_file', help='Path to a file that contains list of webpages to download')
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='Output directory')

    args = parser.parse_args()

    data = load(args.input_file)
    urls = [url for url in data.split('\n') if url]

    download_webpages(urls, args.output_dir)


if __name__ == '__main__':
    main()
