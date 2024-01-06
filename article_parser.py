from bs4 import BeautifulSoup, Tag

from data_model import FaqEntry


def replace_normalize(text: str) -> str:
    """Replace non-breaking spaces, zero-width spaces and strip text."""
    return text.replace(' ', ' ').replace('​', '').strip()


def _extract_section_text(soup: BeautifulSoup) -> list[str]:
    texts = []
    for s in soup.find_all('section'):
        # only sections with 'id' attribute
        if 'id' not in s.attrs:
            continue
        texts.append(s.text)
    return texts


def parse_article_base(html: str) -> str:
    """Extract text from html article."""
    soup = BeautifulSoup(html, features='html.parser')

    try:
        article_text = soup.article.text
    except AttributeError:
        article_text = ''

    section_texts = _extract_section_text(soup)

    for st in section_texts:
        article_text += '\n\n' + st

    article_text = replace_normalize(article_text)

    return article_text


def split_title(text: str) -> tuple[str, str]:
    """Split article into title and text."""
    title, text = text.split('\n', maxsplit=1)
    return title, text


def __parse_details_tags(details_tags: Tag) -> dict:
    question = ''
    answer = ''
    for detail_tag in details_tags:
        if detail_tag.name == 'summary':
            question = detail_tag.text
        else:
            answer += ' ' + detail_tag.text
    question = replace_normalize(question)
    answer = replace_normalize(answer)

    return {'question': question, 'answer': answer}


def __parse_editor_content_div(editor_content_div: Tag) -> list[FaqEntry]:
    category = None
    qa_pairs = []
    for element in editor_content_div:
        if isinstance(element, Tag):
            if element.name == 'h3':
                category = element.text
            if element.name == 'div':
                for details_tags in element.find_all('details'):
                    qa_pair = __parse_details_tags(details_tags)
                    qa_pairs.append(FaqEntry(**qa_pair, category=category))
    return qa_pairs


def __modify_a_tags(soup: BeautifulSoup) -> None:
    a_tags = soup.find_all('a')
    for tag in a_tags:
        tag.string = f"{tag.text} ({tag.attrs['href']})"


def parse_faq(html: str) -> list[FaqEntry]:
    """Parse FAQ article into list of question-answer pairs along with category."""
    soup = BeautifulSoup(html, features='html.parser')

    __modify_a_tags(soup)

    qa_pairs = []
    for ecd in soup.find_all('div', attrs={'class': 'editor-content'}):
        qa_pairs.extend(__parse_editor_content_div(ecd))

    return qa_pairs
