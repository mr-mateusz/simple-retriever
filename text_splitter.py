from langchain.text_splitter import RecursiveCharacterTextSplitter


def __fix_splits_on_dot(texts: list[str]) -> None:
    """After split by '. ' or '.', separator is appended to the second text, so we fix it here.

    >>> texts = ['Sentence 1', '. Sentence 2.']
    >>> __fix_splits_on_dot(texts)
    >>> texts
    ['Sentence 1.', 'Sentence 2.']

    >>> texts = ['Sentence 1', '.Sentence 2.']
    >>> __fix_splits_on_dot(texts)
    >>> texts
    ['Sentence 1.', 'Sentence 2.']
    """
    for index, t in enumerate(texts[1:], 1):
        if t.startswith('. '):
            texts[index - 1] += '.'
            texts[index] = t[2:]
        elif t.startswith('.'):
            texts[index - 1] += '.'
            texts[index] = t[1:]


def to_chunks(text: str, chunk_size: int = 500, separators: list[str] | None = None) -> list[str]:
    """Split text into chunks using langchain RecursiveCharacterTextSplitter.

    >>> to_chunks('Sentence 1. Sentence 2.', 15)
    ['Sentence 1.', 'Sentence 2.']
    """
    separators = separators or ['\n\n', '\n', '. ', ' ', '.', '']
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separators=separators)
    texts = splitter.split_text(text)
    __fix_splits_on_dot(texts)
    return texts
