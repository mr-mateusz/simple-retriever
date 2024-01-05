import uuid


def save(data: str, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)


def load(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def generate_id() -> str:
    return str(uuid.uuid4())
