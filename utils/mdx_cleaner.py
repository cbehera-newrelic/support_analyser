import re

def clean_mdx(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML/JSX tags
    text = re.sub(r'{.*?}', '', text)    # remove JS expressions
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # code blocks
    return text.strip()