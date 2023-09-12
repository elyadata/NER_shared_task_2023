import argparse
import json
import os
import uuid
from pathlib import Path

from loguru import logger
from tqdm import tqdm


def extract_sentences(file_path: str | Path) -> list[str]:
    """
    Extracts sentences from the specified file.
    Args:
        file_path: File to be read.

    Returns:
        Sentences in the file.
    """
    lines = []
    current_line = ""

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == "":
                if current_line:
                    lines.append(current_line.strip())
                    current_line = ""
            else:
                current_line += line

    return lines


def extract_labels_from_sentence(sentence: str) -> list[tuple]:
    """
    This function will parse the sentence into tokens and labels in an ordered sequence.
    Args:
        sentence: The sentence for which the tags are to be extracted.

    Returns:
        An ordered list of sentences with tags.

    """
    tokens_with_labels = sentence.split('\n')

    intermediates = []
    spans = []
    completed_spans = []

    for index in range(len(tokens_with_labels)):
        token_with_label = tokens_with_labels[index]
        labels = token_with_label.split("\t")[1].split(" ")
        token = token_with_label.split("\t")[0].split(" ")[0]

        spans.extend([(tag, index, index + 1, token) for tag in labels if "B-" in tag])
        intermediates.extend([(tag, index) for tag in labels if "I-" in tag])

    for span in spans:
        entity = span[0].split("B-")[-1]
        index = span[2]
        while ("I-" + entity, index) in intermediates:
            span = (entity, span[1], index + 1, span[-1])
            index += 1

        completed_spans.append(span)

    return completed_spans


def find_word_indices(sentence: str, word: str) -> (int, int):
    """
    This function computes the span of a word in a given sentence.
    Args:
        sentence: Sentence where the word is located.
        word: Word for which the spans are to be computed.

    Returns:
        The `start` and `end` indices for the considered word.

    """
    start = sentence.find(word)
    end = start + len(word)

    return start, end


def extract_tokens(sentence: str) -> list[str]:
    """
    Extracts tokens from a sentence. A token, in this context, is a word.
    Args:
        sentence: Sentence considered.

    Returns:
        A list of the tokens in that sentence.
    """
    tokens = list()

    for token_with_label in sentence.split('\n'):
        token = token_with_label.split("\t")[0]
        tokens.append(token)
    return tokens


def find_all_words_indices(sentence: str) -> (list[int], list[int]):
    """
    Computes the spans for all words in a given sentence.
    Args:
        sentence: Considered sentence.

    Returns:
        A list of all spans of words in this sentence.
    """
    start_indices = []
    end_indices = []

    start_indices.append(0)
    for i, char in enumerate(sentence):
        if char == ' ':
            end_indices.append(i)
            start_indices.append(i + 1)

    return start_indices, end_indices


def tokens_to_text(tokens: list) -> str:
    """
    Converts tokens to text.
    Args:
        tokens: list of words to be converted.

    Returns:
        The obtained text.
    """
    return " ".join(tokens)


def create_one_line(sentence: str) -> dict:
    """
    Creates a single json line for the manifest file to be created.
    Args:
        sentence: The sentence to be formatted.

    Returns:
        Formatted sentence.
    """
    json_line = dict()

    text = tokens_to_text(extract_tokens(sentence))
    starts, ends = find_all_words_indices(text)
    entity_starts, entity_ends = [], []
    entities = [elem[-1] for elem in extract_labels_from_sentence(sentence)]

    for entity in entities:
        start, end = find_word_indices(word=entity, sentence=text)

        # logger.debug(f"start={start}, end={end}")   

        entity_starts.append(start)
        entity_ends.append(end)
        if start in starts:
            starts.remove(start)
        if end in ends:
            ends.remove(end)

    json_line["text"] = text
    json_line["entity_types"] = [tag[0].replace('B-', '') for tag in extract_labels_from_sentence(sentence)]
    json_line["id"] = str(uuid.uuid4())
    json_line["entity_start_chars"] = entity_starts
    json_line["entity_end_chars"] = entity_ends
    json_line["word_start_chars"] = starts
    json_line["word_end_chars"] = ends

    return json_line


def delete_if_exists(file: str | Path) -> None:
    """
    Checks if a file exists and deletes it.
    Args:
        file: File path.

    Returns:
        None.
    """
    if os.path.exists(file):
        os.remove(file)


def create_jsonlines_file(sentences: list[str], output_file_path: str | Path) -> None:
    """
    Given a list of sentences read from the Wojood manifest file, returns a JSON file compatible with BINDER.
    Args:
        sentences: List of sentences
        output_file_path: Path of the produced BINDER-compatible manifest file.

    Returns:

    """
    delete_if_exists(output_file_path)

    with open(output_file_path, 'a', encoding='utf-8') as file:
        for sentence in tqdm(sentences):
            json_line = json.dumps(create_one_line(sentence), ensure_ascii=False)
            file.write(json_line + '\n')


def preprocess_file(file_path: str | Path, output_file_name: str | Path, save_dir: str | Path) -> None:
    """
    The whole pre-processing pipeline. Converts a OIB-style manifest into a JONSLINES manifest.
    Args:
        file_path: Path to the OIB manifest.
        output_file_name: Name of the created manifest.
        save_dir: Output path.

    Returns:

    """
    logger.info(f"Pre-processing file: {file_path}")
    lines = extract_sentences(file_path)
    create_jsonlines_file(lines, os.path.join(save_dir, output_file_name))


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset-directory", "-d", type=str)
    group.add_argument("--file", "-f", type=str)

    parser.add_argument("--save-directory", "-s", required=False, type=str, default="save/")
    args = parser.parse_args()

    save_dir = Path(args.save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_directory is not None:
        data_dir = Path(args.dataset_directory)
        for split in ["train", "val"]:
            file_name = split + ".txt"
            preprocess_file(os.path.join(data_dir, file_name), f"{split}_preprocessed", save_dir)

    elif args.file is not None:
        file_path = Path(args.file)
        file_name = str(file_path).split("/")[-1].split(".txt")[0]

        preprocess_file(file_path, f"{file_name}_preprocessed", save_dir)


if __name__ == "__main__":
    main()
