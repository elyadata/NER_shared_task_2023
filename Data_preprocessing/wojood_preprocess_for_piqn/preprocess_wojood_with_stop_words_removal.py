import argparse
import json
import os
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from utils import extract_spans_from_sentence, create_entity_mentions, delete_if_exists, extract_sentences


def extract_tokens(sentence: str, stop_words: list) -> list:
    """
    Extracts from the sentences only the tokens that are not stop words.
    :param sentence: the input sentence from which to extract the token.
    :param stop_words: the list of the stop words.
    :return: the extracted tokens
    """
    tokens = []
    for token_with_label in sentence.split('\n'):
        token = token_with_label.split("\t")[0]
        if token not in stop_words:
            tokens.append(token)
    return tokens


def extract_stop_words(stop_words_file: Path) -> list:
    """
    Extracts the stop words from the input file.
    :param stop_words_file: the file from which to extract the stop words.
    :return: the list of the stop words.
    """
    with open(stop_words_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def create_one_line(sentence: str, ltokens: list, rtokens: list, stop_words: list) -> dict:
    """
    Creates a single json line for the manifest file to be created.
    :param sentence: the sentences from which to extract the tokens and the entities
    :param ltokens: left tokens
    :param rtokens: right tokens
    :param stop_words: the list of the stop words
    :return: the dict containing the needed data
    """
    json_line = dict()
    completed_spans = extract_spans_from_sentence(sentence)
    tokens = extract_tokens(sentence, stop_words)
    entity_mentions = create_entity_mentions(completed_spans, tokens)
    json_line["tokens"] = tokens
    json_line["entities"] = entity_mentions
    json_line["ltokens"] = ltokens
    json_line["rtokens"] = rtokens
    json_line["org_id"] = "placeholder"
    json_line["relations"] = []

    return json_line


def create_json_file(sentences: str, output_file_name: str, save_dir: Path, stop_words: list):
    """
    Generates a JSON file compatible with PIQN.
    :param sentences: the sentences from which to extract the tokens and the entities
    :param output_file_name: the output file name
    :param save_dir: the directory where to save the generates files
    :param stop_words: the list of the stop words
    """
    json_filename = output_file_name + ".json"
    json_file_path = os.path.join(save_dir, json_filename)
    delete_if_exists(json_file_path)
    contents = []

    for i in tqdm(range(len(sentences))):
        if i < len(sentences) - 1:
            rtokens = extract_tokens(sentences[i + 1], stop_words)
        else:
            rtokens = []
        if i > 0:
            ltokens = extract_tokens(sentences[i - 1], stop_words)
        else:
            ltokens = []
        contents.append(create_one_line(
            sentences[i], ltokens, rtokens, stop_words))

    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(contents, file, ensure_ascii=False)


def preprocess_file(file_path: str, output_file_name: str, save_dir: Path, stop_words: list) -> None:
    """
    Performs files preprocessing.
    :param file_path: the path to the file
    :param output_file_name: the name of the output file
    :param save_dir: the directory where to save the preprocessed files
    :param stop_words:  the list of the stop words
    """
    logger.info(f"Pre-processing file: {file_path}")
    lines = extract_sentences(file_path)
    create_json_file(lines, output_file_name, save_dir, stop_words)


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset-directory", "-d", type=str)
    group.add_argument("--file", "-f", type=str)
    parser.add_argument("--stop-words-file", "-w", required=True, type=str)
    parser.add_argument("--save-directory", "-s",
                        required=False, type=str, default="save/")
    args = parser.parse_args()

    save_dir = Path(args.save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    stop_words_file = Path(args.stop_words_file)
    stop_words = extract_stop_words(stop_words_file)
    if args.dataset_directory is not None:
        data_dir = Path(args.dataset_directory)
        for split in ["train", "val"]:
            file_name = split + ".txt"
            preprocess_file(os.path.join(data_dir, file_name),
                            f"{split}_preprocessed", save_dir, stop_words)

    elif args.file is not None:
        file_path = Path(args.file)
        file_name = str(file_path).split("/")[-1].split(".txt")[0]

        preprocess_file(
            file_path, f"{file_name}_preprocessed", save_dir, stop_words)


if __name__ == "__main__":
    main()
