import argparse
import json
import os
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from utils import extract_spans_from_sentence, create_entity_mentions, delete_if_exists, extract_sentences


def extract_tokens(sentence):
    """
        Extracts tokens from sentences.
        :param sentence: the input sentence from which to extract the token.
        :return: the extracted tokens
        """
    tokens = []
    for token_with_label in sentence.split('\n'):
        token = token_with_label.split("\t")[0]
        tokens.append(token)
    return tokens


def create_one_line(sentence: str, ltokens: list, rtokens: list)->dict:
    """
      Creates a single json line for the manifest file to be created.
      :param sentence: the sentences from which to extract the tokens and the entities
      :param ltokens: left tokens
      :param rtokens: right tokens
      :return: the dict containing the needed data
      """
    json_line = dict()
    completed_spans = extract_spans_from_sentence(sentence)
    tokens = extract_tokens(sentence)
    entity_mentions = create_entity_mentions(completed_spans, tokens)
    json_line["tokens"] = tokens
    json_line["entities"] = entity_mentions
    json_line["ltokens"] = ltokens
    json_line["rtokens"] = rtokens
    json_line["org_id"] = "placeholder"
    json_line["relations"] = []

    return json_line


def create_json_file(sentences: str, output_file_name: str, save_dir: Path):
    """
    Generates a JSON file compatible with PIQN.
    :param sentences: the sentences from which to extract the tokens and the entities
    :param output_file_name: the output file name
    :param save_dir: the directory where to save the generates files
    """
    json_filename = output_file_name + ".json"
    json_file_path = os.path.join(save_dir, json_filename)
    delete_if_exists(json_file_path)
    contents = []

    for i in tqdm(range(len(sentences))):
        if i < len(sentences) - 1:
            rtokens = extract_tokens(sentences[i + 1])
        else:
            rtokens = []
        if i > 0:
            ltokens = extract_tokens(sentences[i - 1])
        else:
            ltokens = []
        contents.append(create_one_line(sentences[i], ltokens, rtokens))

    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(contents, file, ensure_ascii=False)


def preprocess_file(file_path: str, output_file_name: str, save_dir: Path) -> None:
    """
    Performs files preprocessing.
    :param file_path: the path to the file
    :param output_file_name: the name of the output file
    :param save_dir: the directory where to save the preprocessed files
    """
    logger.info(f"Pre-processing file: {file_path}")
    lines = extract_sentences(file_path)
    create_json_file(lines, output_file_name, save_dir)


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
