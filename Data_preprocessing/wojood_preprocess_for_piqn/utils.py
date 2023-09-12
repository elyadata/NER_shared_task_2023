import os
from pathlib import Path


def extract_sentences(file_path: Path) -> list:
    """
    Extracts the sentences from the input file.
    :param file_path: the path of the file from which to extract the sentences
    :return: the extracted sentences
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

        if current_line:
            lines.append(current_line.strip())

    return lines


def create_entity_mentions_dict(entity_type: str, start: str, end: str, text: str) -> dict:
    """
    Creates a dict containing the metadata for one entity.
    :param entity_type: the type of the entity
    :param start: the start position of the entity
    :param end: the end position of the entity
    :param text: the text of the entity
    :return: the metadata of the entity
    """
    entity_mention = dict()
    entity_mention["start"] = start
    entity_mention["end"] = end
    entity_mention["type"] = entity_type
    entity_mention["text"] = text
    return entity_mention


def extract_spans_from_sentence(sentence: str) -> list:
    """
    Extracts the spans from the sentences.
    :param sentence: the input sentence
    :return: the extracted spans
    """
    tokens_with_labels = sentence.split('\n')

    intermediates = []
    spans = []
    completed_spans = []

    for index in range(len(tokens_with_labels)):
        token_with_label = tokens_with_labels[index]
        labels = token_with_label.split("\t")[1].split(" ")

        spans.extend([(tag, index, index + 1)
                      for tag in labels if "B-" in tag])
        intermediates.extend([(tag, index) for tag in labels if "I-" in tag])

    for span in spans:
        entity = span[0].split("B-")[-1]
        index = span[2]
        while ("I-" + entity, index) in intermediates:
            span = (entity, span[1], index + 1)
            index += 1

        completed_spans.append(span)

    return completed_spans


def create_entity_mentions(spans: list, tokens: list) -> list:
    """
    Creates a list containing the metadata for entities.
    :param spans: a list containing the spans
    :param tokens: the sentence tokens
    :return: list containing the metadata for entities
    """
    entity_mentions = []
    for span in spans:
        start = span[1]
        end = span[2]
        text = " ".join(tokens[start:end])

        entity_mentions.append(create_entity_mentions_dict(
            span[0].replace("B-", ""), start, end, text))

    return entity_mentions


def delete_if_exists(file: Path):
    """
    It deletes the file if it exists.
    :param file: the input file
    """
    if os.path.exists(file):
        os.remove(file)
