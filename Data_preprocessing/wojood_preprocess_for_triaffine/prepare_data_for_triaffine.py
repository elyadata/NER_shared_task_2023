import json
import os
import argparse
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def extract_sentences(file_path):
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


def create_entity_mentions_dict(entity_type, start, end, text):
    entity_mention = dict()
    entity_mention["start"] = start
    entity_mention["end"] = end
    entity_mention["type"] = entity_type
    entity_mention["text"] = text
    return entity_mention


def extract_tokens_from_sentence(sentence):
    tokens_with_labels = sentence.split('\n')

    intermediates = []
    spans = []
    completed_spans = []

    for index in range(len(tokens_with_labels)):       
        token_with_label = tokens_with_labels[index]
        labels = token_with_label.split("\t")[1].split(" ")

        spans.extend([(tag, index, index + 1) for tag in labels if "B-" in tag])
        intermediates.extend([(tag, index) for tag in labels if "I-" in tag])

    for span in spans:
        entity = span[0].split("B-")[-1]
        index = span[2]
        while ("I-" + entity, index) in intermediates:
            span = (entity, span[1], index + 1)
            index += 1

        completed_spans.append(span)

    return completed_spans


def extract_tokens(sentence):
    tokens = []
    for token_with_label in sentence.split('\n'):
        token = token_with_label.split("\t")[0]
        tokens.append(token)
    return tokens


def create_entity_mentions(spans, tokens):
    entity_mentions = []
    for span in spans:
        start = span[1]
        end = span[2]
        text = " ".join(tokens[start:end])

        entity_mentions.append(create_entity_mentions_dict(span[0].replace("B-", ""), start, end, text))

    return entity_mentions


def create_one_line(sentence, ltokens, rtokens):
    json_line = dict()
    completed_spans = extract_tokens_from_sentence(sentence)
    tokens = extract_tokens(sentence)
    entity_mentions = create_entity_mentions(completed_spans, tokens)
    json_line["tokens"] = tokens
    json_line["entities"] = entity_mentions
    json_line["ltokens"] = ltokens
    json_line["rtokens"] = rtokens
    json_line["org_id"] = "placeholder"
    json_line["pos"] = extract_pos(sentence)
    return json_line

def extract_pos(sentence)->list:
    sw = stopwords.words('arabic')
    tokens = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(tokens)
    # stopped_tokens = [i for i in tokens if not i in sw]
    # print(tags)
    pos = [tup[1] for tup in tags]
    # print(pos)
    return pos

def delete_if_exists(file):
    if os.path.exists(file):
        os.remove(file)


def create_jsonlines_file(sentences, split):
    json_filename = split + ".jsonlines"
    delete_if_exists(json_filename)

    with open(json_filename, 'a', encoding='utf-8') as file:
        for i in tqdm(range(len(sentences))):
            if i < len(sentences) - 1:
                rtokens = extract_tokens(sentences[i + 1])
            else:
                rtokens = []
            if i > 0:
                ltokens = extract_tokens(sentences[i - 1])
            else:
                ltokens = []
            json_line = create_one_line(sentences[i], ltokens, rtokens)
            file.write(json_line + '\n')
            
            
def create_json_file(sentences, split, save_dir):
    json_filename = split + ".json"
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
                    
              
def preprocess_file(file_path, output_file_name, save_dir) -> None:
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
    nltk.download('stopwords')

    if args.dataset_directory is not None:  
        data_dir = Path(args.dataset_directory)
        for split in ["train", "val"]:
            file_name = split + ".txt"
            preprocess_file(os.path.join(data_dir, file_name), f"{split}_preprocessed", save_dir)

    elif args.file is not None:
        file_path = Path(args.file)
        file_name = str(file_path).split("/")[-1].split(".txt")[0]
        
        preprocess_file(file_path, f"{file_name}_preprocessed", save_dir)
    # extract_pos("صورة عملة ورقية من فئة ملز خلال فترة الانتداب البريطاني على فلسطين")


if __name__ == "__main__":
    main()