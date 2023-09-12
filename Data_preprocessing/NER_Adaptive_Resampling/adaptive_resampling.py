import os
import tkinter
from collections import Counter
from math import log, sqrt, ceil
from pathlib import Path

from resampling_methods import Methods


class AdaptiveResampling:

    def __init__(self, train_file_path: str, resampling_directory: str, sep="\t"):
        """
        This will create an Adaptive Resampler following the original implementation in
        https://github.com/XiaoChen-W/NER_Adaptive_Resampling.
        :param train_file_path: The input NER dataset file to be resampled
        :param resampling_directory: Output folder where the resampled files will be written.
        """

        self.train_file_path = train_file_path
        self.resampling_directory = resampling_directory
        Path(resampling_directory).mkdir(parents=True, exist_ok=True)
        self.sep = sep
        
    def _extract_sentences(self, file_path):
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

    def _extract_element(self, sentence: str, position: int) -> list[str]:
        """
        This method parses a sentences according to the separator specified when instantiating this class.
        The element can be either a token (pos = 0) or a tag (pos = -1). However, this may vary depending on the
        dataset used.
        :param sentence: sentence to be parsed.
        :param position: position of the element to be extracted.
        :return: A list of the extracted elements.
        """
        elements = []
        for token_with_label in sentence.split('\n'):
            token = token_with_label.split(self.sep)[position]
            elements.append(token)
        return elements

    def _parse_dataset(self) -> (list[str], list[str]):
        """
        This method will parse the dataset and return the tokens with their corresponding tags.
        The parsing will be done according to the separator specified when instantiating the resampling class.
        :return: tagged tokens: tuple
        """
        lines = self._extract_sentences(self.train_file_path)
        tokens = list()
        tags = list()
        for line in lines:
            tokens.append(self._extract_element(line, 0))
            tags.append(self._extract_element(line, -1))

        return tokens, tags

    def compute_statistics(self):
        """
        Compute dataset statistics: class distributions.
        The "O" (Other) type tokens are ignored.
        :return: Entity proportions: dict
        """
        # Get stats of the class distribution of the dataset
        labels = list(tkinter._flatten(self._parse_dataset()[-1]))
        num_tokens = len(labels)
        ent = [label[2:] for label in labels if label != 'O']
        count_ent = Counter(ent)
        for key in count_ent:
            # Use frequency instead of count
            count_ent[key] = count_ent[key] / num_tokens
        return count_ent

    def resample(self, method: Methods):
        """
        Select method by setting hyperparameters listed below:

        - sc: the smoothed resampling incorporating count
        - sCR: the smoothed resampling incorporating Count & Rareness
        - sCRD: the smoothed resampling incorporating Count, Rareness, and Density
        - nsCRD: the normalized and smoothed  resampling  incorporating Count, Rareness, and Density
        :param method: Resampling method as described.
        :return: None
        """

        if method not in Methods:
            raise ValueError("Unidentified Resampling Method")

        filename = os.path.join(self.resampling_directory, f"{method}.txt")
        output = open(filename, 'w', encoding='utf-8')
        tokens, tags = self._parse_dataset()
        stats = self.compute_statistics()

        for sen in range(len(tokens)):
            # Resampling time can at least be 1, which means sentence without 
            # entity will be reserved in the dataset  
            rsp_time = 1
            sen_len = len(tags[sen])
            entities = Counter([label[2:] for label in tags[sen] if label != 'O'])
            # Pass if there's no entity in a sentence
            if entities:
                for ent in entities.keys():
                    # Resampling method selection and resampling time calculation, 
                    # see section 'Resampling Functions' in our paper for details.
                    if method == Methods.sC:
                        rsp_time += entities[ent]
                    if method == Methods.sCR or method == Methods.sCRD:
                        weight = -log(stats[ent], 2)
                        rsp_time += entities[ent] * weight
                    if method == Methods.nsCRD:
                        weight = -log(stats[ent], 2)
                        rsp_time += sqrt(entities[ent]) * weight
                if method == Methods.sCR:
                    rsp_time = sqrt(rsp_time)
                if method == Methods.sCRD or method == Methods.nsCRD:
                    rsp_time = rsp_time / sqrt(sen_len)
                # Ceiling to ensure the integrity of resampling time
                rsp_time = ceil(rsp_time)
            for _ in range(rsp_time):
                for token in range(sen_len):
                    output.write(tokens[sen][token] + self.sep + tags[sen][token] + '\n')
                output.write('\n')
        output.close()
