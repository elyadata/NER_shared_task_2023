import sys
import json
import argparse
from arabiner.bin.train import main as train
from loguru import logger

sys.path.append('..')


def main(config_file: str) -> None:
    logger.info("Start of model training.")
    # Loading config file
    with open(config_file) as file:
        config = json.load(file)
        
    # Training the model with the loaded config 
    args = argparse.Namespace()
    args.__dict__ = config
    train(args)
    
    logger.success("Trainig completed.")
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', 
                        type=str, 
                        required=False, 
                        default="arabiner/config/flat_ner.json")
    args = parser.parse_args()
    
    main(args.config)
    