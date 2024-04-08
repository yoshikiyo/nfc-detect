import argparse

from .config import ClassifierConfig

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Example script with commands and flags")

    parser.add_argument("command", help="The command to execute")

    # Define flags (optional arguments)
    parser.add_argument('--cls_config', type=str, help='Classifier config file.')
    parser.add_argument('--input', type=str, help='Path to input audio file.')

    args = parser.parse_args()

    # Perform actions based on command and flags
    if args.command == 'classify':
        with open(args.cls_config, 'r') as f:
            json_data = f.read()
        cls_config = ClassifierConfig.model_validate_json(json_data)
        classifier = classifier.EmbeddingClassifier(cls_config)

    # Unknown command
    else:
        print(f'Invalid command: {args.command}')


# Run only if script is executed directly
if __name__ == "__main__":
    main()
