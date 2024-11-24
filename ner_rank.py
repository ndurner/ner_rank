import argparse
import logging
from typing import List, Dict
from datasets import load_dataset
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from gliner import GLiNER
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import spacy
from flair.models import SequenceTagger
from flair.data import Sentence
from presidio_analyzer.predefined_recognizers import SpacyRecognizer
from spacy_gliner_nlp_engine import SpacyGlinerNlpEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the tag map
TAG_MAP = {
    0: 'O', 1: 'B-PER', 2: 'I-PER', 
    3: 'B-ORG', 4: 'I-ORG',
    5: 'B-LOC', 6: 'I-LOC',
    7: 'B-MISC', 8: 'I-MISC'
}

from presidio_analyzer.nlp_engine import NlpEngineProvider

def load_presidio_analyzer(backend: str) -> AnalyzerEngine:
    if backend == "spacy":
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "de", "model_name": "de_core_news_lg"}],
            "labels_to_ignore": [""],
        }
    elif backend == "flair":
        configuration = {
            "nlp_engine_name": "flair",
            "models": [{"lang_code": "en", "model_name": "flair/ner-english-large"}],
        }
    elif backend == "huggingface":
        configuration = {
            "nlp_engine_name": "transformers",
            "models": [
                {
                    "lang_code": "de",
                    "model_name": {"spacy": "de_core_news_lg", "transformers": "obi/deid_roberta_i2b2"},
                }
            ],
        }
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Create NlpEngine using the configuration
    # Spacy:
    # nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
    # Gliner direct
    #nlp_engine = NlpEngineProvider(conf_file="language-config.yml").create_engine()
    # gliner-spacy
    nlp_engine = SpacyGlinerNlpEngine(models=[{"lang_code": "de", "model_name": "de_core_news_lg"}])

    spacy_reco = SpacyRecognizer(supported_language="de")

    registry = RecognizerRegistry(supported_languages=["de"])
    registry.add_recognizer(spacy_reco)


    # Create AnalyzerEngine with the NlpEngine
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=["de"])
    supported_entities = analyzer.get_supported_entities(language='de')
    print(f"Supported entities for de: {supported_entities}")
    return analyzer

def load_gliner_model():
    return GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

def load_llm_model(model_name: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map=device,
    )
    
    return pipe

def redact_with_presidio(text: str, analyzer: AnalyzerEngine) -> str:
    analyzer_results = analyzer.analyze(text=text, language='de', entities=["PERSON", "ORGANIZATION"])
    anonymizer = AnonymizerEngine()
    anonymized_text = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        # operators={"DEFAULT": OperatorConfig("replace", {"new_value": lambda ent: f"[{ent.entity_type}]"})}
    )
    return anonymized_text.text

def redact_with_gliner(text: str, model: GLiNER) -> str:
    entities = model.predict_entities(text, labels=["person", "organization"])
    for entity in reversed(entities):
        start, end = entity["start"], entity["end"]
        label = entity["label"].upper()
        text = text[:start] + f"[{label}]" + text[end:]
    return text

def redact_with_llm(text: str, pipe) -> str:
    prompt = f"""You are a natural-entity and PII redaction system. Redact any names using [PERSON], redact any organization names with [ORGANISATION].
    Example #1:
    ---
    Input: Anneliese works at IBM.
    Output: [PERSON] works at [ORGANISATION].
    ---

    Example #2
    ---
    Input: Back in March, Peter worked at Google.
    Output: Back in March, [PERSON] worked at [ORGANISATION].
    ---

    Avoid returning anything extra in addition to the redacted text. Ensure word-by-word faithfulness aside from the redactions.

    Redact entities from the following text: "{text}"
    """
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    outputs = pipe(messages, max_new_tokens=256)
    
    # Extract the generated text
    generated_text = outputs[0]["generated_text"][-1]['content']
    
    return generated_text

def evaluate_redaction(original_text: str, redacted_text: str, ner_tags: List[int]) -> bool:
    entities_to_redact = []
    current_entity = ""
    logger.debug(f"zip: {list(zip(original_text.split(), ner_tags))}\n")
    for token, tag in zip(original_text.split(), ner_tags):
        if TAG_MAP[tag] in ['B-PER', 'I-PER', 'B-ORG', 'I-ORG']:
            current_entity += " " + token if current_entity else token
        elif current_entity:
            entities_to_redact.append(current_entity.strip())
            current_entity = ""
    
    if current_entity:  # Add the last entity if it exists
        entities_to_redact.append(current_entity.strip())

    for entity in entities_to_redact:
        if entity in redacted_text:
            logger.debug(f"residual token: {entity}")
            return False  # Entity that should be redacted is still present
    
    return True  # All entities that should be redacted are not present in the redacted text

def run_evaluation(backends: List[str], llm_model: str, debug: bool) -> Dict[str, float]:
    dataset = load_dataset("ndurner/german-ner", split="train")
    results = {}

    # Load the LLM model once if it's in the backends
    llm_pipe = None
    if "llm" in backends:
        llm_pipe = load_llm_model(llm_model)

    for backend in backends:
        if backend.startswith("presidio_"):
            analyzer = load_presidio_analyzer(backend.split("_")[1])
            redact_func = lambda text: redact_with_presidio(text, analyzer)
        elif backend == "gliner":
            model = load_gliner_model()
            redact_func = lambda text: redact_with_gliner(text, model)
        elif backend == "llm":
            redact_func = lambda text: redact_with_llm(text, llm_pipe)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        total_samples = 0
        correct_redactions = 0

        for sample in dataset:
            original_text = " ".join(sample['tokens'])
            ner_tags = sample['ner_tags']
            
            redacted_text = redact_func(original_text)
            
            if evaluate_redaction(original_text, redacted_text, ner_tags):
                correct_redactions += 1
            
            total_samples += 1
            
            if debug:
                logger.debug(f"Backend: {backend}")
                logger.debug(f"Original: {original_text}")
                logger.debug(f"Redacted: {redacted_text}")
                logger.debug(f"Success: {evaluate_redaction(original_text, redacted_text, ner_tags)}")
                logger.debug("---")

        accuracy = correct_redactions / total_samples
        results[backend] = accuracy

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate NER redaction methods")
    parser.add_argument("--backends", nargs="+", choices=["presidio_spacy", "presidio_flair", "presidio_huggingface", "gliner", "llm"], required=True)
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="LLM model to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    results = run_evaluation(args.backends, args.llm_model, args.debug)
    
    print("Evaluation Results:")
    for backend, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{backend}: {accuracy:.2%}")

if __name__ == "__main__":
    main()