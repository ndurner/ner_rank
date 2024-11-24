import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

from presidio_analyzer.nlp_engine import NerModelConfiguration, NlpArtifacts, NlpEngine

logger = logging.getLogger("presidio-analyzer")


class SpacyGlinerNlpEngine(NlpEngine):
    """
    SpacyGlinerNlpEngine is an abstraction layer over the nlp module.

    It provides processing functionality as well as other queries
    on tokens.
    The SpacyGlinerNlpEngine uses SpaCy with GLiNER integration as its NLP module
    """

    engine_name = "spacy_gliner"
    is_available = bool(spacy)

    def __init__(
        self,
        models: Optional[List[Dict[str, str]]] = None,
        ner_model_configuration: Optional[NerModelConfiguration] = None,
        gliner_model: str = "urchade/gliner_multi-v2.1",
    ):
        """
        Initialize a wrapper on spaCy functionality with GLiNER integration.

        :param models: Dictionary with the name of the spaCy model per language.
        For example: models = [{"lang_code": "en", "model_name": "en_core_web_lg"}]
        :param ner_model_configuration: Parameters for the NER model.
        :param gliner_model: The GLiNER model to be used.
        """
        if not models:
            models = [{"lang_code": "en", "model_name": "en_core_web_lg"}]
        self.models = models

        if not ner_model_configuration:
            ner_model_configuration = NerModelConfiguration()
        self.ner_model_configuration = ner_model_configuration
        self.ner_model_configuration.labels_to_ignore.remove("ORGANIZATION")

        self.gliner_model = gliner_model
        self.nlp = None

    def load(self) -> None:
        """Load the spaCy NLP model with GLiNER integration."""
        logger.debug(f"Loading SpaCy models with GLiNER: {self.models}")

        self.nlp = {}
        for model in self.models:
            self._validate_model_params(model)
            self._download_spacy_model_if_needed(model["model_name"])
            nlp = spacy.load(model["model_name"])
            
            # Add GLiNER to the pipeline
            gliner_config = {
                "gliner_model": self.gliner_model,
                "chunk_size": 250,  # You can adjust this
                "labels": self.get_supported_entities(),
                "style": "ent"
            }
            nlp.add_pipe("gliner_spacy", config=gliner_config)
            
            self.nlp[model["lang_code"]] = nlp

    @staticmethod
    def _download_spacy_model_if_needed(model_name: str) -> None:
        if not (spacy.util.is_package(model_name) or Path(model_name).exists()):
            logger.warning(f"Model {model_name} is not installed. Downloading...")
            spacy.cli.download(model_name)
            logger.info(f"Finished downloading model {model_name}")

    @staticmethod
    def _validate_model_params(model: Dict) -> None:
        if "lang_code" not in model:
            raise ValueError("lang_code is missing from model configuration")
        if "model_name" not in model:
            raise ValueError("model_name is missing from model configuration")
        if not isinstance(model["model_name"], str):
            raise ValueError("model_name must be a string")

    def get_supported_entities(self) -> List[str]:
        """Return the supported entities for this NLP engine."""
        # You can customize this list based on the entities you want to recognize
        return ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "MONEY", "PERCENT", "FACILITY", "PRODUCT"]

    def get_supported_languages(self) -> List[str]:
        """Return the supported languages for this NLP engine."""
        if not self.nlp:
            raise ValueError("NLP engine is not loaded. Consider calling .load()")
        return list(self.nlp.keys())

    def is_loaded(self) -> bool:
        """Return True if the model is already loaded."""
        return self.nlp is not None

    def process_text(self, text: str, language: str) -> NlpArtifacts:
        """Execute the SpaCy NLP pipeline with GLiNER on the given text and language."""
        if not self.nlp:
            raise ValueError("NLP engine is not loaded. Consider calling .load()")

        doc = self.nlp[language](text)
        return self._doc_to_nlp_artifact(doc, language)

    def process_batch(
        self,
        texts: Union[List[str], List[Tuple[str, object]]],
        language: str,
        batch_size: Optional[int] = None,
        as_tuples: bool = False,
    ) -> Iterator[Optional[NlpArtifacts]]:
        """Execute the NLP pipeline on a batch of texts."""
        if not self.nlp:
            raise ValueError("NLP engine is not loaded. Consider calling .load()")

        for text in texts:
            if as_tuples:
                text, context = text
            doc = self.nlp[language](str(text))
            yield text, self._doc_to_nlp_artifact(doc, language)

    def is_stopword(self, word: str, language: str) -> bool:
        """
        Return true if the given word is a stop word.

        (within the given language)
        """
        return self.nlp[language].vocab[word].is_stop

    def is_punct(self, word: str, language: str) -> bool:
        """
        Return true if the given word is a punctuation word.

        (within the given language).
        """
        return self.nlp[language].vocab[word].is_punct

    def get_nlp(self, language: str) -> Language:
        """
        Return the language model loaded for a language.

        :param language: Language
        :return: Model from spaCy
        """
        return self.nlp[language]

    def _doc_to_nlp_artifact(self, doc: Doc, language: str) -> NlpArtifacts:
        lemmas = [token.lemma_ for token in doc]
        tokens_indices = [token.idx for token in doc]

        entities = self._get_entities(doc)
        scores = self._get_scores_for_entities(doc)

        entities, scores = self._get_updated_entities(entities, scores)

        return NlpArtifacts(
            entities=entities,
            tokens=doc,
            tokens_indices=tokens_indices,
            lemmas=lemmas,
            nlp_engine=self,
            language=language,
            scores=scores,
        )

    def _get_entities(self, doc: Doc) -> List[Span]:
        """
        Extract entities out of a spaCy pipeline with GLiNER.

        :param doc: the output spaCy doc.
        :return: List of entities
        """
        return doc.ents

    def _get_scores_for_entities(self, doc: Doc) -> List[float]:
        """Extract scores for entities from the doc."""
        return [ent._.score for ent in doc.ents]

    def _get_updated_entities(
        self, entities: List[Span], scores: List[float]
    ) -> Tuple[List[Span], List[float]]:
        """
        Get an updated list of entities based on the ner model configuration.

        Remove entities that are in labels_to_ignore,
        update entity names based on model_to_presidio_entity_mapping

        :param entities: Entities that were extracted from a spaCy pipeline
        :param scores: Original confidence scores for the entities extracted
        :return: Tuple holding the entities and confidence scores
        """
        if len(entities) != len(scores):
            raise ValueError("Entities and scores must be the same length")

        new_entities = []
        new_scores = []

        mapping = self.ner_model_configuration.model_to_presidio_entity_mapping
        to_ignore = self.ner_model_configuration.labels_to_ignore
        for ent, score in zip(entities, scores):
            # Remove model labels in the ignore list
            if ent.label_ in to_ignore:
                continue

            # Update entity label based on mapping
            if ent.label_ in mapping:
                ent.label_ = mapping[ent.label_]
            else:
                logger.warning(
                    f"Entity {ent.label_} is not mapped to a Presidio entity, "
                    f"but keeping anyway. "
                    f"Add to `NerModelConfiguration.labels_to_ignore` to remove."
                )

            # Remove presidio entities in the ignore list
            if ent.label_ in to_ignore:
                continue

            new_entities.append(ent)

            # Update score if entity is in low score entity names
            if ent.label_ in self.ner_model_configuration.low_score_entity_names:
                score *= self.ner_model_configuration.low_confidence_score_multiplier

            new_scores.append(score)

        return new_entities, new_scores