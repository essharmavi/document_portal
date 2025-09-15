import os
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser
import sys
from prompts.prompt_library import PROMPT_REGISTRY

class DocumentAnalyzer():

    """    Class to analyze documents and extract metadata.
    """

    def __init__(self):
        self.logger = CustomLogger().get_logger(__name__)
        try:
            self.load = ModelLoader()
            self.llm = self.load.load_llm()

            self.parser = JsonOutputParser(pydantic_object= Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm = self.llm)

            self.prompt = PROMPT_REGISTRY['document_analysis']
            self.logger.info("Document Analyzer initialized successfully.")

        except Exception as e:
            self.logger.error(f"Error initializing DocumentAnalyzer: {e}")
            raise DocumentPortalException("Error in DocumentAnalyzer initialization", sys)

    def analyze_document(self, document_text:str)-> dict:
        """
        Analyze a document's text and extract structured metadata & summary.
        """
        try:
            chain = self.prompt | self.llm | self.fixing_parser
            
            self.logger.info("Meta-data analysis chain initialized")

            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text
            })

            self.logger.info("Metadata extraction successful", keys=list(response.keys()))
            
            return response

        except Exception as e:
            self.logger.error("Metadata analysis failed", error=str(e))
            raise DocumentPortalException("Metadata extraction failed", sys)
        