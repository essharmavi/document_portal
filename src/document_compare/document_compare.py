import sys
from dotenv import load_dotenv
import pandas as pd
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from prompts.prompt_library import PROMPT_REGISTRY
from utils.model_loader import ModelLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser


class DocumentComparerLLM():
    def __init__(self):
        load_dotenv()
        self.logger = CustomLogger().get_logger(__name__)
        self.model = ModelLoader()
        self.llm = self.model.load_llm()
        self.output_parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.output_fixer = OutputFixingParser.from_llm(llm = self.llm, parser = self.output_parser)
        self.prompt = PROMPT_REGISTRY['document_comparison']
        # self.chain = self.prompt | self.llm | self.output_parser | self.output_fixer
        self.chain = self.prompt | self.llm | self.output_parser
        self.logger.info("DocumentComparerLLM initialized with model and prompt.")

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instruction": self.output_parser.get_format_instructions()
            }
            print("**"*100)
            print(inputs)
            self.logger.info("Starting document comparison with inputs: %s", inputs)
            response = self.chain.invoke(inputs)
            self.logger.info("Document comparison completed successfully.")

            return self.format_response(response)

        except Exception as e:
            self.logger.error(f"Error in compare_documents: {str(e)}")
            raise DocumentPortalException("An error occurred while comparing documents.", sys)

    def format_response(self, response: list[dict]) -> pd.DataFrame:
        try:
            df = pd.DataFrame(response)
            self.logger.info("Response formatted into DataFrame successfully.", dataframe=df)
            return df
        except Exception as e:
            self.logger.error(f"Error in format_response: {str(e)}")
            raise DocumentPortalException("An error occurred while formatting the response.", sys)
