from dotenv import load_dotenv
import os
import sys
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from utils.config_loader import load_config
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException


log = CustomLogger().get_logger(__name__)  # Initialize the custom logger

class ModelLoader():
    """
    Class to load models and configurations for the Document Portal application.
    """
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        self.config = load_config()  # Load configuration settings
        log.info("Configuration loaded successfully", config_keys=list(self.config.keys()))  # Log the successful loading of configuration


    def _validate_env(self):
        """        
        Validate that all required environment variables are set.
        """
        required_env_vars = ['GOOGLE_API_KEY', 'GROQ_API_KEY', 'OPENAI_API_KEY'] # List of required environment variables
        self.api_keys = {key: os.getenv(key) for key in required_env_vars} # Get the values of the required environment variables
        missing_keys = [value for key, value in self.api_keys.items() if not value] # Check for missing environment variables
        if missing_keys:
            log.error("Missing required environment variables", missing_keys=missing_keys)
            raise DocumentPortalException(f"Missing required environment variables: {', '.join(missing_keys)}", sys)
        log.info("All required environment variables are set", env_vars=required_env_vars)
        

    def load_embeddings(self):
        """
        Load embeddings based on the configuration settings.
        """
        try:
            log.info("Loading embedding model ...")
            model_name = self.config["embedding_model"]["model_name"] # Get the model name from the configuration
            return GoogleGenerativeAIEmbeddings(model=model_name)
        except Exception as e:
            log.error("Failed to load embedding model", error=str(e))
            raise DocumentPortalException(f"Failed to load embedding model: {str(e)}", sys)


    def load_llm(self):
        """
        Load chat model based on the configuration settings.
        """

        try:
            log.info("Loading chat model ...")
            llm_block = self.config["llm"]
            provider_key = os.getenv("PROVIDER_NAME", "groq")  # Default to Groq if not set

            if provider_key not in llm_block:
                log.error("Provider key not found in configuration", provider_key=provider_key)
                raise ValueError(f"Provider key '{provider_key}' not found in configuration", sys)
            
            llm_config = llm_block[provider_key] # Get the configuration for the specified provider
            provider = llm_config.get("provider") # Get the provider name from the configuration
            model_name = llm_config.get("model_name") # Get the model name from the configuration
            temperature = llm_config.get("temperature", 0.2) # Get the temperature from the configuration
            max_tokens = llm_config.get("max_tokens", 1024) # Get the maximum tokens from the configuration

            log.info("Chat model configuration", provider=provider, model_name=model_name, temperature=temperature, max_tokens=max_tokens)

            if provider == "google":
                llm = ChatGoogleGenerativeAI(temperature=temperature, max_tokens=max_tokens, model=model_name)
                return llm
            elif provider == "groq":
                llm = ChatGroq(temperature=temperature, max_tokens=max_tokens, model=model_name)
                return llm
            elif provider == "open_ai":  
                llm = ChatOpenAI(temperature=temperature, max_tokens=max_tokens, model=model_name)
                return llm
            else:
                log.error("Unsupported provider", provider=provider)
                raise DocumentPortalException(f"Unsupported provider: {provider}", sys)


        
        except KeyError as e:
            log.error("Configuration key missing", error=str(e))
            raise DocumentPortalException(f"Configuration key missing: {str(e)}", sys)


if __name__ == "__main__":
    loader = ModelLoader()
    
    # Test embedding model loading
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    
    # Test the ModelLoader
    # result=embeddings.embed_query("Hello, how are you?")
    # print(f"Embedding Result: {result}")
    
    # Test LLM loading based on YAML config
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    
    # Test the ModelLoader
    result=llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")