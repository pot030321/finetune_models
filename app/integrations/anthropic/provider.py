from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv('.env')
# ANTHROPIC_API_KEY

class AnthropicProvider:
    def __init__(
        self,
        model: str = 'anthropic',
        chat_model: str = 'claude-3-opus-20240229',
    ):
        self.model = model
        self.chat_model = chat_model
        self.chat_model_instance = self._get_chat_model()

    def _get_chat_model(self):
        return ChatAnthropic(model_name=self.chat_model)
    