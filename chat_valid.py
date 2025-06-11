from datetime import date
import dateparser
from dataclasses import dataclass, field, replace
from pydantic import BaseModel, Field
from pydantic.functional_validators import AfterValidator
from typing import Annotated, Optional
import ollama
from openai import OpenAI
import instructor

def validate_and_clean_date(input_date: str) -> date:
    pass

def validate_full_name(input_name: str) -> str:
    pass

def validate_nationality(input_nationality: str) -> str:
    pass

class GuarantorInfo(BaseModel):
    name: str

class LetterOfGuarantee(BaseModel):
    date: Annotated[
        Optional[str],
        Field(description="Date on which this form is being filled up."),
        AfterValidator(validate_and_clean_date)
    ] = None

    name: Annotated[
        Optional[str],
        Field(description="Full name of the person filling the form."),
        AfterValidator(validate_full_name)
    ] = None

    nationality: Annotated[
        Optional[str],
        Field(description="Nationality of the person filling the form."),
        AfterValidator(validate_nationality)
    ] = None

class ChatInfo(BaseModel):
    field: str = Field(description="The field that is captured from the chat.")
    is_valid: bool = Field(description="Whether this is valid.")
    error_message: Optional[str] = Field(description="Error message, if any.")

@dataclass
class FormFields:
    form_date: date
    full_name: str
    nationality: str

    def add_field(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)

class ChatBot:
    def __init__(self):
        self.model_name = "llama3.1:8b"
        self.instructor_client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )
        self.form_data = {
            "date" : {
                "base_prompt" : "What date do you want to put on this form?",
                "description" : "Date for this form.",
                "validation_rule" : "Should be a valid date in any format."
            },
            "full_name" : {
                "base_prompt" : "What is your full name?",
                "description" : "Full name of the person filling this form.",
                "validation_rule" : "Must have first name and last name."
            },
            "nationality" : {
                "base_prompt" : "What is your nationality?",
                "description" : "Nationality of the person filling this form.",
                "validation_rule" : "Check against a list of countries if the input is a valid country."
            }
        }
        self.flat_fields = list(self.form_data.keys())
        self.current_field_index = 0

    def start_conversation(self):
        prompt = """Hello! I'm here to help you fill this form.

Let's begin!"""

        field = self.flat_fields[self.current_field_index]
        print(f"{prompt}\n\n{self.form_data[field]['base_prompt']}")

    def process_user_input(self, user_input):
        # Validate the user input
        # If it is correct, save it and return is_completed=True with positive ack.
        # If not return False along with negative ack.
        current_field = self.flat_fields[self.current_field_index]
        field_data = self.form_data[current_field]
        response = self.instructor_client.chat.completions.create(
            model=self.model_name,
            messages = [
                {
                    "role" : "system",
                    "content" : f"""You are an expert at validating and extracting data.

Follow these instructions:
1. Return only the extracted data in the 'field' of the response model. 
2. If the user input is not valid, write a message in the 'error_message' of the response model, and set is_valid to False.

Description of the user input: {field_data['description']}
Validation rules of the user input: {field_data['validation_rule']}
"""
                },
                {
                    "role" : "user",
                    "content" : f"{user_input}"
                }
            ],
            response_model=ChatInfo
        )
        
        if response.is_valid:
            print(f"Thank you! The date has been saved as {response.field}. {dateparser.parse(response.field)}.")
        else:
            print(f"""That did not work out quite well for the following reason:
{response.error_message}

Lets try again. {field_data['base_prompt']}""")
        print(response)
        is_complete = False

        return response, is_complete

    def get_collected_data(self):
        pass

if __name__ == "__main__":
    chatbot = ChatBot()

    chatbot.start_conversation()

    is_complete = False
    while not is_complete:
        user_input = input("You: ")
        response, is_complete = chatbot.process_user_input(user_input)
        print("Bot: ", response)

    final_data = chatbot.get_collected_data()
    print(final_data)