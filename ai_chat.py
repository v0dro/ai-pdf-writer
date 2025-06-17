import phonenumbers
import dateparser
from pydantic import BaseModel, Field
from typing import  Optional
from openai import OpenAI # needed only for API conformity for instructor.
import instructor

class ChatInfo(BaseModel):
    """
    Model to capture chat information.
    """
    field: str = Field(description="The field that is captured from the chat.")
    is_valid: bool = Field(description="Whether this is valid.")
    error_message: Optional[str] = Field(description="Error message, if any.")

class ChatBot:
    """
    A chatbot to help fill out a letter of guarantee form. 
    Uses the Instructor API to validate and extract data from user input with a Llama 3.1-8b model.
    It will ask questions one by one, validate the input, and save the data.
    """
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
                "validation_rule" : "Should be a valid date in any format.",
            },
            "full_name" : {
                "base_prompt" : "What is your full name?",
                "description" : "Full name of the person filling this form.",
                "validation_rule" : "Must have first name and last name."
            },
            "nationality" : {
                "base_prompt" : "What is your nationality?",
                "description" : "Nationality of the person filling this form.",
                "validation_rule" : "Check against a list of countries if the input is a valid country. "
                                    "Convert adjective to noun if needed, e.g. 'Japanese' to 'Japan', or 'American' to 'United States of America'. "
                                    "The name of the country should be a valid name in English.",
            },
            "guarantor" : {
                "nested" : True,
                "name": {
                    "description": "The full name of the guarantor.",
                    "base_prompt": "Could you please provide the full name of your guarantor?",
                    "validation_rule" : "The name must be a non-empty string. It should at least contain a first name and a last name."
                },
                "address_in_japan": {
                    "description": "The address of the guarantor in Japan.",
                    "base_prompt": "What is the address of your guarantor in Japan? Please provide the full address including postal code.",
                    "validation_rule" : "The address should be a valid address in Japan in the format Postal Code, Prefecture, City, Building Name (if applicable)."
                },
                "guarantor_phone_number": {
                    "description": "The phone number of the guarantor in Japan.",
                    "base_prompt": "What is your guarantor's phone number in Japan?",
                    "validation_rule" :"""1. The phone number should be a valid Japanese phone number. It should start with a country code +81 or 0, followed by 9 to 11 digits.
2. For 'field': Return the EXACT input with only formatting changes if needed (like adding hyphens)
3. If you need to reformat, only add separators - NEVER change digits
4. If the input is invalid, set is_valid to False but still preserve the original digits in 'field'"""
                },
                "place_of_employment": {
                    "description": "The place of employment of the guarantor.",
                    "base_prompt": "Where does your guarantor work?",
                    "validation_rule" : "The place of employment can be a company name, organization, or institution. It should not be empty."
                },
                "occupation_phone_number": {
                    "description": "The phone number of the guarantor's place of employment.",
                    "base_prompt": "What is the phone number of your guarantor's place of employment?",
                    "validation_rule" :"""1. The phone number should be a valid Japanese phone number. It should start with a country code +81 or 0, followed by 9 to 11 digits.
2. For 'field': Return the EXACT input with only formatting changes if needed (like adding hyphens)
3. If you need to reformat, only add separators - NEVER change digits
4. If the input is invalid, set is_valid to False but still preserve the original digits in 'field'"""
                },
                "nationality": {
                    "description": "The nationality of the guarantor.",
                    "base_prompt": "What is your guarantor's nationality?",
                    "validation_rule" : "Check against a list of countries if the input is a valid country. "
                                        "Convert adjective to noun if needed, e.g. 'Japanese' to 'Japan', or 'American' to 'United States of America'. "
                                        "The name of the country should be a valid name in English.",
                },
                "status_of_residence": {
                    "description": "The status of residence of the guarantor.",
                    "base_prompt": "What is your guarantor's status of residence in Japan?",
                    "validation_rule" : "Check against a list of valid statuses of residence in Japan. It should be a valid status such as 'Permanent Resident', 'Student', 'Work Visa', etc. If the guarantor is not a Japanese citizen, it should be specified."
                },
                "period_of_stay": {
                    "description": "The period of stay of the guarantor if they are not a Japanese citizen.",
                    "base_prompt": "If your guarantor is not a Japanese citizen, what is their period of stay in Japan? Please provide the start and end dates.",
                    "validation_rule" : "Should be a valid date range. If the guarantor is a Japanese citizen, this field can be left empty."
                },
                "guarantor_relationship": {
                    "description": "The relationship of the guarantor to the user.",
                    "base_prompt": "What is your relationship with the guarantor?",
                    "validation_rule" : "The relationship should be a valid relationship such as 'Parent', 'Sibling', 'Friend', 'Colleague', etc. It should not be empty."
                }   
            }
        }
        self.flat_fields = self._parse_form_fields("", self.form_data)
        self.saved_info = dict()
        self.current_field_index = 0

    def _parse_form_fields(self, prefix, form_dict):
        """
        Recursively flattens the form fields, including nested fields, into a list of dot-separated field names.

        Args:
            prefix (str): The prefix for nested fields.
            form_dict (dict): The dictionary representing the form structure.

        Returns:
            list: A list of flattened field names.
        """
        flat_fields = list()
        for k, v in form_dict.items():
            if isinstance(v, dict) and v.get('nested'):
                flat_fields += self._parse_form_fields(f"{prefix}{k}.", form_dict[k])
            elif k != "nested":
                flat_fields.append(prefix + k)

        return flat_fields

    def _find_form_data(self, current_field):
        """
        Retrieves the form field metadata dictionary for a given dot-separated field name.

        Args:
            current_field (str): The dot-separated field name.

        Returns:
            dict: The metadata dictionary for the field.
        """
        keys = current_field.split(".")
        form_dict = self.form_data
        
        for k in keys:
            form_dict = form_dict[k]

        return form_dict
    
    def _save_info(self, value, field_name):
        """
        Saves the validated value into the saved_info dictionary at the correct nested location.

        Args:
            value: The value to save.
            field_name (str): The dot-separated field name.
        """
        keys = field_name.split(".")
        store = self.saved_info

        for k in keys[:-1]:
            if k not in store:
                store[k] = dict()
            store = store[k]
        store[keys[-1]] = value

    def start_conversation(self):
        """
        Starts the conversation with the user by printing the initial greeting and the first question.
        """
        prompt = """Hello! I'm here to help you fill this form.

Let's begin!"""
        form_field = self._find_form_data(self.flat_fields[self.current_field_index])
        print(f"{prompt}\n\n{form_field['base_prompt']}")

    def process_user_input(self, user_input):
        """
        Processes the user's input for the current field, validates it, saves it if valid, and generates the next prompt.

        Args:
            user_input (str): The user's input.

        Returns:
            tuple: (bot_reply (str), is_complete (bool))
        """
        # Validate the user input
        # If it is correct, save it and return is_completed=True with positive ack.
        # If not return False along with negative ack.
        current_field = self.flat_fields[self.current_field_index]
        field_data = self._find_form_data(current_field)
        response = self.instructor_client.chat.completions.create(
            model=self.model_name,
            messages = [
                {
                    "role" : "system",
                    "content" : f"""You are an expert at validating and extracting data.

Follow these instructions:
1. Return only the extracted data in the 'field' of the response model. 
2. If the user input is not valid, write a message in the 'error_message' of the response model, and set is_valid to False.
3. The error message should be helpful, and assist the user in providing the proper input.

Description of the user input: {field_data['description']}
Validation rules of the user input: {field_data['validation_rule']}"""
                },
                {
                    "role" : "user",
                    "content" : f"{user_input}"
                }
            ],
            max_retries=5,
            response_model=ChatInfo
        )
        
        if response.is_valid:
            valid_response = response.field

            if current_field == "date":
                valid_response = dateparser.parse(valid_response).strftime("%Y-%m-%d")
            elif "phone_number" in current_field:
                parsed_number = phonenumbers.parse(valid_response, "JP")
                if phonenumbers.is_valid_number(parsed_number):
                    valid_response = f"+{parsed_number.country_code}-{parsed_number.national_number}"
                else:
                    response.is_valid = False
                    response.error_message = f"The phone number provided {valid_response}, is not valid. Please provide a valid Japanese phone number."
                    return f"That did not work out quite well for the following reason:\n{response.error_message}\n\nLets try again. {field_data['base_prompt']}", False

            self._save_info(valid_response, current_field)

            self.current_field_index += 1

            bot_reply = f"Thank you! The data has been saved as {valid_response}."
            if self.current_field_index < len(self.flat_fields):
                # Skip the status of residence and period of stay if the guarantor is Japanese.
                if current_field == "guarantor.nationality" and response.field == "Japan":
                    self.current_field_index += 2
                    self.saved_info["guarantor"]["status_of_residence"] = "NA"
                    self.saved_info["guarantor"]["period_of_stay"] = "NA"

                next_field = self.flat_fields[self.current_field_index]
                if "guarantor" in next_field and "guarantor" not in current_field:
                    bot_reply += "\n\nNow let's find out your Guarantor's data."

                next_field_data = self._find_form_data(next_field)
                bot_reply += f"\n\nNext question. {next_field_data['base_prompt']}"
        else:
            bot_reply = f"""That did not work out quite well for the following reason:
{response.error_message}

Lets try again. {field_data['base_prompt']}"""
        
        is_complete = True if self.current_field_index == len(self.flat_fields) else False

        return bot_reply, is_complete

    def get_collected_data(self):
        """
        Returns the dictionary of all collected and validated form data.

        Returns:
            dict: The saved form data.
        """
        return self.saved_info

def letter_of_guarantee_chat():
    """
    Runs the interactive chat session for filling out the letter of guarantee form.

    Returns:
        dict: The final collected form data after completion.
    """
    chatbot = ChatBot()
    chatbot.start_conversation()

    is_complete = False
    while not is_complete:
        user_input = input("You: ")
        response, is_complete = chatbot.process_user_input(user_input)
        print("Bot:", response)

    final_data = chatbot.get_collected_data()

    return final_data