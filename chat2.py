import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import ollama

class FieldType(Enum):
    STRING = "string"
    DATE = "date"

@dataclass
class FormField:
    name: str
    field_type: str
    description: str
    path: List[str]  # Path in the nested structure
    completed: bool = False
    value: Any = None
    attempts: int = 0

class FormFillingChatbot:
    def __init__(self, tools_json: List[Dict], model_name: str = "llama3.2:3b"):
        self.tools = tools_json
        self.model_name = model_name
        self.client = ollama.Client()
        
        # Parse the form structure
        self.form_name = tools_json[0]["name"]
        self.form_description = tools_json[0]["description"]
        self.fields = self._parse_fields(tools_json[0]["parameters"])
        
        # Conversation state
        self.conversation_history = []
        self.current_field_index = 0
        self.collected_data = {}
        
        # System prompt for the LLM
        self.system_prompt = """You are a friendly assistant helping users fill out a letter of guarantee form. 
Your role is to:
1. Ask for information in a conversational, friendly manner
2. Validate responses and ask for clarification when needed
3. Acknowledge the information received positively
4. Guide the user through the entire form

Be warm, professional, and helpful. Keep your responses concise but friendly."""

    def _parse_fields(self, parameters: Dict, path: List[str] = None) -> List[FormField]:
        """Parse the nested parameter structure into a flat list of fields."""
        if path is None:
            path = []
        
        fields = []
        for field_name, field_info in parameters.items():
            current_path = path + [field_name]
            
            # Check if this is a nested structure
            if isinstance(field_info, dict) and "type" not in field_info:
                # This is a nested object, recurse
                fields.extend(self._parse_fields(field_info, current_path))
            else:
                # This is a field
                field = FormField(
                    name=field_name,
                    field_type=field_info.get("type", "string"),
                    description=field_info.get("description", ""),
                    path=current_path
                )
                fields.append(field)
        
        return fields

    def _create_field_prompt(self, field: FormField) -> str:
        """Create a natural prompt for asking about a specific field."""
        prompts = {
            "date": f"First, could you tell me today's date? (Please provide it in YYYY-MM-DD format)",
            "nationality": f"What is your nationality (country of citizenship)?",
            "name": f"May I have your full name, please?",
            "guarantor.name": f"Now I need some information about your guarantor. What is your guarantor's full name?",
            "guarantor.address_in_japan": f"What is your guarantor's address in Japan?",
            "guarantor.guarantor_phone_number": f"What is your guarantor's phone number in Japan?",
            "guarantor.place_of_employment": f"Where does your guarantor work?",
            "guarantor.occupation_phone_number": f"What is the phone number of your guarantor's workplace?",
            "guarantor.nationality": f"What is your guarantor's nationality?",
            "guarantor.status_of_residence": f"What is your guarantor's status of residence in Japan?",
            "guarantor.period_of_stay": f"What is your guarantor's period of stay? (This is needed if they are not a Japanese citizen)",
            "guarantor.guarantor_relationship": f"What is your relationship to the guarantor?"
        }
        
        # Create field key from path
        field_key = ".".join(field.path)
        return prompts.get(field_key, f"Could you provide {field.description}?")

    def _get_llm_response(self, user_input: str, current_field: FormField) -> str:
        """Get a response from the LLM for processing user input."""
        # Build conversation context
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history (last 5 exchanges)
        for msg in self.conversation_history[-10:]:
            messages.append(msg)
        
        # Add current interaction
        messages.append({"role": "user", "content": user_input})
        
        # Add context about what we're looking for
        context = f"\n\nContext: We are collecting '{current_field.name}' - {current_field.description}"
        messages.append({"role": "system", "content": context})
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return self._get_fallback_response(user_input, current_field)

    def _get_fallback_response(self, user_input: str, current_field: FormField) -> str:
        """Fallback response if LLM fails."""
        return f"Thank you! I've noted that your {current_field.name} is '{user_input}'."

    def _validate_field(self, value: str, field: FormField) -> Tuple[bool, Optional[str]]:
        """Validate field input and return (is_valid, error_message)."""
        if not value.strip():
            return False, "This field cannot be empty. Could you please provide this information?"
        
        if field.field_type == "date":
            try:
                datetime.strptime(value.strip(), "%Y-%m-%d")
                return True, None
            except ValueError:
                return False, "Please provide the date in YYYY-MM-DD format (e.g., 2024-01-15)."
        
        # For phone numbers, do basic validation
        if "phone" in field.name.lower():
            # Remove spaces and hyphens
            cleaned = re.sub(r'[\s\-\(\)]', '', value)
            if not cleaned.isdigit() or len(cleaned) < 10:
                return False, "Please provide a valid phone number with at least 10 digits."
        
        # For other fields, just ensure they're not empty
        return True, None

    def _extract_value_with_llm(self, user_input: str, field: FormField) -> str:
        """Use LLM to extract the relevant value from user input."""
        extraction_prompt = f"""From the following user input, extract only the {field.description}.
User input: "{user_input}"
Field we're looking for: {field.name} ({field.description})

Return ONLY the extracted value, nothing else."""

        messages = [
            {"role": "system", "content": "You are a data extraction assistant. Extract only the requested information from user input."},
            {"role": "user", "content": extraction_prompt}
        ]
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages
            )
            return response['message']['content'].strip()
        except:
            # Fallback: return the original input
            return user_input.strip()

    def _store_value(self, field: FormField, value: str):
        """Store the validated value in the nested structure."""
        current = self.collected_data
        
        # Navigate to the correct nested location
        for i, key in enumerate(field.path[:-1]):
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Store the value
        current[field.path[-1]] = value
        field.value = value
        field.completed = True

    def start_conversation(self) -> str:
        """Start the conversation."""
        greeting = """Hello! I'm here to help you fill out a letter of guarantee form. 
I'll guide you through each field step by step. Let's begin!"""
        
        # Get the first field prompt
        if self.fields:
            first_prompt = self._create_field_prompt(self.fields[0])
            return f"{greeting}\n\n{first_prompt}"
        
        return greeting

    def process_user_input(self, user_input: str) -> Tuple[str, bool]:
        """
        Process user input and return (response, is_complete).
        """
        if self.current_field_index >= len(self.fields):
            return "Thank you! I have collected all the necessary information.", True
        
        current_field = self.fields[self.current_field_index]
        
        # Extract value using LLM
        extracted_value = self._extract_value_with_llm(user_input, current_field)
        
        # Validate the extracted value
        is_valid, error_message = self._validate_field(extracted_value, current_field)
        
        if not is_valid:
            current_field.attempts += 1
            if current_field.attempts > 2:
                # After 3 attempts, be more helpful
                return f"{error_message}\n\nLet me help you. {self._create_field_prompt(current_field)}", False
            return error_message, False
        
        # Store the value
        self._store_value(current_field, extracted_value)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Get acknowledgment from LLM
        acknowledgment = self._get_llm_response(user_input, current_field)
        self.conversation_history.append({"role": "assistant", "content": acknowledgment})
        
        # Move to next field
        self.current_field_index += 1
        
        # Check if we're done
        if self.current_field_index >= len(self.fields):
            summary = self._create_summary()
            return f"{acknowledgment}\n\n{summary}", True
        
        # Get next field prompt
        next_field = self.fields[self.current_field_index]
        next_prompt = self._create_field_prompt(next_field)
        
        # Special transition when moving to guarantor section
        if self.current_field_index > 0 and self.fields[self.current_field_index - 1].path[0] != next_field.path[0]:
            if next_field.path[0] == "guarantor":
                transition = "\n\nGreat! Now I need to collect some information about your guarantor."
                return f"{acknowledgment}{transition}\n\n{next_prompt}", False
        
        return f"{acknowledgment}\n\n{next_prompt}", False

    def _create_summary(self) -> str:
        """Create a summary of collected information."""
        summary = "Excellent! I've collected all the necessary information. Here's a summary:\n\n"
        
        # Personal Information
        summary += "**Your Information:**\n"
        summary += f"- Date: {self.collected_data.get('date', 'Not provided')}\n"
        summary += f"- Name: {self.collected_data.get('name', 'Not provided')}\n"
        summary += f"- Nationality: {self.collected_data.get('nationality', 'Not provided')}\n"
        
        # Guarantor Information
        if 'guarantor' in self.collected_data:
            summary += "\n**Guarantor Information:**\n"
            g = self.collected_data['guarantor']
            summary += f"- Name: {g.get('name', 'Not provided')}\n"
            summary += f"- Relationship: {g.get('guarantor_relationship', 'Not provided')}\n"
            summary += f"- Nationality: {g.get('nationality', 'Not provided')}\n"
            summary += f"- Address in Japan: {g.get('address_in_japan', 'Not provided')}\n"
            summary += f"- Phone: {g.get('guarantor_phone_number', 'Not provided')}\n"
            summary += f"- Workplace: {g.get('place_of_employment', 'Not provided')}\n"
            summary += f"- Work Phone: {g.get('occupation_phone_number', 'Not provided')}\n"
            summary += f"- Status of Residence: {g.get('status_of_residence', 'Not provided')}\n"
            if g.get('period_of_stay'):
                summary += f"- Period of Stay: {g.get('period_of_stay', 'Not provided')}\n"
        
        summary += "\nThank you for providing all this information!"
        return summary

    def get_collected_data(self) -> Dict[str, Any]:
        """Return the collected data as a dictionary."""
        return self.collected_data

# Example usage
def main():
    # The provided tools JSON
    tools = [{
        "name": "letter_of_guarantee_details",
        "description": "Form fields for the letter of guarantee.",
        "parameters": {
            "date": {
                "type": "date",
                "description": "The date on which you are filling this form."
            },
            "nationality": {
                "type": "string",
                "description": "The name of the country of citizenship of the user."
            },
            "name": {
                "type": "string",
                "description": "The full name of the user."
            },
            "guarantor": {
                "name": {
                    "type": "string",
                    "description": "The full name of the guarantor."
                },
                "address_in_japan": {
                    "type": "string",
                    "description": "The address of the guarantor in Japan."
                },
                "guarantor_phone_number": {
                    "type": "string",
                    "description": "The phone number of the guarantor in Japan."
                },
                "place_of_employment": {
                    "type": "string",
                    "description": "The place of employment of the guarantor."
                },
                "occupation_phone_number": {
                    "type": "string",
                    "description": "The phone number of the guarantor's place of employment."
                },
                "nationality": {
                    "type": "string",
                    "description": "The nationality of the guarantor."
                },
                "status_of_residence": {
                    "type": "string",
                    "description": "The status of residence of the guarantor."
                },
                "period_of_stay": {
                    "type": "string",
                    "description": "The period of stay of the guarantor if they are not a Japanese citizen."
                },
                "guarantor_relationship": {
                    "type": "string",
                    "description": "The relationship of the guarantor to the user."
                }
            }
        }
    }]
    
    # Create chatbot instance
    chatbot = FormFillingChatbot(tools)
    
    # Start conversation
    print("Bot:", chatbot.start_conversation())
    print()
    
    # Simulation loop (replace with actual user input in production)
    is_complete = False
    while not is_complete:
        user_input = input("You: ")
        response, is_complete = chatbot.process_user_input(user_input)
        print("Bot:", response)
        print()
    
    # Get the final collected data
    final_data = chatbot.get_collected_data()
    print("\nCollected Data (as Python dict):")
    print(json.dumps(final_data, indent=2))
    
    return final_data

if __name__ == "__main__":
    # Ensure Ollama is running and has llama3.2:3b model pulled
    # Run: ollama pull llama3.2:3b
    main()