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
    base_prompt: str
    validation_rules: str
    path: List[str]  # Path in the nested structure
    completed: bool = False
    value: Any = None
    attempts: int = 0
    validation_history: List[str] = None  # Track validation attempts

    def __post_init__(self):
        if self.validation_history is None:
            self.validation_history = []

class FormFillingChatbot:
    def __init__(self, form_fields: List[Dict], model_name: str = "llama3.2:3b"):
        self.tools = form_fields
        self.model_name = model_name
        self.client = ollama.Client()
        
        # Parse the form structure
        self.form_name = form_fields["name"]
        self.form_description = form_fields["description"]
        self.fields = self._parse_fields(form_fields["parameters"])
        
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

    def start_conversation(self) -> str:
        """Start the conversation."""
        greeting = """Hello! I'm here to help you fill out a letter of guarantee form. 
I'll guide you through each field step by step. Don't worry if you make a mistake - I'll help you correct it!

Let's begin! Lets start with the date."""
        
        # Get the first field prompt
        if self.fields:
            first_prompt = self.fields[0].base_prompt
            return f"{greeting}\n\n{first_prompt}"
        
        return greeting
    
    def _validate_with_llm(self, user_input: str, field: FormField) -> Tuple[bool, Any, Optional[str]]:
        """
        Validate user input using the LLM.
        Returns a tuple of (is_valid, cleaned_value, error_explanation).
        """
        validation_prompt = f"""You are an assistant for validating user inputs for a form.

The form field, type, description and validation rules are specified below.
Field Name: {field.name}
Field Type: {field.field_type}
Description: {field.description}
Validation rules: {field.validation_rules}
User Input: "{user_input}"

Your task:
1. Check if the input is valid for this field type.
2. From the "Validation rules" of the field, you must pre-process the user input as per the validation rules and extract, clean the relevant information.
3. If the user input is invalid, explain what is wrong.

For all user inputs, empty strings should be considered as invalid inputs. Return ONLY the JSON and NOTHING else.
Respond in JSON format:
{{
    "is_valid": true/false,
    "cleaned_value": "the extracted and cleaned value",
    "error_explanation": "explanation if invalid, null if valid",
    "suggestions": "helpful suggestions for the user if invalid"
}}
"""
        messages = [
            {"role": "system", "content": "You are a form validation assistant. Validate inputs and provide helpful feedback."},
            {"role": "user", "content": validation_prompt}
        ]

        print("INPUT MESSAGES:")
        print(messages)
        response = self.client.chat(
            model=self.model_name,
            messages=messages
        )
        print("PRINTING RESPONSE CONTENT.")
        print(response['message']['content'])
        result = json.loads(response['message']['content'])
        return (
            result.get('is_valid', False),
            result.get('cleaned_value', user_input),
            result.get('error_explanation')
        )
    
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

    def _create_validation_response(self, field: FormField, error_explanation: str, user_input: str) -> str:
        """Use LLM to create a natural response for validation errors."""
        context = f"""The user provided invalid input for a form field.

Field: {field.name} ({field.description})
User said: "{user_input}"
Problem: {error_explanation}
This is attempt #{field.attempts + 1}

Create a friendly, helpful response that:
1. Acknowledges what they said
2. Explains the issue clearly
3. Provides a specific example
4. Re-asks the question naturally

Be conversational and patient. If this is their 3rd+ attempt, be extra helpful with examples."""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context}
        ]
    
        response = self.client.chat(
            model=self.model_name,
            messages=messages
        )
        return response['message']['content']

    def _generate_acknowledgment(self, field: FormField, value: str) -> str:
        """Generate a natural acknowledgment using LLM."""
        context = f"""The user just provided valid information for a form field.

Field: {field.name} ({field.description})
Value provided: {value}
Field type: {field.field_type}

Generate a brief, friendly acknowledgment that:
1. Confirms you received the information
2. Sounds natural and conversational
3. Varies based on the type of information

Keep it short and friendly. Don't repeat the exact value unless it adds value."""

        messages = [
            {"role": "system", "content": "You are a friendly form assistant. Acknowledge user inputs naturally and briefly."},
            {"role": "user", "content": context}
        ]
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages
            )
            return response['message']['content']
        except:
            # Fallback acknowledgments
            return f"Thank you! I've recorded that."

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
   
    def process_user_input(self, user_input: str) -> Tuple[str, bool]:
        """
        Process user input and return (response, is_complete).
        """
        if self.current_field_index >= len(self.fields):
            return "Thank you! I have collected all the necessary information.", True
        
        current_field = self.fields[self.current_field_index]
        
        # Add to validation history
        current_field.validation_history.append(user_input)
        
        # Validate using LLM
        is_valid, cleaned_value, error_explanation = self._validate_with_llm(user_input, current_field)
        
        if not is_valid:
            current_field.attempts += 1
            
            # Create a natural validation response
            validation_response = self._create_validation_response(
                current_field, 
                error_explanation or "The input doesn't seem correct", 
                user_input
            )
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": validation_response})
            
            return validation_response, False
        
        # Store the cleaned value
        self._store_value(current_field, cleaned_value)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Generate acknowledgment
        acknowledgment = self._generate_acknowledgment(current_field, cleaned_value)
        self.conversation_history.append({"role": "assistant", "content": acknowledgment})
        
        # Move to next field
        self.current_field_index += 1
        
        # Check if we're done
        if self.current_field_index >= len(self.fields):
            summary = self._create_summary()
            return f"{acknowledgment}\n\n{summary}", True
        
        # Get next field prompt
        next_field = self.fields[self.current_field_index]
        next_prompt = next_field.base_prompt
        
        # Special transition when moving to guarantor section
        if self.current_field_index > 0 and self.fields[self.current_field_index - 1].path[0] != next_field.path[0]:
            if next_field.path[0] == "guarantor":
                transition = "\n\nGreat! Now I need to collect some information about your guarantor."
                return f"{acknowledgment}{transition}\n\n{next_prompt}", False
        
        return f"{acknowledgment}\n\n{next_prompt}", False
    
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
                    base_prompt=field_info.get("base_prompt", ""),
                    validation_rules=field_info.get("validation_rules", ""),
                    path=current_path
                )
                fields.append(field)
        
        return fields

# Example usage with simulated validation scenarios
def main():
    # The provided tools JSON
    form_fields = {
        "name": "Letter of Guarantee Form",
        "description": "Form fields for the letter of guarantee.",
        "parameters": {
            "date": {
                "type": "string",
                "description": "The date on which you are filling this form.",
                "base_prompt": "Could please provide the date you are filling this form? It can be in any format you prefer.",
                "validation_rules" : "Check if is a valid date and return the value. Do not impose any constraint on the format of the date."
            },
            "nationality": {
                "type": "string",
                "description": "The name of the country of citizenship of the user.",
                "base_prompt": "What is your nationality (country of citizenship)?",
                "validation_rules" : "Validate the user input against a list of countries."
            },
            "name": {
                "type": "string",
                "description": "The full name of the user.",
                "base_prompt": "May I have your full name, please?",
                "validation_rules" : "The name must be a non-empty string. It should at least contain a first name and a last name."
            },
            "guarantor": {
                "name": {
                    "type": "string",
                    "description": "The full name of the guarantor.",
                    "base_prompt": "Could you please provide the full name of your guarantor?",
                    "validation_rules" : "The name must be a non-empty string. It should at least contain a first name and a last name."
                },
                "address_in_japan": {
                    "type": "string",
                    "description": "The address of the guarantor in Japan.",
                    "base_prompt": "What is the address of your guarantor in Japan? Please provide the full address including postal code.",
                    "validation_rules" :""
                },
                "guarantor_phone_number": {
                    "type": "string",
                    "description": "The phone number of the guarantor in Japan.",
                    "base_prompt": "What is your guarantor's phone number in Japan?",
                    "validation_rules" :""
                },
                "place_of_employment": {
                    "type": "string",
                    "description": "The place of employment of the guarantor.",
                    "base_prompt": "Where does your guarantor work?",
                    "validation_rules" :""
                },
                "occupation_phone_number": {
                    "type": "string",
                    "description": "The phone number of the guarantor's place of employment.",
                    "base_prompt": "What is the phone number of your guarantor's place of employment?",
                    "validation_rules" :""
                },
                "nationality": {
                    "type": "string",
                    "description": "The nationality of the guarantor.",
                    "base_prompt": "What is your guarantor's nationality?",
                    "validation_rules" :""
                },
                "status_of_residence": {
                    "type": "string",
                    "description": "The status of residence of the guarantor.",
                    "base_prompt": "What is your guarantor's status of residence in Japan?",
                    "validation_rules" :""
                },
                "period_of_stay": {
                    "type": "string",
                    "description": "The period of stay of the guarantor if they are not a Japanese citizen.",
                    "base_prompt": "If your guarantor is not a Japanese citizen, what is their period of stay in Japan? Please provide the start and end dates.",
                    "validation_rules" :""
                },
                "guarantor_relationship": {
                    "type": "string",
                    "description": "The relationship of the guarantor to the user.",
                    "base_prompt": "What is your relationship with the guarantor?",
                    "validation_rules" :""
                }
            }
        }
    }
    
    # Create chatbot instance
    chatbot = FormFillingChatbot(form_fields)
    
    # Start conversation
    print("Bot:", chatbot.start_conversation())
    print()
    
    # Interactive mode
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