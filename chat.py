import requests
import json
import re
from typing import Dict, List, Optional

class InformationBot:
    def __init__(self, model_name: str = "llama2", ollama_url: str = "http://0.0.0.0:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.conversation_history = []
        self.extracted_info = {}
        
        # Define what information you want to extract
        self.required_fields = {
            "name": {"collected": False, "value": None},
            "email": {"collected": False, "value": None},
            "phone": {"collected": False, "value": None},
            "age": {"collected": False, "value": None},
            "purpose": {"collected": False, "value": None}
        }
    
    def generate_response(self, user_input: str) -> str:
        """Generate response using Ollama"""
        # Create context-aware prompt
        prompt = self._create_extraction_prompt(user_input)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 150
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _create_extraction_prompt(self, user_input: str) -> str:
        """Create a prompt that guides information extraction"""
        missing_fields = [field for field, info in self.required_fields.items() 
                         if not info["collected"]]
        
        extracted_summary = self._get_extracted_summary()
        
        prompt = f"""You are a helpful assistant collecting information from users. 

Current conversation context:
{extracted_summary}

Still need to collect: {', '.join(missing_fields) if missing_fields else 'All information collected!'}

User just said: "{user_input}"

Instructions:
1. First, try to extract any information from the user's message
2. Respond naturally and conversationally
3. If information is missing, ask for the next piece in a friendly way
4. Don't ask for multiple pieces of information at once
5. Keep responses brief and focused

Your response:"""
        
        return prompt
    
    def extract_information(self, user_input: str) -> Dict:
        """Extract structured information from user input"""
        extracted = {}
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, user_input)
        if emails:
            extracted["email"] = emails[0]
        
        # Phone extraction (basic pattern)
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, user_input)
        if phones:
            extracted["phone"] = ''.join(phones[0]) if isinstance(phones[0], tuple) else phones[0]
        
        # Age extraction
        age_pattern = r'\b(\d{1,2})\s*(?:years?\s*old|yo|yrs?)\b'
        ages = re.findall(age_pattern, user_input.lower())
        if ages:
            extracted["age"] = int(ages[0])
        
        # Name extraction (simple heuristic)
        name_indicators = ["my name is", "i'm", "i am", "call me"]
        for indicator in name_indicators:
            if indicator in user_input.lower():
                words = user_input.lower().split()
                try:
                    idx = words.index(indicator.split()[-1])
                    if idx + 1 < len(words):
                        extracted["name"] = user_input.split()[idx + 1].title()
                except ValueError:
                    continue
        
        return extracted
    
    def update_extracted_info(self, new_info: Dict):
        """Update the extracted information"""
        for key, value in new_info.items():
            if key in self.required_fields and not self.required_fields[key]["collected"]:
                self.required_fields[key]["value"] = value
                self.required_fields[key]["collected"] = True
    
    def _get_extracted_summary(self) -> str:
        """Get summary of collected information"""
        collected = []
        for field, info in self.required_fields.items():
            if info["collected"]:
                collected.append(f"{field}: {info['value']}")
        
        return "Collected information: " + (", ".join(collected) if collected else "None yet")
    
    def is_complete(self) -> bool:
        """Check if all required information is collected"""
        return all(info["collected"] for info in self.required_fields.values())
    
    def chat(self):
        """Main chat loop"""
        print("Hello! I'm here to help collect some information from you.")
        print("Type 'quit' to exit or 'summary' to see collected info.\n")
        
        while not self.is_complete():
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'summary':
                print(f"Bot: {self._get_extracted_summary()}")
                continue
            
            # Extract information from user input
            extracted = self.extract_information(user_input)
            self.update_extracted_info(extracted)
            
            # Generate response
            bot_response = self.generate_response(user_input)
            print(f"Bot: {bot_response}")
            
            # Add to conversation history
            self.conversation_history.append({"user": user_input, "bot": bot_response})
        
        if self.is_complete():
            print(f"\nThank you! I've collected all the information:")
            for field, info in self.required_fields.items():
                print(f"- {field.title()}: {info['value']}")
        
        return self.required_fields

# Usage example
if __name__ == "__main__":
    # Make sure Ollama is running: ollama serve
    bot = InformationBot(model_name="llama2")  # or another model you have
    bot.chat()