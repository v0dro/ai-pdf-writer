import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import requests

class LocalLLMClient:
    """Client for communicating with local LLM"""
    
    def __init__(self, endpoint: str, model_name: str = "llama2"):
        self.endpoint = endpoint
        self.model_name = model_name
    
    async def chat_completion(self, messages: List[Dict], tools: List[Dict] = None, temperature: float = 0.7):
        """Simulate OpenAI chat completion with local LLM"""
        
        # Convert messages to a single prompt for local LLM
        prompt = self._messages_to_prompt(messages)
        
        # Add tool information to prompt if tools are provided
        if tools:
            tool_info = self._format_tools_for_prompt(tools)
            prompt = f"{tool_info}\n\n{prompt}"
        
        try:
            # For Ollama
            if "ollama" in self.endpoint:
                response = requests.post(
                    self.endpoint,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": temperature}
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("response", "")
                    
                    # Check if LLM wants to call a function
                    tool_calls = self._extract_tool_calls(content)
                    
                    return {
                        "choices": [{
                            "message": {
                                "content": content,
                                "tool_calls": tool_calls if tool_calls else None
                            }
                        }]
                    }
            
            # For other local LLM APIs (Text Generation WebUI, LocalAI, etc.)
            else:
                response = requests.post(
                    self.endpoint,
                    json={
                        "prompt": prompt,
                        "max_tokens": 200,
                        "temperature": temperature
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("text", "").strip()
                    
                    tool_calls = self._extract_tool_calls(content)
                    
                    return {
                        "choices": [{
                            "message": {
                                "content": content,
                                "tool_calls": tool_calls if tool_calls else None
                            }
                        }]
                    }
        
        except Exception as e:
            print(f"LLM API error: {e}")
            return {
                "choices": [{
                    "message": {
                        "content": "I'm having trouble processing your request right now.",
                        "tool_calls": None
                    }
                }]
            }
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert OpenAI messages format to single prompt"""
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                prompt_parts.append(f"Tool Result: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def _format_tools_for_prompt(self, tools: List[Dict]) -> str:
        """Format tools information for the prompt"""
        tool_descriptions = []
        
        for tool in tools:
            func = tool["function"]
            name = func["name"]
            description = func["description"]
            parameters = func["parameters"]
            
            tool_desc = f"Available function: {name}\nDescription: {description}\nParameters: {json.dumps(parameters, indent=2)}"
            tool_descriptions.append(tool_desc)
        
        instructions = """
When you need to call a function, respond with exactly this format:
FUNCTION_CALL: {
  "name": "function_name",
  "arguments": {"param1": "value1", "param2": "value2"}
}

Only call functions when you have collected all required information from the user.
"""
        
        return f"{instructions}\n" + "\n\n".join(tool_descriptions)
    
    def _extract_tool_calls(self, content: str) -> List[Dict]:
        """Extract function calls from LLM response"""
        if "FUNCTION_CALL:" not in content:
            return None
        
        try:
            # Find the JSON part after FUNCTION_CALL:
            start_idx = content.find("FUNCTION_CALL:") + len("FUNCTION_CALL:")
            json_part = content[start_idx:].strip()
            
            # Try to extract JSON
            if json_part.startswith("{"):
                end_idx = json_part.find("}")
                if end_idx != -1:
                    json_str = json_part[:end_idx + 1]
                    func_call = json.loads(json_str)
                    
                    return [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": func_call["name"],
                            "arguments": json.dumps(func_call["arguments"])
                        }
                    }]
        except Exception as e:
            print(f"Error extracting tool calls: {e}")
        
        return None

class TerminalChatbot:
    """Terminal-based chatbot with data collection"""
    
    def __init__(self, llm_endpoint: str, model_name: str = "llama2"):
        self.client = LocalLLMClient(llm_endpoint, model_name)
        self.conversation_history = []
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "save_product_order",
                    "description": "Save a product order when all required information is collected",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_name": {
                                "type": "string",
                                "description": "Name of the product to order"
                            },
                            "user_name": {
                                "type": "string", 
                                "description": "Customer's full name"
                            },
                            "email": {
                                "type": "string",
                                "description": "Customer's email address"
                            }
                        },
                        "required": ["product_name", "user_name", "email"]
                    }
                }
            }
        ]
        
        self.bot_instructions = """
You are a helpful assistant for an online store. Your job is to help customers place product orders.

When a user expresses interest in ordering a product, you need to collect the following information:
1. Product name - what product they want to order
2. User name - their full name for the order
3. Email address - their email for order confirmation

Ask for this information in a natural, conversational way. Only ask for one piece of information at a time.
Don't ask for information you already have.

Once you have collected all three pieces of information (product name, user name, and email), 
call the save_product_order function with the collected data.

Be friendly and helpful throughout the conversation.
"""
    
    async def save_product_order(self, product_name: str, user_name: str, email: str) -> str:
        """Save the product order"""
        order_data = {
            "product_name": product_name,
            "user_name": user_name,
            "email": email,
            "timestamp": datetime.now().isoformat()
        }
        
        print("\n" + "="*50)
        print("🎉 ORDER COLLECTED SUCCESSFULLY!")
        print("="*50)
        print(f"📦 Product: {product_name}")
        print(f"👤 Customer: {user_name}")
        print(f"📧 Email: {email}")
        print(f"⏰ Time: {order_data['timestamp']}")
        print("="*50)
        print("💾 Order saved to system!")
        print("📧 Confirmation email will be sent shortly.")
        print("="*50 + "\n")
        
        # Here you could save to database, send email, etc.
        
        return f"Perfect! Your order for '{product_name}' has been saved successfully! We'll send a confirmation email to {email} shortly."
    
    async def invoke_function_in_tool_calls(self, tool_calls: List[Dict]) -> str:
        """Invoke the appropriate function based on tool calls"""
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            
            if function_name == "save_product_order":
                result = await self.save_product_order(
                    function_args.get("product_name"),
                    function_args.get("user_name"), 
                    function_args.get("email")
                )
                return json.dumps({
                    "tool_call_id": tool_call["id"],
                    "result": result
                })
        
        return json.dumps({"error": "Unknown function"})
    
    async def get_chat_response(self, user_message: str) -> str:
        """Get response from local LLM with data collection logic"""
        
        # Initialize conversation with system message if empty
        if not self.conversation_history:
            self.conversation_history.append({
                "role": "system",
                "content": self.bot_instructions
            })
        
        # Add user message
        self.conversation_history.append({
            "role": "user", 
            "content": user_message
        })
        
        # Get response from LLM
        response = await self.client.chat_completion(
            messages=self.conversation_history,
            tools=self.tools,
            temperature=0.7
        )
        
        response_message = response["choices"][0]["message"]
        tool_calls = response_message.get("tool_calls")
        
        # If no tool calls, just return the response
        if not tool_calls:
            self.conversation_history.append({
                "role": "assistant",
                "content": response_message["content"]
            })
            return response_message["content"]
        
        # If tool calls exist, handle them
        self.conversation_history.append({
            "role": "assistant", 
            "content": response_message["content"],
            "tool_calls": tool_calls
        })
        
        # Execute the function
        function_result = await self.invoke_function_in_tool_calls(tool_calls)
        
        # Add function result to conversation
        self.conversation_history.append({
            "role": "tool",
            "content": function_result
        })
        
        # Get final response after function execution
        second_response = await self.client.chat_completion(
            messages=self.conversation_history,
            temperature=0.1
        )
        
        final_message = second_response["choices"][0]["message"]["content"]
        
        # Clear conversation history after successful order
        self.conversation_history = []
        
        return final_message
    
    def print_welcome(self):
        """Print welcome message"""
        print("\n" + "="*60)
        print("🤖 WELCOME TO LOCAL LLM CHATBOT WITH DATA COLLECTION")
        print("="*60)
        print("💬 I can help you place product orders!")
        print("📝 I'll collect: Product name, Your name, Email address")
        print("🚪 Type 'quit', 'exit', or 'bye' to end the conversation")
        print("="*60 + "\n")
    
    def print_separator(self):
        """Print conversation separator"""
        print("-" * 60)
    
    async def run(self):
        """Run the terminal chat interface"""
        self.print_welcome()
        
        # Send welcome message
        print("🤖 Bot: Hello! I can help you place product orders. What would you like to order today?")
        self.print_separator()
        
        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n🤖 Bot: Thank you for using our service! Goodbye! 👋")
                    break
                
                if not user_input:
                    print("🤖 Bot: Please enter a message.")
                    continue
                
                # Show processing message
                print("🤖 Bot: Processing... ⏳")
                
                # Get bot response
                bot_response = await self.get_chat_response(user_input)
                
                # Clear the processing line and show response
                print(f"\r🤖 Bot: {bot_response}")
                self.print_separator()
                
            except KeyboardInterrupt:
                print("\n\n🤖 Bot: Conversation interrupted. Goodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("🤖 Bot: I encountered an error. Please try again.")
                self.print_separator()

async def main():
    """Main function to run the chatbot"""
    
    # Configuration
    LLM_ENDPOINT = "http://localhost:11434/api/generate"  # Ollama default
    MODEL_NAME = "llama2"  # Change to your model
    
    print("🚀 Initializing Local LLM Chatbot...")
    print(f"🔗 Endpoint: {LLM_ENDPOINT}")
    print(f"🤖 Model: {MODEL_NAME}")
    
    try:
        # Test connection to LLM
        test_response = requests.get("http://localhost:11434/api/tags")  # Ollama health check
        if test_response.status_code != 200:
            print("⚠️  Warning: Could not connect to LLM server. Make sure it's running.")
    except:
        print("⚠️  Warning: Could not connect to LLM server. Make sure it's running.")
        print("   For Ollama: Run 'ollama serve' and 'ollama run llama2'")
    
    # Create and run chatbot
    chatbot = TerminalChatbot(LLM_ENDPOINT, MODEL_NAME)
    await chatbot.run()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())