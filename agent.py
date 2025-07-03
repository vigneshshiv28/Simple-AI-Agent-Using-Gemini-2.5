from google import genai
from dotenv import load_dotenv
import os 
from google.genai import types
load_dotenv()
from datetime import datetime

import re

API_KEY=os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-2.5-pro"


def get_time() -> str:
    """Get the current time in YYYY-MM-DD HH:MM:SS format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def evaluate_expression(expression: str) -> str:
    """
    Evaluate a mathematical expression. Input should be a valid Python expression.
    
    Parameters:
    - expression: A mathematical expression to evaluate, e.g., "2 + 2"
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"
    

get_time_declaration = types.FunctionDeclaration(
    name="get_time",
    description="Get the current time in YYYY-MM-DD HH:MM:SS format.",
    parameters=None 
)

evaluate_expression_declaration = types.FunctionDeclaration(
    name="evaluate_expression",
    description="Evaluate a mathematical expression. Input should be a valid Python expression.",
    parameters={
        "type":"OBJECT",
        "properties":{
            "expression": {
                "type":"STRING",
                "description":"A mathematical expression to evaluate, e.g., '2 + 2'.",
            },
        },
        "required":["expression"],
    },
)



get_time_tool = types.Tool( function_declarations=[get_time_declaration],)
evaluate_expression_tool = types.Tool(function_declarations=[evaluate_expression_declaration],)




chat_config = types.GenerateContentConfig(
    temperature=0,
    tools=[get_time_tool, evaluate_expression_tool],
)



user_prompt = "What is the current time? Also, what is 2 + 2?"
history = f"You are a helpful AI agent.\nUser: {user_prompt}\n"

def call_llm(prompt):
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=chat_config,
    )

    return response
    



def call_agent():
    user_prompt = "What is the current time? Also, what is 2 + 2?"
    conversation = [user_prompt]

    iteration = 0
    max_iterations = 10

    while iteration < max_iterations:
        print(f"Iteration {iteration + 1}:")


        response = client.models.generate_content(
            model=MODEL_ID,
            contents=conversation,
            config=chat_config,
        )

      
      
        candidate = response.candidates[0]


        

        if candidate.content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            print(f"Function to call: {function_call.name}")
            print(f"Arguments: {function_call.args}")


            
            if function_call.name == "get_time":
                print(f"Calling get_time function...")
                current_time = get_time()
                print(f"Current time: {current_time}")
                
               
                function_response = types.Part(
                    function_response=types.FunctionResponse(
                        name="get_time",
                        response={"result": current_time}
                    )
                )
                conversation.append(types.Content(parts=[function_response]))
                
            if function_call.name == "evaluate_expression":
                print(f"Calling evaluate_expression function...")
                expression = function_call.args.get("expression", "")
                if expression:
                    result = evaluate_expression(expression)
                    print(f"Result of expression '{expression}': {result}")
                    
                    
                    function_response = types.Part(
                        function_response=types.FunctionResponse(
                            name="evaluate_expression",
                            response={"result": result}
                        )
                    )
                    conversation.append(types.Content(parts=[function_response]))
                else:
                    print("No expression provided to evaluate.")
        else:
            print("No function call detected in the response.")
            print("Final response:", response.candidates[0].content.parts[0].text)
            break
        
        iteration += 1
    
    if iteration > max_iterations:
        print("Maximum iterations reached. Stopping.")

if __name__ == "__main__":
    call_agent()

