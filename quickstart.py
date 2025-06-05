#Author: Roni Herschmann
import weave
from openai import OpenAI

client = OpenAI()

# Weave will track the inputs, outputs and code of this function
@weave.op()
def extract_dinos(sentence: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """In JSON format extract a list of `dinosaurs`, with their `name`,
their `common_name`, and whether its `diet` is a herbivore or carnivore"""
            },
            {
                "role": "user",
                "content": sentence
            }
            ],
            response_format={ "type": "json_object" }
        )
    return response.choices[0].message.content


# Initialise the weave project
weave.init('AI-Analyst')

sentence = """I am an AI Analyst that will optimize your business."""

result = extract_dinos(sentence)
print(result)