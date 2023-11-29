from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

client = OpenAI()

def get_result(id, text):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "For the following id and text, detect if it is a disaster tweet or not. 0 is not a disaster, 1 is a disaster. Provide result as : id,label\n\nExample input: 41,I'm afraid that the tornado is coming to our area\n\nExample output: 41,1\n\n\n\n"},
            {"role": "user", "content": f"{id},{text}"},
        ]
    )
    return completion.choices[0].message.content

# get_result(7460, "Does Renovation Mean Obliteration? http://t.co/nntkiy7AXV #entrepreneur #management #leadership #smallbiz #startup #business")
# get_result(7461, "Forest fire near La Ronge Sask. Canada")
# get_result(7462, "I'm afraid that the tornado is coming to our area")
# get_result(7463, "Three people died from the heat wave so far")
# get_result(7464, "Hello how are you doing today?")


df = pd.read_csv("./nlp-getting-started/test.csv")
res = ''
for index, row in df.iterrows():
    print(row["id"], row["text"])
    res += get_result(row["id"], row["text"]) + '\n'
    print(res)

with open('result.txt', 'w') as f:
    f.write(res)


