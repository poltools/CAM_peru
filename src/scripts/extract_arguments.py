
# Initializing the API
import pandas as pd
import os
from openai import OpenAI, AzureOpenAI
api_key = ""

endpoint = os.getenv("ENDPOINT_URL","" )  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", api_key)  

# Initialize Azure OpenAI Service client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-05-01-preview",
)


def generate_response(prompt, model):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a classifier that maps text to predefined categories based on detailed descriptions."},
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    
    return completion.choices[0].message.content


df = pd.read_excel('../../data/TradyAlt Fase1.xlsx', sheet_name='Completos', header=[0, 1])

df_copy = df.copy()
df_copy.columns = df_copy.columns.droplevel(1)

df_copy = df_copy[["Técnica", "Abierta"]]




prompt = """The following texts consist of descriptions of how people use certain pseudotherapies. 
Condense these texts into a list of reasons why they use these pseudotherapies.
Each reason should be expressed in a few nominalized terms. 
For example `I don't trust vaccines`-> `vaccine distrust`. 
These reasons should be generic enough so that they can be identfied in other texts as well.
Provide the list as a python list. Don't provide further comments. Only the list. Make sure this list is as complete and condense as possible. Do not include the therapy name in the list. Do not generate excessive reasons.
This is the text '<REASONS>', and this is the therapy they talk about '<THERAPY>'. \n\n Please do this in its original Spanish language."""


if __name__ == "__main__":
    extracted_reasons = []
    with open('../../data/extracted_reasons.csv', 'a') as f:
        for index, row in df_copy.iterrows():
            therapy = row["Técnica"]
            reasons = row["Abierta"]
            prompt_ = prompt.replace("<REASONS>", reasons).replace("<THERAPY>", therapy)
            response = generate_response(prompt_, 'gpt-4o')
            extracted_reasons.append(response)
            print("---")
            print(reasons)
            print(response)
            print("---")
            f.write(f"{row.name}; {therapy}; {reasons}; {response}\n")
    f.close()