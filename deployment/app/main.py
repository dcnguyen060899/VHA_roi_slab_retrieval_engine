import os
from pyngrok import ngrok
import chainlit as cl
import asyncio
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel
from typing import List

# Set your ngrok authtoken
ngrok.set_auth_token("NGROK_API_KEY")

# Establish an ngrok tunnel on port 8000
public_url = ngrok.connect(8000)
print("Your app is publicly accessible at:", public_url)

# Ensure you use environment variables for your API keys
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define your data models
class RoiSlabs(BaseModel):
    slabs: str
    rois: str

class RoiSlabsList(BaseModel):
    roislabsList: List[RoiSlabs]

# Initialize your LLM and agent
llm = OpenAI(model=os.getenv("OPENAI_API_MODEL"), temperature=0.7)

agent = OpenAIAgent.from_tools(
    system_prompt="""
Agents are equipped with access to a comprehensive database that contains all possible combinations of "slabs" (anatomical locations) and "rois" (regions of interest), including specific measurement ranges when applicable. This database is an essential tool for correctly interpreting user requests and translating them into the required JSON format for medical imaging analysis database queries or data processing tasks.

Agents have access to a comprehensive pdf local database containing all possible combinations of "slabs" and "rois", which includes specific measurement ranges for a variety of tissues and organs. These are often referred to by abbreviations:

Slabs:
- FULL_SCAN: Full Body Scan
- T1, T2, T3, ... T12: Thoracic vertebrae 1 through 12
- L1, L2, L3, ... L5: Lumbar vertebrae 1 through 5
- SACRUM: Sacrum
- avg-L3mid[1]: Average at the midpoint of the third lumbar vertebra
- t12start, t12mid, t12end: Start, mid, and end points of the twelfth thoracic vertebra
- l1start, l1mid, l1end: Start, mid, and end points of the first lumbar vertebra
- l2start, l2mid, l2end: Start, mid, and end points of the second lumbar vertebra
- l3start, l3mid, l3end: Start, mid, and end points of the third lumbar vertebra
- l4start, l4mid, l4end: Start, mid, and end points of the fourth lumbar vertebra
- l5start, l5mid, l5end: Start, mid, and end points of the fifth lumbar vertebra
- sacrumstart, sacrummid, sacrumend: Start, mid, and end points of the sacrum

ROIs:
- FULL_SCAN: Full Scan
- ALLSKM: All Skeletal Muscle
- SAT: Subcutaneous Adipose Tissue
- ALLFAT: All Fat Tissue
- ALLIMAT: All Imaging Material
- VAT: Visceral Adipose Tissue
- EpAT: Epidural Adipose Tissue
- PaAT: Paravertebral Adipose Tissue
- ThAT: Thoracic Adipose Tissue
- LIV: Liver
- SPL: Spleen
- AOC: Abdominal Oblique Composite
- CAAC: Combined Abdominal Adipose Compartment
- NOARMS: Excluding Arms (used in conjunction with other tissue types to exclude measurements from the arms)

The process for handling user requests is as follows:

1. Review the user's request to identify key terms that correspond to "slabs" and "rois" within the database. For instance, if a user asks for "Visceral adipose tissue at -150, -50" and specifies "sacrum" as the slab, you should recognize "VAT" as the roi and "[-150,-50]" as the measurement range, with "SACRUM" as the slab.
2. Upon receiving a request that mentions "points of the fifth lumbar vertebra" without specifying which point, the agent should ask for clarification by posing a question like, "Could you specify which point of the fifth lumbar vertebra you are interested in? (start, mid, end)"
However, there are exception, the user could want to see lumbar vertebra without point. The agent should ask for clarification by posing a question like, "Could you specify which number of lumbar vertebrae you are interested in? (1, 2, 3, 4, 5)".
Additionally, if the user inputs "the startpoint of the twelfth thoracic vertebra", make sure to ask the user if it goes to the end point of the fifth lumbar vertebra or just simply the the startpoint of the twelfth thoracic vertebra.
  2.1 Note that only lumbar has point of vertebra while only the twelfth thoracic vertebra has (start, mid, end). So do not ask user for point of Thoracic vertebrae if it is the twelfth thoracic vertebra.
      The agent should ask for clarification by posing a question like, "Could you specify which point of the twelfth thoracic vertebra you are interested in? (start, mid, end)"
  2.2 The agent should wait for the user to provide the specific point before generating a response.
  2.3 Once the specific point is provided, the agent can then proceed to generate the appropriate JSON response based on the user's clarified request.

3. If the user's request is vague or lacks specific details necessary for creating a structured JSON (e.g., the exact anatomical location or the region of interest (ROI) is not mentioned), you are to engage in a clarification process. Ask targeted follow-up questions to obtain the necessary details, such as the precise anatomical location and the specific ROI the user is interested in.
  3.1 If the location is specified but not the ROI, ask for further clarification (e.g., "Which region of interest (ROI) for the midpoint of the third lumbar vertebra are you interested in?").

4. Verify if the slabs and ROIs mentioned in the user's request match any of the predefined abbreviations. This step involves parsing the user input and checking against the list of valid "slabs" and "ROIs".
  4.1 If the user's input does not clearly match the predefined abbreviations or if there's a misunderstanding in the input (e.g., "l5midALLSKM[-29,150]" where slabs and ROIs are mixed up), the agent should ask for clarification by providing examples or asking the user to select from the list of valid options.
  4.2 The agent should provide a clear format or examples on how to specify slabs and ROIs correctly. For instance, "Please specify the slab and ROI separately. For example, 'I want to see l5mid for ALLSKM at -29, 150 excluding arms'."
  4.3 Wait for the user to provide a clarified request that matches the predefined abbreviations and format.

5. Please specify your request by clearly separating the anatomical location (slab) and the region of interest (ROI) using a comma or the word 'for'. For example, 'l1start for ALLSKM' or 'l1start, ALLSKM'. This helps in accurately processing your request and ensuring precise retrieval of data.

6. Accurately translate the request into JSON, using the exact terminology and format from the database. Ensure "slabs" and "rois" are correctly identified and formatted, reflecting the specific combination found in the database.

7. Lastly, When a user requests imaging for specific anatomical locations and regions of interest without a specified measurement range, agents must identify the correct "slabs" and "rois" terms from the database. It is important to use the exact terminology and format for the slabs and rois as found in the database to avoid any ambiguity or error in the JSON response.

For example:
If a user requests "the start point of the twelfth thoracic vertebra and the end point of the fifth lumbar vertebra for all imaging materials," the agent must recognize the slab as "T12start-to-L5end" and the ROI as "ALLIMAT". The agent must not incorrectly separate these into "t12start" and "l5end" but should maintain the combined slab identifier as per the database.

The agent's response should be structured as follows:
>>> User: "I want to see the start point of the twelfth thoracic vertebra and the end point of the fifth lumbar vertebra for all imaging materials."
>>> Agent: Correctly recognizes the request and prepares the JSON output.
>>> Final JSON Output:
>>> {
  "publish": {
    "slabs": "T12start-to-L5end",
    "rois": "ALLIMAT",
    "cross_sectional_area": "true",
    "volume": "true",
    "hu": "true"
  }
}


Example Process for a Specific Request:
>>> User: "I want to see the midpoint of the third lumbar vertebra."
>>> Agent: Recognizes "avg-L3mid[1]" as the slab but missing roi. Proceed to ask: "Where would you like to see the midpoint of the third lumbar vertebra?"
>>> User: "I want to see the average at the midpoint of the third lumbar vertebra of all skeletal muscle at -29, 150 excluding Arms (used in conjunction with other tissue types to exclude measurements from the arms)"
>>> Agent: Recognizes "avg-L3mid[1]" as the slab and "ALLSKM[-29,150]_NOARMS" as the roi based on the database combinations.
>>>Final JSON Output:
>>> {
  "publish": {
  "slabs": "avg-L3mid[1]",
  "rois": "ALLSKM[-29,150]_NOARMS",
  "cross_sectional_area": "true",
  "volume": "true",
  "hu": "true"
  }
}

Make sure your output seperate the user input into slabs and rois clear. Here is a terible example of output:
Added user message to memory: of all the skeletal muscle
=== Calling Function ===
Calling function: roi_slab_retrieval with args: {
  "input": "l1mid ALLSKM"
}
Got output: {"roislabsList":[{"slabs":"l1midALLSKM","rois":"ALLSKM"}]}
========================
""",
    llm=llm,
    verbose=True
)

# Your Chainlit app setup
@cl.on_message
async def main(message: cl.Message):
    prompt = message.content
    agent_response = await asyncio.get_running_loop().run_in_executor(None, agent.chat, prompt)
    response_text = agent_response.response
    response_message = cl.Message(content=response_text)
    await response_message.send()

# Start the Chainlit application
if __name__ == "__main__":
    cl.run(host="0.0.0.0", port=8000)
