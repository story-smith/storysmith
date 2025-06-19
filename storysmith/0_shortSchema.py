import json
import os
import re
from datetime import date

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain_openai import ChatOpenAI

# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# -----------------------------
# Output parser
# -----------------------------
class JSONLDParser(BaseOutputParser):
    def parse(self, text: str):
        # 念のため空でないかチェック
        if not text.strip():
            raise ValueError("Model returned empty response.")

        # そのまま JSON として試す
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # ```json ... ``` の中に含まれていないか探す
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    raise ValueError("Failed to parse matched JSON block.")
            # ダンプ出力してデバッグしやすくする
            print("❌ Model returned invalid JSON:\n", text)
            raise ValueError("JSON parse failed")


# -----------------------------
# Prompt template
# -----------------------------
system_message = """
You are a JSON-LD generator that converts short natural language stories into structured data.

You MUST use the schema.org vocabulary, specifically the type: ShortStory (a subtype of CreativeWork).
Only include properties that can be directly inferred from the story text. DO NOT invent or assume anything not mentioned.

Generate only valid JSON-LD with:
- "@context": "https://schema.org"
- "@type": "ShortStory"

Always try to extract these properties when possible:
- name (title of the story)
- abstract (short summary)
- text (full story)
- character (people or named entities)
- contentLocation (named places or settings)
- mentions (notable items, creatures, concepts, or objects mentioned in the story)
- about (theme or subject the story is centrally about)

Respond ONLY with valid JSON-LD. No explanations.
""".strip()

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_message), ("human", "{input}")]
)

# -----------------------------
# Chain
# -----------------------------
model = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=OPENAI_API_KEY)
chain = prompt_template | model | JSONLDParser()

# -----------------------------
# Example story
# -----------------------------
input_story = """
In the forest of Myrwood, a boy named Ryn meets a talking crow who offers him a quest to find the lost amulet of dusk.
"""

# -----------------------------
# Generate JSON-LD
# -----------------------------
jsonld_data = chain.invoke({"input": input_story})

# -----------------------------
# Save to stories/ directory
# -----------------------------
output_dir = "stories"
os.makedirs(output_dir, exist_ok=True)

filename = f"shortstory_{date.today().isoformat()}.jsonld"
output_path = os.path.join(output_dir, filename)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(jsonld_data, f, indent=2, ensure_ascii=False)

print(f"✅ JSON-LD saved to: {output_path}")
