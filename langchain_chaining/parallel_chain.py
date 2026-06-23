from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    task="text2text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
)

llm2 = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    task="text2text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
)

prompt1 = PromptTemplate(
    template="""
Generate short and simple notes from the following text:

{text}
""",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="""
Generate 5 short question-answer pairs from the following text:

{text}
""",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="""
Merge the following notes and quiz into a single well-structured document.

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    notes=prompt1 | llm1 | parser,
    quiz=prompt2 | llm2 | parser
)

merge_chain = prompt3 | llm1 | parser

chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification,
regression and outlier detection.

Advantages:
- Effective in high-dimensional spaces.
- Works well when dimensions exceed samples.
- Memory efficient due to support vectors.
- Supports multiple kernel functions.

Disadvantages:
- Kernel selection is important to avoid overfitting.
- Does not directly provide probability estimates.
- Sparse data predictions require appropriate training data.
"""

result = chain.invoke({
    "text": text
})

print(result)