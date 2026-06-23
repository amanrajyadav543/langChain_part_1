from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser
)
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda
)

load_dotenv()

# LLM
model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Output parser
parser = StrOutputParser()

# Structured output schema
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback"
    )

parser2 = PydanticOutputParser(
    pydantic_object=Feedback
)

# Sentiment classification prompt
prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback.

Feedback:
{feedback}

{format_instructions}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions":
        parser2.get_format_instructions()
    }
)

classifier_chain = prompt1 | model | parser2

# Positive response prompt
prompt2 = PromptTemplate(
    template="""
Write a professional response to this positive feedback:

{feedback}
""",
    input_variables=["feedback"]
)

# Negative response prompt
prompt3 = PromptTemplate(
    template="""
Write a professional response to this negative feedback:

{feedback}
""",
    input_variables=["feedback"]
)

positive_chain = prompt2 | model | parser
negative_chain = prompt3 | model | parser

# Branch logic
branch_chain = RunnableBranch(
    (
        lambda x: x.sentiment == "positive",
        RunnableLambda(
            lambda x: {
                "feedback":
                "This customer gave positive feedback."
            }
        ) | positive_chain
    ),
    (
        lambda x: x.sentiment == "negative",
        RunnableLambda(
            lambda x: {
                "feedback":
                "This customer gave negative feedback."
            }
        ) | negative_chain
    ),
    RunnableLambda(
        lambda x: "Invalid sentiment"
    )
)

# Final chain
chain = classifier_chain | branch_chain

# Test
result = chain.invoke({
    "feedback":
    "This is a great product. I love using it."
})

print("\nResponse:\n")
print(result)

print("\nGraph:\n")
chain.get_graph().print_ascii()