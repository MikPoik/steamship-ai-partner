"""Answers questions with the assistance of a VectorSearch plugin."""
from typing import Any, List, Optional, Union

from steamship import Block, Tag, Task #upm package(steamship)
from steamship.agents.llms import OpenAI #upm package(steamship)
from steamship.agents.schema import AgentContext #upm package(steamship)
from steamship.agents.tools.question_answering.vector_search_tool import VectorSearchTool #upm package(steamship)
from steamship.agents.utils import get_llm, with_llm #upm package(steamship)
from steamship.utils.repl import ToolREPL #upm package(steamship)

DEFAULT_QUESTION_ANSWERING_PROMPT = (
    "Use the following pieces of memory to answer the question about User at the end."
    """If you don't know the answer, respond politely that you don't remember, do no try to make up answer. 

{source_text}

Question: {question}

Helpful Answer:"""
)


DEFAULT_SOURCE_DOCUMENT_PROMPT = "Source Document: {text}"


class VectorSearchQATool(VectorSearchTool):
    """Tool to answer questions with the assistance of a vector search plugin."""

    name: str = "VectorSearchQATool"
    human_description: str = "Answers questions with help from a Vector Database."
    agent_description: str = (
        "Used to answer questions about User or assistant's role-play character. "
        "The input should be a plain text question. "
        "The output is a plain text answer."
    )
    question_answering_prompt: Optional[str] = DEFAULT_QUESTION_ANSWERING_PROMPT
    source_document_prompt: Optional[str] = DEFAULT_SOURCE_DOCUMENT_PROMPT
    load_docs_count: int = 2

    def answer_question(self, question: str, context: AgentContext) -> List[Block]:
        index = self.get_embedding_index(context.client)
        task = index.search(question, k=self.load_docs_count)
        task.wait()

        source_texts = []

        for item in task.output.items:
            if item.tag and item.tag.text:
                item_data = {"text": item.tag.text}
                source_texts.append(self.source_document_prompt.format(**item_data))

        final_prompt = self.question_answering_prompt.format(
            **{"source_text": "\n".join(source_texts), "question": question}
        )
        llm = get_llm(context, default=OpenAI(client=context.client))
        return llm.complete(prompt=final_prompt)

    def run(self, tool_input: List[Block], context: AgentContext) -> Union[List[Block], Task[Any]]:
        """Answers questions with the assistance of an Embedding Index plugin.

        Inputs
        ------
        tool_input: List[Block]
            A list of blocks to be rewritten if text-containing.
        context: AgentContext
            The active AgentContext.

        Output
        ------
        output: List[Blocks]
            A lit of blocks containing the answers.
        """

        output = []
        for input_block in tool_input:
            if not input_block.is_text():
                continue
            for output_block in self.answer_question(input_block.text, context):
                output.append(output_block)
        return output


if __name__ == "__main__":
    tool = VectorSearchQATool()
    repl = ToolREPL(tool)

    with repl.temporary_workspace() as client:
        index = tool.get_embedding_index(client)
        index.insert([Tag(text="Ted loves apple pie."), Tag(text="The secret passcode is 1234.")])
        repl.run_with_client(
            client, context=with_llm(context=AgentContext(), llm=OpenAI(client=client))
        )
