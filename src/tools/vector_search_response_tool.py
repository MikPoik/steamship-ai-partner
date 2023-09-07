"""Answers questions with the assistance of a VectorSearch plugin."""
from typing import Any, List, Optional, Union

from steamship import Block, Tag, Task #upm package(steamship)
from steamship.agents.llms import OpenAI #upm package(steamship)
from steamship.agents.schema import AgentContext #upm package(steamship)
from steamship.agents.tools.question_answering.vector_search_tool import VectorSearchTool #upm package(steamship)
from steamship.agents.utils import with_llm #upm package(steamship)
from steamship.utils.repl import ToolREPL #upm package(steamship)

class VectorSearchResponseTool(VectorSearchTool):
    """Tool to answer questions with the assistance of a vector search plugin."""

    name: str = "ResponseHintTool"
    human_description: str = "Answers questions with help from a Vector Database."
    agent_description: str = (
        "Used to answer questions about assistant's role-play character from VectorDatabase "
        "The input should be a plain text question. "
        "The output is a plain text answer."
    )

    load_docs_count: int = 2

    def answer_question(self, question: str, context: AgentContext) -> List[Block]:
        index = self.get_embedding_index(context.client)
        task = index.search(question, k=self.load_docs_count)
        result_items = task.wait()
        blocks = []
        for result in result_items.items:            
            tag = result.tag            
            blocks.append(Block(text=tag.text))
        return blocks


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

        output_str = "\n"
        for input_block in tool_input:
            if not input_block.is_text():
                continue
            for output_block in self.answer_question(input_block.text, context):
                #print("output"+str(output_block))
                output_str += output_block.text + "\n"
            
        return [Block(text=output_str)]


if __name__ == "__main__":
    tool = VectorSearchResponseTool()
    repl = ToolREPL(tool)
    with repl.temporary_workspace() as client:
        index = tool.get_embedding_index(client)
        index.insert(
                        [Tag(text="I love apple pie."), Tag(text="My was to Spain"),Tag(text="my favourite music is reggaeton")],
                        [Tag(text="My last travel was to Spain"),Tag(text="I play the guitar")]

                    )
        index.insert(                        
                        [Tag(text="I like concerts"),Tag(text="I advocate mental health awarness")]                        
                    )        
        repl.run_with_client(
            client, context=with_llm(context=AgentContext(), llm=OpenAI(client=client))
        )
