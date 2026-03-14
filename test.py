from agent.tools.rag_tool import RAGTool
tool = RAGTool("telecom_system_iom____new_radar_system")
result = tool.retrieve("I couldn't find start button")
print(result.successful)
print(result.source_pages)
print(result.answer[:200])