The following prompts doesn't give any result and falling back to LLM, in the code. 

what extracts are using table VW_PE7QAYD8
which table uses xyz extracts
which source code is used by xyz extracts
what source code is used by xyz extracts
which extract uses the source code  101


can your newly generated code handle below test cases flawlessy?
# Ambiguous numerical input
Test: "123" when both ID and SourceCode contain 123
Assert: Asks for clarification context

# Contextual reference
Test: "Which extracts use 101?" -> Follow-up: "Show me those using 102 in same column"
Assert: Uses previous column context

# Typed column names
Test: "Find extracts with 101 in src"
Assert: Matches "Source Code" column

# Validation failure
Test: "Which extracts use ABC in Source Code?"
Assert: "Column 'Source Code' requires numeric values"

# Similar value suggestion
Test: "Find extracts with 1010 in Source Code" (Actual has 101)
Assert: Suggests "101" as similar value


---------------------------
You: which extracts uses mbr table
Bot: Member uses Table Name: mbr

mbr table extract
which extracts uses mbr table
what extract uses mbr table
which extract uses VWCMAK table



Prompts designed to trigger a Value Search (most common for finding extracts based on Source Code or Table Name):

Find extracts with source code [Source Code Value]. success
Which extract uses table [Table Name Value]?  failed  -- success
Search for [Source Code Value] in source code. success
What extract uses table name [Table Name Value]? success
Extracts using source code [Source Code Value]. success
Give me the extract with table [Table Name Value]. success
Source code [Source Code Value] is used by which extract? success
Table [Table Name Value] is used by which extract? failed
Find the extract for source code [Source Code Value]. success
What extract is linked to table [Table Name Value]? success
Search for value [Source Code Value]. (This will search all columns) failed
Search for value [Table Name Value]. (This will search all columns) failed
Which extract has source code [Source Code Value]? success
Which extract has table name [Table Name Value]? success
Details for source code [Source Code Value]. success

Prompts designed to trigger an Identifier Search (for getting details about a specific extract):

Tell me about the [Extract Name] extract. Failed
Details for [Extract Name]. failed
Show me the [Extract Name] record. failed 
What information do you have on [Extract Name]? Failed
Give me everything on [Extract Name]. failed
What is the source code for [Extract Name]? Failed 
Tell me the table name for [Extract Name]. Failed
Find the source code for the [Extract Name] extract. failed
What table is used by extract [Extract Name]? failed 
Show me the Table Name for [Extract Name]. failed 

Mixed/Varied Prompts (designed to test heuristics and parsing flexibility):

[Extract Name] source code failed 
[Extract Name] table name failed
[Source Code Value] source code success
[Table Name Value] table name success (i was expecting other column details also)
[Extract Name] details failed 

