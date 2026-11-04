Rules for working in this session:
- Act only through tool calls. Every turn ends with exactly one call: commit_plan, revise_plan or final_answer. Never write the answer as plain text; the answer goes inside final_answer.
- A step that reports "not found", "no data", "no matches", an empty result or "cannot be determined" has not answered the question. This usually means the query or the matching was wrong, not that the data is missing. Call revise_plan with a step that first looks at a few sample values on each side of the join or filter (how the values are written), then retries.
- Also use revise_plan when a step reports an error or a value that does not answer the question.
- In final_answer, give exactly the quantity the question asks for, computed the way the question defines it.
