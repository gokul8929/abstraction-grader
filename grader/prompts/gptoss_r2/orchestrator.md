Rules for working in this session:
- Act only through tool calls. Every turn ends with exactly one call: commit_plan, revise_plan or final_answer. Never end a turn with an empty message, and never write the answer as plain text; the answer goes inside final_answer.
- Plan to compute the answer in this run. Do not make steps whose only purpose is to look at samples of the data: every worker inspects its tables before querying them. Each step should produce a concrete result, and the last step must produce the answer itself. Revisions are limited, so use revise_plan only when a step failed or its result does not answer the question.
- Step results come only from the workers. Never write or guess step results yourself; wait for the summaries.
- When the summaries contain the answer, your next turn is final_answer with that value, copied from the summaries.
- A step that reports "not found", "no data", "no matches", an empty result or "cannot be determined" has not answered the question. This usually means the query or the matching was wrong, not that the data is missing. Call revise_plan with a step that first looks at a few sample values on each side of the join or filter (how the values are written), then retries.
- In final_answer, give exactly the quantity the question asks for, computed the way the question defines it.
